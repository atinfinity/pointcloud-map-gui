"""Run noise estimation in a separate process, so the GUI keeps responding.

Open3D's `cluster_dbscan`, `remove_radius_outlier` and `remove_statistical_outlier`
hold the GIL for the whole call -- measured at 3.0 s, 1.0 s and 1.0 s on a
1.5-million-point cloud. A worker *thread* is therefore useless for them: while
one runs, no other Python code executes, so the GUI stops handling the mouse and
stops drawing frames. Only a separate process gets around that. The ground
removal methods release the GIL (5-48 ms) and stay on a thread.

The process is spawned rather than forked. Forking a process that has already
brought up the GUI inherits whatever locks Open3D's threads happened to hold,
which can deadlock the child; spawning costs 0.01 s here and works on platforms
without fork. Spawn resolves the child entry point by module and function name,
so `_serve` lives in this module and importing it must stay free of side
effects -- in particular it must never start the GUI.
"""
import multiprocessing as mp
import os
import sys
import threading
import time

import numpy as np

_READY = "ready"
_SHUTDOWN = None
_STOP_TIMEOUT = 5.0


def _serve(conn, repo_path):
    """Child process: answer estimation jobs until told to stop."""
    sys.path.insert(0, repo_path)

    from multiprocessing import shared_memory

    import numpy as np

    import noise_removal

    # noise_removal imports Open3D lazily, inside the methods that need it.
    # Paying that here means the first job costs what every later one does
    # (measured: 1.6 s), spent while the user is still opening a file.
    import open3d  # noqa: F401

    conn.send(_READY)
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            # The parent went away without asking us to stop -- killed, or
            # Ctrl+C. Leave quietly instead of printing a traceback over
            # whatever the user was actually looking at.
            return
        if job is _SHUTDOWN:
            return
        shm_name, shape, dtype, method, params = job
        block = None
        try:
            block = shared_memory.SharedMemory(name=shm_name)
            points = np.ndarray(shape, dtype=np.dtype(dtype), buffer=block.buf)
            mask = noise_removal.estimate_noise_mask(points, method, **params)
            # Bool arrays travel as one byte per point; packed, a 1.5M-point
            # mask is 194 KB instead of 1.5 MB.
            conn.send(("ok", np.packbits(mask)))
        except Exception as exc:  # noqa: BLE001 - reported, never raised here
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            if block is not None:
                block.close()


class LatestOnly:
    """Keeps one job running and at most one more waiting -- always the newest.

    Dragging a parameter fires a request per step. Running every one of them
    would queue minutes of work to arrive at a single answer nobody waited for,
    so anything superseded before it starts is simply dropped.
    """

    def __init__(self):
        self._running = False
        self._pending = None

    @property
    def running(self):
        return self._running

    def submit(self, job):
        """Return the job to start now, or None if it has to wait its turn."""
        if self._running:
            self._pending = job
            return None
        self._running = True
        return job

    def finish(self):
        """Mark the running job done; return the next one, or None."""
        job, self._pending = self._pending, None
        self._running = job is not None
        return job


class NoiseWorker:
    """Client for the estimation process, with a thread as the fallback.

    `submit` answers through `on_done(mask, elapsed, error)` on some background
    thread -- never the caller's -- so the caller is responsible for getting
    back onto the GUI thread.
    """

    def __init__(self):
        self._process = None
        self._conn = None
        self._reader = None
        self._lock = threading.Lock()
        self._queue = LatestOnly()
        self._current = None  # (on_done, n_points, block, started_at)
        self._failure = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def available(self):
        return self._process is not None and self._process.is_alive()

    @property
    def failure(self):
        """Why the process is not running, or None if it is."""
        return self._failure

    def start(self):
        """Launch the process. Returns False if it could not be launched at
        all, which leaves every later submit on the thread fallback.

        Does not wait for the child to finish starting. The child re-imports
        this program to reach `_serve`, which pulls in Open3D and costs ~1.6 s;
        blocking on that would delay the window by as much. Jobs sent before it
        is ready simply sit in the pipe, and the reader thread takes the ready
        message off the front.
        """
        conn = None
        try:
            context = mp.get_context("spawn")
            conn, child = context.Pipe()
            process = context.Process(
                target=_serve,
                args=(child, os.path.dirname(os.path.abspath(__file__))),
                daemon=True,
            )
            process.start()
            # Published only now: an unstarted Process cannot be joined, so
            # letting the failure path see one would raise over the real error.
            self._process = process
            self._conn = conn
            child.close()  # the parent must not hold the child's end open
        except Exception as exc:  # noqa: BLE001
            self._failure = f"{type(exc).__name__}: {exc}"
            self._shutdown_process()
            if self._conn is None and conn is not None:
                conn.close()
            return False
        self._reader = threading.Thread(target=self._read_results, daemon=True)
        self._reader.start()
        return True

    def close(self):
        with self._lock:
            conn = self._conn
        if conn is not None:
            try:
                conn.send(_SHUTDOWN)
            except (OSError, BrokenPipeError, ValueError):
                pass
        self._shutdown_process()

    def _shutdown_process(self):
        process, self._process = self._process, None
        conn, self._conn = self._conn, None
        if process is not None:
            process.join(_STOP_TIMEOUT)
            if process.is_alive():
                process.terminate()
                process.join(_STOP_TIMEOUT)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    def submit(self, points, method, params, on_done):
        if not self.available:
            self._run_on_thread(points, method, params, on_done)
            return
        job = (points, method, params, on_done)
        with self._lock:
            job = self._queue.submit(job)
        if job is not None:
            self._send(job)

    def _send(self, job):
        points, method, params, on_done = job
        try:
            block = _share(points)
            with self._lock:
                self._current = (on_done, points.shape[0], block, time.perf_counter())
                self._conn.send((block.name, points.shape, points.dtype.str, method, params))
        except Exception as exc:  # noqa: BLE001
            # The process died between the availability check and the send.
            self._finish_current(None, f"{type(exc).__name__}: {exc}")

    def _read_results(self):
        """Blocks on the pipe, which releases the GIL -- the whole point."""
        try:
            first = self._conn.recv()
        except (EOFError, OSError, ValueError) as exc:
            self._fail(f"worker exited during startup: {type(exc).__name__}")
            return
        if first != _READY:
            self._fail(f"worker sent {first!r} instead of a ready message")
            return
        while True:
            try:
                status, payload = self._conn.recv()
            except (EOFError, OSError, ValueError):
                self._fail("noise removal process stopped")
                return
            if status == "ok":
                self._finish_current(payload, None)
            else:
                self._finish_current(None, payload)

    def _fail(self, reason):
        """The process is gone: report whatever was in flight and make sure
        later jobs take the thread instead."""
        self._failure = reason
        self._shutdown_process()
        self._finish_current(None, reason)

    def _finish_current(self, packed, error):
        with self._lock:
            current, self._current = self._current, None
            following = self._queue.finish()
        if current is not None:
            on_done, n_points, block, started_at = current
            _release(block)
            mask = None if packed is None else np.unpackbits(packed, count=n_points).astype(bool)
            on_done(mask, time.perf_counter() - started_at, error)
        if following is not None:
            self._send(following)

    def _run_on_thread(self, points, method, params, on_done):
        """Fallback when there is no process: correct, but it freezes the GUI
        for as long as Open3D holds the GIL."""

        def work():
            import noise_removal

            started_at = time.perf_counter()
            try:
                mask = noise_removal.estimate_noise_mask(points, method, **params)
                error = None
            except Exception as exc:  # noqa: BLE001
                mask, error = None, f"{type(exc).__name__}: {exc}"
            on_done(mask, time.perf_counter() - started_at, error)

        threading.Thread(target=work, daemon=True).start()


def _share(points):
    """Copy `points` into a shared block the child can map.

    A fresh block per job rather than one kept across them: the copy costs
    24 ms against a job of seconds, and reusing one would mean tracking which
    jobs still have it mapped before it could be rewritten.
    """
    from multiprocessing import shared_memory

    points = np.ascontiguousarray(points)
    block = shared_memory.SharedMemory(create=True, size=max(points.nbytes, 1))
    np.ndarray(points.shape, dtype=points.dtype, buffer=block.buf)[:] = points
    return block


def _release(block):
    if block is None:
        return
    try:
        block.close()
        block.unlink()
    except (FileNotFoundError, OSError):
        pass
