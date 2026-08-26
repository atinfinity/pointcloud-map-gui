import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import noise_removal
from noise_worker import LatestOnly, NoiseWorker


def _cloud(n=4000, seed=0):
    """A dense blob plus a scattering of strays, so every method finds noise."""
    rng = np.random.default_rng(seed)
    dense = rng.normal(0.0, 0.3, size=(n, 3))
    strays = rng.uniform(-6.0, 6.0, size=(40, 3))
    return np.ascontiguousarray(np.vstack([dense, strays]))


# ----------------------------------------------------------------------
# Job queueing
# ----------------------------------------------------------------------
def test_first_job_starts_immediately():
    queue = LatestOnly()
    assert queue.submit("a") == "a"
    assert queue.running


def test_only_the_newest_waiting_job_survives():
    queue = LatestOnly()
    queue.submit("a")
    assert queue.submit("b") is None
    assert queue.submit("c") is None
    assert queue.submit("d") is None
    # b and c were superseded before they ever ran.
    assert queue.finish() == "d"
    assert queue.finish() is None
    assert not queue.running


def test_finish_with_nothing_waiting_goes_idle():
    queue = LatestOnly()
    queue.submit("a")
    assert queue.finish() is None
    assert not queue.running
    assert queue.submit("b") == "b"


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
def _collect(worker, points, method, timeout=120.0):
    done = threading.Event()
    box = {}

    def on_done(mask, elapsed, error):
        box["mask"], box["elapsed"], box["error"] = mask, elapsed, error
        done.set()

    worker.submit(points, method, noise_removal.default_params(method), on_done)
    assert done.wait(timeout), f"{method} never reported back"
    return box


@pytest.mark.parametrize("method", list(noise_removal.DEFAULT_PARAMS))
def test_process_result_matches_running_it_here(method):
    points = _cloud()
    expected = noise_removal.estimate_noise_mask(points, method, **noise_removal.default_params(method))
    worker = NoiseWorker()
    if not worker.start():
        pytest.skip("no worker process available on this platform")
    try:
        got = _collect(worker, points, method)
    finally:
        worker.close()
    assert got["error"] is None
    assert np.array_equal(got["mask"], expected)
    assert got["elapsed"] > 0.0


def test_fallback_matches_when_no_process_was_started():
    """An unstarted worker must still answer -- on a thread, correctly."""
    points = _cloud()
    method = "voxel_count"
    expected = noise_removal.estimate_noise_mask(points, method, **noise_removal.default_params(method))
    worker = NoiseWorker()
    assert not worker.available
    got = _collect(worker, points, method)
    assert got["error"] is None
    assert np.array_equal(got["mask"], expected)


def test_a_failing_job_is_reported_not_raised():
    """A bad parameter must come back as an error the GUI can show, and must
    leave the worker able to take the next job."""
    points = _cloud(500)
    worker = NoiseWorker()
    if not worker.start():
        pytest.skip("no worker process available on this platform")
    try:
        done = threading.Event()
        box = {}

        def on_done(mask, elapsed, error):
            box["mask"], box["error"] = mask, error
            done.set()

        worker.submit(points, "not_a_method", {}, on_done)
        assert done.wait(60.0)
        assert box["mask"] is None
        assert box["error"] and "not_a_method" in box["error"]

        # Still usable afterwards.
        good = _collect(worker, points, "voxel_count")
        assert good["error"] is None
        assert good["mask"] is not None
    finally:
        worker.close()


def test_close_is_safe_to_call_twice_and_without_starting():
    worker = NoiseWorker()
    worker.close()
    worker.close()
    assert not worker.available


def test_start_failure_is_reported_not_raised(monkeypatch):
    """A process that cannot start must leave a usable worker behind, not an
    exception -- an unstarted Process cannot even be joined."""
    import multiprocessing as mp

    class Refuses:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("no processes here")

    context = mp.get_context("spawn")
    monkeypatch.setattr(context, "Process", Refuses)
    monkeypatch.setattr(mp, "get_context", lambda *_a, **_k: context)

    worker = NoiseWorker()
    assert worker.start() is False
    assert not worker.available
    assert "no processes here" in worker.failure
    worker.close()

    # And it still answers, on the fallback path.
    points = _cloud(500)
    got = _collect(worker, points, "voxel_count")
    assert got["error"] is None
    assert got["mask"] is not None
