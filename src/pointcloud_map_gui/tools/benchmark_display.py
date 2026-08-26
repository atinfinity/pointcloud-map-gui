"""Measure how long one height-filter change takes in the 3D view.

    uv run python -m pointcloud_map_gui.tools.benchmark_display [input.pcd] \\
        [--points 5000000] [--budgets 1000000 2000000 0] [--ticks 10]

With no input file it defaults to `sample_data/sample_large_site.pcd`, the
1,554,944-point cloud the README's "Display thinning" numbers were measured on.
Pass a path to measure your own cloud, or `--generate` to build a warehouse
scene of any size in memory: five million points is a 60 MB file, too big to
keep in the repository, but a few seconds to rebuild.

Reports, per display budget, the cost of the two operations a slider drag
triggers: rebuilding the drawn subset (only when the point set changes) and
recoloring it (every tick). Check out an older commit and run it there to
reproduce a before/after comparison: it looks up the height-filter update by
name and prints which one it found, so it works across the versions where
that method was called something else.

Needs a display; it opens the real window because the timings depend on the
live renderer.
"""
import argparse
import os
import threading
import time

from .. import paths
from .generate_sample import build_benchmark_point_cloud


def build_benchmark_cloud(n_points):
    """The cloud the README numbers come from: see generate_sample.py."""
    return build_benchmark_point_cloud(n_points)


def _height_filter_update(window):
    """The work one slider event turns into, on whichever version this is.

    Resolved by name rather than assumed: the method has been renamed twice as
    the drawing path changed, and a probe pinned to one spelling silently
    reports the wrong version instead of failing.
    """
    for name in ("_update_display_buffers", "_update_display_colors"):
        update = getattr(window, name, None)
        if update is not None:
            return name, lambda: (update(), window._request_map_preview())
    legacy = getattr(window, "_refresh_point_cloud_geometry", None)
    if legacy is not None:
        return "_refresh_point_cloud_geometry", legacy
    raise AttributeError("no height-filter update found on MainWindow")


def _time_ticks(window, action, ticks):
    """Fastest of `ticks` height-filter updates, in milliseconds.

    Fastest rather than mean: the slow runs are this process losing the CPU,
    not the code under test getting slower.
    """
    span = window.max_height_slider.double_value - window.min_height_slider.double_value
    base = window.min_height_slider.double_value
    best = float("inf")
    for i in range(ticks):
        # Move the filter every tick so nothing can short-circuit on an
        # unchanged range.
        window.min_height_slider.double_value = base + span * 0.01 * i
        start = time.perf_counter()
        action()
        best = min(best, time.perf_counter() - start)
    window.min_height_slider.double_value = base
    return best * 1000.0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="cloud to measure (default: sample_data/sample_large_site.pcd)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="synthesize the cloud instead of reading the committed one",
    )
    parser.add_argument(
        "--points", type=int, default=5_000_000,
        help="size of the generated cloud (default: %(default)s)",
    )
    parser.add_argument("--write", metavar="PATH", help="write the generated cloud and exit")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1_000_000, 2_000_000, 0],
        help="Max display points values to measure; 0 means no thinning",
    )
    parser.add_argument(
        "--ticks", type=int, default=10,
        help="height-filter updates to time per budget; the fastest is reported "
             "(default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.write:
        import open3d as o3d

        pcd = build_benchmark_cloud(args.points)
        o3d.io.write_point_cloud(args.write, pcd)
        print(f"Wrote {len(pcd.points):,} points to {args.write}")
        return

    import open3d.visualization.gui as gui

    from .. import app
    from ..pointcloud_io import load_point_cloud

    if args.generate:
        pcd = build_benchmark_cloud(args.points)
        label = f"generated warehouse scene ({len(pcd.points):,} points)"
    else:
        path = args.input or str(
            paths.require_sample_data_dir() / "sample_large_site.pcd"
        )
        pcd = load_point_cloud(path)
        label = f"{os.path.basename(path)} ({len(pcd.points):,} points)"

    gui.Application.instance.initialize()
    window = app.MainWindow()
    update_name, update = _height_filter_update(window)
    thins = hasattr(window, "max_display_edit")

    def load():
        print(f"{label}\nheight-filter update: {update_name}\n")
        start = time.perf_counter()
        window._on_load_success(label, pcd)
        print(f"load + first upload : {time.perf_counter() - start:.2f} s\n")

    def measure():
        if not thins:
            # An older build with no display budget to vary: one code path,
            # and it draws every point.
            print(f"{'all points':>18}  {len(pcd.points):>10,}  "
                  f"tick {_time_ticks(window, update, args.ticks):7.1f} ms")
        else:
            print(f"{'budget':>18}  {'drawn':>10}  {'rebuild':>12}  {'tick':>16}")
            for budget in args.budgets:
                window.max_display_edit.int_value = budget
                start = time.perf_counter()
                window._rebuild_display_geometry()
                rebuild = (time.perf_counter() - start) * 1000.0
                tick = _time_ticks(window, update, args.ticks)
                name = "no limit" if budget == 0 else f"{budget:,}"
                print(f"{name:>18}  {window._display_active_count:>10,}  "
                      f"{rebuild:9.1f} ms  {tick:7.1f} ms ({1000.0 / tick:4.1f} fps)")
        gui.Application.instance.quit()

    # The window has to be up and the first frame drawn before any of this
    # means anything, so both steps are posted onto the GUI thread on a timer.
    for delay, step in ((2.0, load), (12.0, measure)):
        timer = threading.Timer(
            delay, lambda s=step: gui.Application.instance.post_to_main_thread(window.window, s)
        )
        timer.daemon = True  # must not outlive the GUI, or teardown races it
        timer.start()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
