"""Run every ground-removal method on one point cloud and compare them.

    uv run python -m pointcloud_map_gui.tools.compare_ground_removal \\
        input.pcd --out-dir out/ \\
        [--methods local_grid pmf csf] [--param pmf.slope=0.5 ...] \\
        [--min-height 0.1 --max-height 1.5 --resolution 0.05] [--min-blob-cells 3]

Prints a table of ground point counts / timings and pairwise agreement (IoU)
between methods, and writes per method:
  out/<name>_<method>.ply         ground = gray, other points = height colormap
  out/<name>_<method>.pgm/.yaml   occupancy grid with ground excluded
plus out/<name>_none.pgm/.yaml as the no-removal baseline.
"""
import argparse
import os
import time

import numpy as np

from ..colorize import GRAY_OUT_COLOR, height_colormap_colors
from ..ground_removal import DEFAULT_PARAMS, METHODS, default_params
from ..map_writer import export_map
from ..occupancy_grid import compute_occupancy_grid, remove_small_occupied_blobs


def parse_param_overrides(items):
    overrides = {m: {} for m in METHODS}
    for item in items or []:
        try:
            key, value = item.split("=", 1)
            method, name = key.split(".", 1)
        except ValueError:
            raise SystemExit(f"--param expects method.name=value, got '{item}'")
        if method not in DEFAULT_PARAMS or name not in DEFAULT_PARAMS[method]:
            raise SystemExit(f"Unknown parameter '{key}'")
        overrides[method][name] = float(value)
    return overrides


def iou(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="point cloud to run every method on (.pcd/.ply)")
    parser.add_argument(
        "--out-dir", default="ground_removal_out",
        help="directory for the per-method .ply and map files (default: %(default)s)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
        help="which methods to run (default: all of them)",
    )
    parser.add_argument(
        "--param", action="append", metavar="METHOD.NAME=VALUE",
        help="override one parameter, e.g. pmf.slope=0.5; repeatable",
    )
    parser.add_argument(
        "--min-height", type=float, default=None,
        help="bottom of the height filter in metres (default: the cloud's own minimum)",
    )
    parser.add_argument(
        "--max-height", type=float, default=None,
        help="top of the height filter in metres (default: the cloud's own maximum)",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.05,
        help="occupancy grid resolution in metres per cell (default: %(default)s)",
    )
    parser.add_argument("--min-blob-cells", type=int, default=0, help="map cleanup: drop occupied blobs smaller than this")
    args = parser.parse_args(argv)

    import open3d as o3d

    from ..pointcloud_io import load_point_cloud

    pcd = load_point_cloud(args.input)
    points = np.asarray(pcd.points)
    n = points.shape[0]
    name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    z = points[:, 2]
    min_h = float(z.min()) if args.min_height is None else args.min_height
    max_h = float(z.max()) if args.max_height is None else args.max_height
    overrides = parse_param_overrides(args.param)
    colors = height_colormap_colors(points)

    print(f"{name}: {n:,} points, height filter [{min_h:.3f}, {max_h:.3f}], resolution {args.resolution}")
    baseline = compute_occupancy_grid(points, min_h, max_h, args.resolution)
    baseline.grid = remove_small_occupied_blobs(baseline.grid, args.min_blob_cells)
    export_map(os.path.join(args.out_dir, f"{name}_none"), baseline)
    n_occ_base = int(np.count_nonzero(baseline.grid == 0))
    print(f"baseline (no removal): {n_occ_base:,} occupied cells\n")

    print(f"{'method':<12}{'params':<52}{'ground':>10}{'ratio':>8}{'time[s]':>9}{'occupied':>10}")
    masks = {}
    for method in args.methods:
        params = default_params(method)
        params.update(overrides[method])
        t0 = time.perf_counter()
        try:
            mask = METHODS[method](points, **params)
        except ImportError as e:
            print(f"{method:<12}skipped: {e}")
            continue
        dt = time.perf_counter() - t0
        masks[method] = mask

        result = compute_occupancy_grid(points, min_h, max_h, args.resolution, exclude_mask=mask)
        result.grid = remove_small_occupied_blobs(result.grid, args.min_blob_cells)
        export_map(os.path.join(args.out_dir, f"{name}_{method}"), result)
        n_occ = int(np.count_nonzero(result.grid == 0))

        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(points)
        out_colors = colors.copy()
        out_colors[mask] = GRAY_OUT_COLOR
        out.colors = o3d.utility.Vector3dVector(out_colors)
        o3d.io.write_point_cloud(os.path.join(args.out_dir, f"{name}_{method}.ply"), out)

        param_str = ", ".join(f"{k}={v:g}" for k, v in params.items())
        print(f"{method:<12}{param_str:<52}{int(mask.sum()):>10,}{mask.mean():>8.3f}{dt:>9.2f}{n_occ:>10,}")

    names = list(masks)
    if len(names) > 1:
        print("\npairwise IoU of ground masks:")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                print(f"  {a} vs {b}: {iou(masks[a], masks[b]):.3f}")
    print(f"\noutputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
