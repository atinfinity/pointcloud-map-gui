"""Run every noise-removal method on one point cloud and compare them.

    uv run python compare_noise_removal.py input.pcd --out-dir out/ \\
        [--methods radius statistical voxel_count cluster] \\
        [--param radius.min_neighbors=12 ...] \\
        [--ground-method pmf] [--min-blob-cells 3] \\
        [--min-height 0.1 --max-height 1.5 --resolution 0.05]

Prints removed point counts / timings / resulting map size and occupied cell
count per method, plus pairwise agreement (IoU) between the noise masks, and
writes per method:
  out/<name>_<method>.ply         noise = red, other points = height colormap
  out/<name>_<method>.pgm/.yaml   occupancy grid built from the remaining points
                                  (after ground removal too, if --ground-method)
plus out/<name>_none.pgm/.yaml as the no-removal baseline.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ground_removal  # noqa: E402
import noise_removal  # noqa: E402
from colorize import height_colormap_colors  # noqa: E402
from map_writer import export_map  # noqa: E402
from occupancy_grid import compute_occupancy_grid, remove_small_occupied_blobs  # noqa: E402

NOISE_COLOR = np.array([1.0, 0.0, 0.0])


def parse_param_overrides(items):
    overrides = {m: {} for m in noise_removal.METHODS}
    for item in items or []:
        try:
            key, value = item.split("=", 1)
            method, name = key.split(".", 1)
        except ValueError:
            raise SystemExit(f"--param expects method.name=value, got '{item}'")
        if method not in noise_removal.DEFAULT_PARAMS or name not in noise_removal.DEFAULT_PARAMS[method]:
            raise SystemExit(f"Unknown parameter '{key}'")
        default = noise_removal.DEFAULT_PARAMS[method][name][0]
        overrides[method][name] = int(value) if isinstance(default, int) else float(value)
    return overrides


def iou(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def build_map(points, args):
    exclude = None
    if args.ground_method:
        exclude = ground_removal.estimate_ground_mask(
            points, args.ground_method, **ground_removal.default_params(args.ground_method)
        )
    z = points[:, 2]
    min_h = float(z.min()) if args.min_height is None else args.min_height
    max_h = float(z.max()) if args.max_height is None else args.max_height
    result = compute_occupancy_grid(points, min_h, max_h, args.resolution, exclude_mask=exclude)
    result.grid = remove_small_occupied_blobs(result.grid, args.min_blob_cells)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("--out-dir", default="noise_removal_out")
    parser.add_argument("--methods", nargs="+", default=list(noise_removal.METHODS), choices=list(noise_removal.METHODS))
    parser.add_argument("--param", action="append", metavar="METHOD.NAME=VALUE")
    parser.add_argument("--ground-method", choices=list(ground_removal.METHODS), default=None)
    parser.add_argument("--min-blob-cells", type=int, default=0, help="map cleanup: drop occupied blobs smaller than this")
    parser.add_argument("--min-height", type=float, default=None)
    parser.add_argument("--max-height", type=float, default=None)
    parser.add_argument("--resolution", type=float, default=0.05)
    args = parser.parse_args(argv)

    import open3d as o3d

    from pointcloud_io import load_point_cloud

    points = np.asarray(load_point_cloud(args.input).points)
    n = points.shape[0]
    name = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    overrides = parse_param_overrides(args.param)
    colors = height_colormap_colors(points)

    ground_note = f", ground removal: {args.ground_method}" if args.ground_method else ""
    if args.min_blob_cells > 1:
        ground_note += f", min blob {args.min_blob_cells} cells"
    print(f"{name}: {n:,} points, resolution {args.resolution}{ground_note}")
    baseline = build_map(points, args)
    export_map(os.path.join(args.out_dir, f"{name}_none"), baseline)
    print(
        f"baseline (no removal): map {baseline.width}x{baseline.height}, "
        f"{int(np.count_nonzero(baseline.grid == 0)):,} occupied cells\n"
    )

    print(f"{'method':<13}{'params':<48}{'noise':>9}{'ratio':>8}{'time[s]':>9}{'map size':>11}{'occupied':>10}")
    masks = {}
    for method in args.methods:
        params = noise_removal.default_params(method)
        params.update(overrides[method])
        t0 = time.perf_counter()
        mask = noise_removal.METHODS[method](points, **params)
        dt = time.perf_counter() - t0
        masks[method] = mask

        kept = points[~mask]
        result = build_map(kept, args)
        export_map(os.path.join(args.out_dir, f"{name}_{method}"), result)

        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(points)
        out_colors = colors.copy()
        out_colors[mask] = NOISE_COLOR
        out.colors = o3d.utility.Vector3dVector(out_colors)
        o3d.io.write_point_cloud(os.path.join(args.out_dir, f"{name}_{method}.ply"), out)

        param_str = ", ".join(f"{k}={v:g}" for k, v in params.items())
        print(
            f"{method:<13}{param_str:<48}{int(mask.sum()):>9,}{mask.mean():>8.4f}{dt:>9.2f}"
            f"{f'{result.width}x{result.height}':>11}{int(np.count_nonzero(result.grid == 0)):>10,}"
        )

    names = list(masks)
    if len(names) > 1:
        print("\npairwise IoU of noise masks:")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                print(f"  {a} vs {b}: {iou(masks[a], masks[b]):.3f}")
    print(f"\noutputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
