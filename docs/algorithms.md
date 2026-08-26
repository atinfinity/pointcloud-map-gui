# Algorithms

How the point cloud becomes a map: what each filter does, which method to
pick, and how cells end up `occupied`, `free` or `unknown`. For symptom ->
parameter guidance backed by measurements, see [TUNING.md](TUNING.md).

## Noise removal

Isolated points (floating reflections, dust) would otherwise become spurious
`occupied`/known cells and stretch the map bounds. Noise removal runs first
and simply deletes those points. Four methods are implemented so they can be
compared on real data (`pointcloud_map_gui/noise_removal.py`, all share the same
`points -> (N,) bool` interface, True = remove):

| Method | Idea | Parameters |
|---|---|---|
| `cluster` (default) | DBSCAN (`eps`, `min_points`); unlabeled points and clusters smaller than `min_cluster_size` -> noise. The only method that also removes small dense clumps (10-40 points), which the per-point tests below keep and which then appear as specks on the map. Raise `min_cluster_size` if specks remain, lower it if thin real structures disappear. Slowest (DBSCAN). | `eps`, `min_points`, `min_cluster_size` |
| `radius` | Fewer than `min_neighbors` points within `radius` -> noise (`remove_radius_outlier`). The literal definition of "isolated". | `radius`, `min_neighbors` |
| `statistical` | Mean distance to `nb_neighbors` nearest points exceeds the cloud-wide mean + `std_ratio` x std -> noise (`remove_statistical_outlier`). Adapts to point density. | `nb_neighbors`, `std_ratio` |
| `voxel_count` | Voxel containing fewer than `min_points` points -> noise. Fastest, pure numpy, but grid-aligned so it can clip thin structures. | `voxel_size`, `min_points` |

All of these except `voxel_count` are Open3D calls that hold the GIL for their
whole duration -- 3.0 s for `cluster` on a 1.5-million-point cloud. A worker
thread cannot help with that: while one runs, no other Python code executes, so
the window stops handling the mouse. Estimation therefore runs in a separate
process (`pointcloud_map_gui/noise_worker.py`), started at launch and fed the cloud through shared
memory, which keeps the GUI at a 5 ms worst-case stall instead of 2.9 s. If the
process cannot be started the work falls back to a thread, which is correct but
freezes the window while it runs.

To compare them on one file outside the GUI (optionally chaining ground
removal so the maps reflect the full pipeline):

```bash
uv run python -m pointcloud_map_gui.tools.compare_noise_removal path/to/cloud.pcd --out-dir out/ \
    --min-height 0.1 --max-height 1.5 --ground-method pmf --min-blob-cells 3
```

This prints removed point count, ratio, run time, resulting map size and
occupied cell count per method plus pairwise IoU between the noise masks, and
writes `out/<name>_<method>.ply` (noise = red) and
`out/<name>_<method>.pgm/.yaml` (plus a `_none` baseline).

## Ground removal

A plain height filter cannot separate the floor from low obstacles when the
floor itself is not flat (slopes, ramps, uneven ground). Ground Removal
estimates the local floor height at every point and marks points within a
small distance of it as ground. Three methods are implemented so they can be
compared on real data (`pointcloud_map_gui/ground_removal.py`, all share the same
`points -> (N,) bool` interface):

| Method | Idea | Parameters |
|---|---|---|
| `local_grid` | Minimum Z per XY cell -> 3x3 morphological opening -> bilinear interpolation. Points within `thickness` of that surface are ground. Fast and simple; on steep slopes the per-cell minimum sits below the cell centre, so raise `thickness` or lower `cell_size`. | `cell_size`, `thickness` |
| `pmf` (default) | Progressive Morphological Filter (Zhang et al. 2003) on the same DEM: the opening window grows from 3 cells up to `max_window` and cells that rise more than a slope-dependent threshold (`slope * dw * cell_size + initial_distance`, capped at `max_distance`) above the opened surface are treated as objects. Handles larger objects sitting on the ground than `local_grid`. | `cell_size`, `max_window`, `slope`, `initial_distance`, `max_distance` |
| `csf` | Cloth Simulation Filter (Zhang et al. 2016) via the `cloth-simulation-filter` package: a cloth is dropped onto the inverted cloud and points within `class_threshold` of it are ground. `rigidness` 1 = steep terrain, 3 = flat; `slope_smooth` helps on ramps. | `cloth_resolution`, `rigidness`, `class_threshold`, `slope_smooth` |

To compare the methods on one file outside the GUI:

```bash
uv run python -m pointcloud_map_gui.tools.compare_ground_removal path/to/cloud.pcd --out-dir out/ \
    --min-height 0.1 --max-height 1.5 --param pmf.slope=0.5
```

This prints the ground point count, ratio, run time and resulting occupied
cell count per method plus pairwise IoU between the ground masks, and writes
`out/<name>_<method>.ply` (ground = gray, other points = height colormap) and
`out/<name>_<method>.pgm/.yaml` (plus a `_none` baseline) for visual
comparison.

## Occupancy grid generation logic

- The grid origin and size are derived automatically from the XY bounds of
  the entire point cloud (before the height filter is applied).
- A cell is considered "known" (scanned) if any point at all -- regardless of
  height -- falls inside it.
- Among known cells, one containing a point within `[min_height, max_height]`
  that is not a detected ground point is `occupied` (pixel value 0); a known cell with no point in range is
  `free` (pixel value 254); a cell that never contains any point is
  `unknown` (pixel value 205).
- Map cleanup (if enabled) then turns occupied blobs smaller than the
  configured cell count into `free`.
- The output YAML follows the standard `map_server` format, including
  `image` / `resolution` / `origin` / `negate` / `occupied_thresh` /
  `free_thresh`.
