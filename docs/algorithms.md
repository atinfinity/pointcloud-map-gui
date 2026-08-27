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
| `scatter` | Neighbourhood that fills a volume rather than covering a surface -> noise, if enough of its neighbours agree. For the haze a moving object leaves in a SLAM map. The only method here that does not count neighbours, and the only one whose accuracy holds as that haze gets denser. | `knn`, `max_scatter`, `agreement` |

All of these except `voxel_count` are Open3D calls that hold the GIL for their
whole duration -- 3.0 s for `cluster` on a 1.5-million-point cloud. A worker
thread cannot help with that: while one runs, no other Python code executes, so
the window stops handling the mouse. Estimation therefore runs in a separate
process (`pointcloud_map_gui/noise_worker.py`), started at launch and fed the cloud through shared
memory, which keeps the GUI at a 5 ms worst-case stall instead of 2.9 s. If the
process cannot be started the work falls back to a thread, which is correct but
freezes the window while it runs.

### Which method finds haze depends on how dense it is

A moving object leaves points spread through the volume it moved through. How
many depends on how long it was in view, and that decides which method can find
it. Recall against the same haze at a range of densities, on the geometry of
`sample_haze`, whose structure has 21 neighbours within 10 cm:

| Haze density | `radius` | `statistical` | `cluster` | `scatter` |
|---:|---:|---:|---:|---:|
| 1 | **0.99** | 0.98 | 0.98 | 0.80 |
| 2 | **0.98** | 0.96 | 0.98 | 0.88 |
| 3 | **0.96** | 0.74 | 0.62 | 0.91 |
| 5 | 0.75 | 0.41 | 0.29 | **0.93** |
| 10 | 0.43 | 0.21 | 0.12 | **0.95** |
| 18 | 0.23 | 0.12 | 0.06 | **0.96** |
| 34 | 0.11 | 0.07 | 0.03 | **0.97** |

Thin haze is a density problem and `radius` solves it outright. From about a
quarter of the surrounding density upward, counting neighbours stops working --
by the time the haze matches the walls `radius` finds one point in ten -- while
`scatter` barely moves, because a volume does not take on the shape of a
surface however many points are in it.

Measure yours before choosing:

```bash
uv run python -c "
import numpy as np, sys
from scipy.spatial import cKDTree
from pointcloud_map_gui.pointcloud_io import load_point_cloud
p = np.asarray(load_point_cloud(sys.argv[1]).points)
d = cKDTree(p).query_ball_point(p, 0.1, workers=-1, return_length=True)
for q in (10, 50, 90):
    print(q, np.percentile(d, q))
" your_cloud.pcd
```

### How `scatter` decides

Take the covariance of a point's `knn` neighbours and compare its smallest
eigenvalue to its largest. On a surface the points are thin in the normal
direction, so the ratio is near zero; a volume spreads in all three, so it is
not. On `sample_haze` the two populations sit two orders of magnitude apart --
0.006 for the walls against 0.507 for the haze -- so where `max_scatter` falls
between them hardly matters.

`agreement` is the part that needs care. Where two surfaces meet, the
neighbourhood spans both and scatters like a volume, so a corner or an edge
reads as haze. Only in a thin band, though: the neighbours of a corner point
are mostly still flat. Requiring that a fraction of them scatter too keeps
corners and edges -- on `sample_haze` it takes the structure wrongly removed
from 1,289 points down to 80 -- and costs almost nothing, because their
neighbourhoods have already been found.

Two things follow from that, and neither is a tuning problem:

- **Vegetation, mesh fences and hanging cables are volumes too.** This cannot
  tell them from haze. Bound it by height or region if the cloud has any.
- **A thin scattering of points does not agree with itself**, so `scatter`
  will not remove it. That is `radius`'s and `cluster`'s job; pair them.

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
