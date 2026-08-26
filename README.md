# Point Cloud -> Occupancy Grid Exporter

A tool that loads a 3D point cloud (PCD/PLY), lets you visualize it and adjust the
height range and resolution in an Open3D GUI, and exports a ROS 2
(`map_server` / `nav2_map_server`) compatible occupancy grid map (PGM + YAML).
It does not depend on ROS 2 itself and runs as a standalone Python application.

![Demo](demo.gif)

## Setup

Requires Python 3.10-3.12. Open3D 0.19.0 (its latest release) publishes wheels
only up to CPython 3.12 and ships no sdist, so 3.13+ cannot install it.
Dependencies are managed with [uv](https://docs.astral.sh/uv/), which reads the
pinned version from `.python-version` and downloads it automatically -- so this
works even if your system Python is newer.

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

## Usage

1. Click "Load Point Cloud (.pcd/.ply)..." and select a point cloud file.
2. The point cloud is displayed in the 3D view, with point count and XYZ
   bounds shown in the panel. Clouds larger than "Max display points" are
   thinned for drawing only (see [Display thinning](#display-thinning)); the
   panel then also reports how many points are being shown.
3. (Optional) Enable "Remove isolated points" under Noise Removal to drop
   stray points before anything else happens (see
   [Noise removal](#noise-removal)). Removed points are shown in faded red
   and are excluded from the map entirely -- they do not count as scanned
   cells and do not enlarge the map bounds.
4. (Optional) Enable "Remove ground points" under Ground Removal to detect
   the floor even when it slopes or undulates, and pick a method from the
   dropdown (see [Ground removal](#ground-removal)). Detected ground points
   are grayed out in the 3D view and never become `occupied` in the map.
5. Adjust the height range with the Height Filter's Min/Max sliders (or the
   number fields): points within the range are colored by a height-based
   colormap (blue = low, red = high), and points outside the range are
   faded into the background.
6. Set the output resolution with Occupancy Grid Resolution (m/cell). A live
   preview of the resulting occupancy grid is overlaid on the top-right of
   the 3D view.
   (Optional) Set "Map Cleanup: min occupied blob (cells)" to 2-4 to erase
   specks: 8-connected groups of `occupied` cells smaller than that are
   turned `free`. 0 or 1 disables it. This catches leftovers no point-level
   filter can (clumps touching a wall, ground-removal misses), but also
   erases real obstacles that small -- keep it just above the speck size.
7. Click "Export Occupancy Grid..." and choose a destination base name to
   write `<name>.pgm` and `<name>.yaml`.

Processing order: noise removal -> ground removal -> height filter ->
occupancy grid -> map cleanup.

Not sure which method or parameter to change? See [TUNING.md](TUNING.md)
for symptom -> method -> parameter guidance backed by measurements on the
sample data.

## Display thinning

Drawing every point of a multi-million-point cloud costs far more than it
shows -- the screen cannot resolve that many points anyway -- so the 3D view
is capped at "Max display points" (default 1,000,000). Above that, points are
voxel-subsampled for drawing: one point per voxel, which keeps a uniform
spatial density rather than leaving sparse regions sparser the way random
thinning would.

**This only affects what is drawn.** The occupancy grid, its preview, and the
exported map are always computed from every point, so thinning costs nothing
in map fidelity. Raise the limit to inspect fine detail, or set it to 0 to
disable thinning entirely.

Measured on `sample_data/sample_large_site.pcd` (1,554,944 points), dragging a
height slider:

| Max display points | Points drawn | Height-filter update |
|---|---:|---:|
| 500,000 | 474,242 | 7.0 ms (144 fps) |
| 1,000,000 (default) | 491,181 | 7.1 ms (142 fps) |
| 0 (no limit) | 1,554,944 | 21.1 ms (47 fps) |

Before the geometry was made updatable in place, the same drag cost 163 ms per
tick (6.1 fps) on that cloud, and 786 ms (1.3 fps) on a five-million-point one.

Fewer points are drawn than the limit allows because the voxel grid only
divides so finely before its lookup table would outgrow the memory budget; the
panel always reports the count actually being drawn.

Reproduce with:

```bash
uv run python benchmark_display.py                              # the committed cloud
uv run python benchmark_display.py --generate --points 5000000  # a generated one
```

Check out the commit before this change and run the same script to get the
"before" column; it detects which version of the code it is on.

## Noise removal

Isolated points (floating reflections, dust) would otherwise become spurious
`occupied`/known cells and stretch the map bounds. Noise removal runs first
and simply deletes those points. Four methods are implemented so they can be
compared on real data (`noise_removal.py`, all share the same
`points -> (N,) bool` interface, True = remove):

| Method | Idea | Parameters |
|---|---|---|
| `cluster` (default) | DBSCAN (`eps`, `min_points`); unlabeled points and clusters smaller than `min_cluster_size` -> noise. The only method that also removes small dense clumps (10-40 points), which the per-point tests below keep and which then appear as specks on the map. Raise `min_cluster_size` if specks remain, lower it if thin real structures disappear. Slowest (DBSCAN). | `eps`, `min_points`, `min_cluster_size` |
| `radius` | Fewer than `min_neighbors` points within `radius` -> noise (`remove_radius_outlier`). The literal definition of "isolated". | `radius`, `min_neighbors` |
| `statistical` | Mean distance to `nb_neighbors` nearest points exceeds the cloud-wide mean + `std_ratio` x std -> noise (`remove_statistical_outlier`). Adapts to point density. | `nb_neighbors`, `std_ratio` |
| `voxel_count` | Voxel containing fewer than `min_points` points -> noise. Fastest, pure numpy, but grid-aligned so it can clip thin structures. | `voxel_size`, `min_points` |

To compare them on one file outside the GUI (optionally chaining ground
removal so the maps reflect the full pipeline):

```bash
uv run python compare_noise_removal.py path/to/cloud.pcd --out-dir out/ \
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
compared on real data (`ground_removal.py`, all share the same
`points -> (N,) bool` interface):

| Method | Idea | Parameters |
|---|---|---|
| `local_grid` | Minimum Z per XY cell -> 3x3 morphological opening -> bilinear interpolation. Points within `thickness` of that surface are ground. Fast and simple; on steep slopes the per-cell minimum sits below the cell centre, so raise `thickness` or lower `cell_size`. | `cell_size`, `thickness` |
| `pmf` (default) | Progressive Morphological Filter (Zhang et al. 2003) on the same DEM: the opening window grows from 3 cells up to `max_window` and cells that rise more than a slope-dependent threshold (`slope * dw * cell_size + initial_distance`, capped at `max_distance`) above the opened surface are treated as objects. Handles larger objects sitting on the ground than `local_grid`. | `cell_size`, `max_window`, `slope`, `initial_distance`, `max_distance` |
| `csf` | Cloth Simulation Filter (Zhang et al. 2016) via the `cloth-simulation-filter` package: a cloth is dropped onto the inverted cloud and points within `class_threshold` of it are ground. `rigidness` 1 = steep terrain, 3 = flat; `slope_smooth` helps on ramps. | `cloth_resolution`, `rigidness`, `class_threshold`, `slope_smooth` |

To compare the methods on one file outside the GUI:

```bash
uv run python compare_ground_removal.py path/to/cloud.pcd --out-dir out/ \
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

## Sample data

```bash
uv run python sample_data/generate_sample.py
```

Generates five synthetic point clouds in `sample_data/`. The first four are
small scenes for exercising one feature at a time, written as both `.pcd` and
`.ply`; the last is large and written as `.pcd` only, without colors (the GUI
colors points by height and ignores any the file carries).

- `sample_room`: a flat floor, walls, and a floating obstacle (height
  1.0-1.8m), useful for exercising the height filter.
- `sample_slope`: a floor rising 15% along X with noise and gentle waves, a
  0.3m-tall box standing on it, walls and a floating obstacle. The box top is
  lower than the far end of the floor, so only ground removal can keep the
  box while dropping the floor.
- `sample_slope_noisy`: `sample_slope` plus 500 uniformly scattered noise
  points, 30 tiny clusters of 2-5 points and 20 speck clumps of 10-40
  points, for exercising noise removal.
- `sample_ramp`: flat and sloped floor mixed -- a flat lower level (z=0), a
  35% ramp rising 0.7m, and a flat upper level (z=0.7) -- with a 0.3m box on
  each level and one on the ramp, plus a floating obstacle. The lower box's
  top is below the upper floor, and the flat/slope breaklines are where
  DEM-based ground removal is most likely to misclassify.
- `sample_large_site`: a 50 x 50 m outdoor yard at 1,554,944 points -- survey
  scale, sparser than an indoor scan but spread over 25x the area. A perimeter
  fence, three buildings, three shipping containers and gently undulating
  ground (+/-0.05m, enough to give ground removal something to do). Everything
  in it is there so the map can be checked by eye: two buildings were entered
  through their doorways and come out hollow and `free`; the third was only
  seen from outside and comes out `unknown` inside its `occupied` walls; two
  patches of ground sit in a building's occlusion shadow, and nothing sees the
  ground under a container, so those come out `unknown` too. Its ground is
  sampled at 0.045 m, just under the 0.05 m/cell default resolution -- coarser
  than the grid and the map fills with `unknown` speckle.

`benchmark_display.py --generate --points N` builds a sixth, a warehouse floor
plan of any size, without writing it to disk; `--write PATH` saves one. It is
what the five-million-point figures above come from, and is not committed
because at that size the file is 60 MB.

## Tests

The GUI-independent core logic (`occupancy_grid.py`, `map_writer.py`,
`colorize.py`, `ground_grid.py`, `ground_removal.py`, `noise_removal.py`,
`map_preview.py`) can be run headless with
pytest.

```bash
uv run pytest tests/ -v
```

## Supported formats

- Input: `.pcd`, `.ply` (whatever `Open3D`'s `read_point_cloud` supports)
- Output: `.pgm` (binary P5) + `.yaml` (ROS 2 map_server / nav2 compatible)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
