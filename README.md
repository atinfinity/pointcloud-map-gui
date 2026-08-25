# Point Cloud -> Occupancy Grid Exporter

A tool that loads a 3D point cloud (PCD/PLY), lets you visualize it and adjust the
height range and resolution in an Open3D GUI, and exports a ROS 2
(`map_server` / `nav2_map_server`) compatible occupancy grid map (PGM + YAML).
It does not depend on ROS 2 itself and runs as a standalone Python application.

![Demo](demo.gif)

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

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
   bounds shown in the panel.
3. (Optional) Enable "Remove ground points" under Ground Removal to detect
   the floor even when it slopes or undulates, and pick a method from the
   dropdown (see [Ground removal](#ground-removal)). Detected ground points
   are grayed out in the 3D view and never become `occupied` in the map.
4. Adjust the height range with the Height Filter's Min/Max sliders (or the
   number fields): points within the range are colored by a height-based
   colormap (blue = low, red = high), and points outside the range are
   rendered translucent/grayed out.
5. Set the output resolution with Occupancy Grid Resolution (m/cell). A live
   preview of the resulting occupancy grid is shown in the bottom-right of
   the panel.
6. Click "Export Occupancy Grid..." and choose a destination base name to
   write `<name>.pgm` and `<name>.yaml`.

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
| `pmf` | Progressive Morphological Filter (Zhang et al. 2003) on the same DEM: the opening window grows from 3 cells up to `max_window` and cells that rise more than a slope-dependent threshold (`slope * dw * cell_size + initial_distance`, capped at `max_distance`) above the opened surface are treated as objects. Handles larger objects sitting on the ground than `local_grid`. | `cell_size`, `max_window`, `slope`, `initial_distance`, `max_distance` |
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
- The output YAML follows the standard `map_server` format, including
  `image` / `resolution` / `origin` / `negate` / `occupied_thresh` /
  `free_thresh`.

## Sample data

```bash
uv run python sample_data/generate_sample.py
```

Generates two synthetic point clouds in `sample_data/`:

- `sample_room`: a flat floor, walls, and a floating obstacle (height
  1.0-1.8m), useful for exercising the height filter.
- `sample_slope`: a floor rising 15% along X with noise and gentle waves, a
  0.3m-tall box standing on it, walls and a floating obstacle. The box top is
  lower than the far end of the floor, so only ground removal can keep the
  box while dropping the floor.

## Tests

The GUI-independent core logic (`occupancy_grid.py`, `map_writer.py`,
`colorize.py`, `ground_grid.py`, `ground_removal.py`, `map_preview.py`) can be run headless with
pytest.

```bash
uv run pytest tests/ -v
```

## Supported formats

- Input: `.pcd`, `.ply` (whatever `Open3D`'s `read_point_cloud` supports)
- Output: `.pgm` (binary P5) + `.yaml` (ROS 2 map_server / nav2 compatible)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
