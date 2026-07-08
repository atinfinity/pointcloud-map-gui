# Point Cloud -> Occupancy Grid Exporter

A tool that loads a 3D point cloud (PCD/PLY), lets you visualize it and adjust the
height range and resolution in an Open3D GUI, and exports a ROS 2
(`map_server` / `nav2_map_server`) compatible occupancy grid map (PGM + YAML).
It does not depend on ROS 2 itself and runs as a standalone Python application.

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
3. Adjust the height range with the Height Filter's Min/Max sliders (or the
   number fields): points within the range are colored by a height-based
   colormap (blue = low, red = high), and points outside the range are
   rendered translucent/grayed out.
4. Set the output resolution with Occupancy Grid Resolution (m/cell). A live
   preview of the resulting occupancy grid is shown in the bottom-right of
   the panel.
5. Click "Export Occupancy Grid..." and choose a destination base name to
   write `<name>.pgm` and `<name>.yaml`.

## Occupancy grid generation logic

- The grid origin and size are derived automatically from the XY bounds of
  the entire point cloud (before the height filter is applied).
- A cell is considered "known" (scanned) if any point at all -- regardless of
  height -- falls inside it.
- Among known cells, one containing a point within `[min_height, max_height]`
  is `occupied` (pixel value 0); a known cell with no point in range is
  `free` (pixel value 254); a cell that never contains any point is
  `unknown` (pixel value 205).
- The output YAML follows the standard `map_server` format, including
  `image` / `resolution` / `origin` / `negate` / `occupied_thresh` /
  `free_thresh`.

## Sample data

```bash
uv run python sample_data/generate_sample.py
```

Generates a synthetic point cloud in `sample_data/` consisting of a floor,
walls, and a floating obstacle (height 1.0-1.8m), useful for exercising the
height filter.

## Tests

The GUI-independent core logic (`occupancy_grid.py`, `map_writer.py`,
`colorize.py`, `ground_grid.py`, `map_preview.py`) can be run headless with
pytest.

```bash
uv run pytest tests/ -v
```

## Supported formats

- Input: `.pcd`, `.ply` (whatever `Open3D`'s `read_point_cloud` supports)
- Output: `.pgm` (binary P5) + `.yaml` (ROS 2 map_server / nav2 compatible)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
