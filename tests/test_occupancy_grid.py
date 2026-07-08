import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from occupancy_grid import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    compute_occupancy_grid,
)
from map_writer import export_map


def make_synthetic_cloud():
    # Floor: a 4x4m flat plane at z=0, sampled on a fine grid -> "known+free"
    xs, ys = np.meshgrid(np.arange(0, 4, 0.05), np.arange(0, 4, 0.05))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)

    # Wall: a vertical strip at x=2, z from 0 to 2m -> "occupied" when height filter includes it
    zs = np.arange(0, 2, 0.05)
    wall_y = np.arange(0, 4, 0.05)
    wall_zz, wall_yy = np.meshgrid(zs, wall_y)
    wall = np.stack(
        [np.full(wall_zz.size, 2.0), wall_yy.ravel(), wall_zz.ravel()], axis=1
    )

    return np.vstack([floor, wall])


def test_wall_is_occupied_floor_is_free():
    points = make_synthetic_cloud()
    result = compute_occupancy_grid(points, min_height=0.1, max_height=1.5, resolution=0.1)

    # A cell near the wall (x=2, y=2) should be occupied.
    col = int((2.0 - result.origin_x) / result.resolution)
    row_from_bottom = int((2.0 - result.origin_y) / result.resolution)
    image_row = result.height - 1 - row_from_bottom
    assert result.grid[image_row, col] == OCCUPIED_VALUE

    # A floor-only cell far from the wall (x=0.5, y=0.5) should be free.
    col2 = int((0.5 - result.origin_x) / result.resolution)
    row2_from_bottom = int((0.5 - result.origin_y) / result.resolution)
    image_row2 = result.height - 1 - row2_from_bottom
    assert result.grid[image_row2, col2] == FREE_VALUE


def test_unscanned_hole_is_unknown():
    # A ring-shaped floor (with a hole in the middle) so the grid's bounding
    # box includes a region no point ever touches -> that cell must be "unknown".
    xs, ys = np.meshgrid(np.arange(0, 4, 0.05), np.arange(0, 4, 0.05))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    center = np.array([2.0, 2.0])
    dist = np.linalg.norm(pts[:, :2] - center, axis=1)
    ring = pts[(dist > 0.5)]  # remove points within 0.5m of center -> hole

    result = compute_occupancy_grid(ring, min_height=0.1, max_height=1.5, resolution=0.1)

    col = int((2.0 - result.origin_x) / result.resolution)
    row_from_bottom = int((2.0 - result.origin_y) / result.resolution)
    image_row = result.height - 1 - row_from_bottom
    assert result.grid[image_row, col] == UNKNOWN_VALUE


def test_height_filter_changes_occupied_cells():
    points = make_synthetic_cloud()
    # Height filter excludes the wall entirely -> no occupied cells at all.
    result = compute_occupancy_grid(points, min_height=5.0, max_height=6.0, resolution=0.1)
    assert not np.any(result.grid == OCCUPIED_VALUE)
    assert np.any(result.grid == FREE_VALUE)


def test_export_map_writes_valid_pgm_and_yaml(tmp_path):
    points = make_synthetic_cloud()
    result = compute_occupancy_grid(points, min_height=0.1, max_height=1.5, resolution=0.1)
    basepath = str(tmp_path / "test_map")
    pgm_path, yaml_path = export_map(basepath, result)

    assert os.path.exists(pgm_path)
    assert os.path.exists(yaml_path)

    with open(pgm_path, "rb") as f:
        header = f.readline().strip()
        dims = f.readline().strip().split()
        maxval = f.readline().strip()
        assert header == b"P5"
        w, h = int(dims[0]), int(dims[1])
        assert w == result.width
        assert h == result.height
        assert maxval == b"255"
        payload = f.read()
        assert len(payload) == w * h

    import yaml as yaml_mod

    with open(yaml_path) as f:
        meta = yaml_mod.safe_load(f)
    assert meta["resolution"] == result.resolution
    assert meta["origin"][0] == result.origin_x
    assert meta["origin"][1] == result.origin_y
    assert meta["image"] == os.path.basename(pgm_path)


def test_empty_point_cloud_raises():
    import pytest

    with pytest.raises(ValueError):
        compute_occupancy_grid(np.empty((0, 3)), 0.0, 1.0, 0.1)


def test_invalid_height_range_raises():
    import pytest

    points = make_synthetic_cloud()
    with pytest.raises(ValueError):
        compute_occupancy_grid(points, min_height=2.0, max_height=1.0, resolution=0.1)
