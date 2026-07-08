"""Point cloud -> ROS2-style occupancy grid conversion.

Pure numpy logic, no Open3D dependency, so it can be unit tested headless.
"""
from dataclasses import dataclass

import numpy as np

OCCUPIED_VALUE = 0
FREE_VALUE = 254
UNKNOWN_VALUE = 205


@dataclass
class OccupancyGridResult:
    grid: np.ndarray  # uint8, shape (height, width), image-row-order (row 0 = max Y)
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int


def compute_occupancy_grid(points, min_height, max_height, resolution):
    """Convert an (N, 3) point array into a ROS2 map_server-style occupancy grid.

    Cells touched by any point (regardless of height) are considered "known"
    (scanned) territory. Among known cells, those touched by a point whose Z
    falls within [min_height, max_height] are "occupied"; the rest are "free".
    Cells never touched by any point are "unknown".
    """
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty")
    if min_height > max_height:
        raise ValueError("min_height must be <= max_height")
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    origin_x = float(xs.min())
    origin_y = float(ys.min())
    max_x = float(xs.max())
    max_y = float(ys.max())

    width = max(1, int(np.ceil((max_x - origin_x) / resolution)) + 1)
    height = max(1, int(np.ceil((max_y - origin_y) / resolution)) + 1)

    cols = np.floor((xs - origin_x) / resolution).astype(np.int64)
    rows = np.floor((ys - origin_y) / resolution).astype(np.int64)
    cols = np.clip(cols, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)

    known_grid = np.zeros((height, width), dtype=bool)
    known_grid[rows, cols] = True

    height_mask = (zs >= min_height) & (zs <= max_height)
    occupied_grid = np.zeros((height, width), dtype=bool)
    if np.any(height_mask):
        occupied_grid[rows[height_mask], cols[height_mask]] = True

    grid = np.full((height, width), UNKNOWN_VALUE, dtype=np.uint8)
    grid[known_grid] = FREE_VALUE
    grid[occupied_grid] = OCCUPIED_VALUE

    # Flip vertically: row index 0 in our array corresponds to origin_y (min Y),
    # but map_server/image convention expects the top image row to be max Y.
    grid = np.flipud(grid)

    return OccupancyGridResult(
        grid=grid,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
    )
