"""Downsample an occupancy grid image into an RGB GUI thumbnail preview."""
import numpy as np


def downsample_to_thumbnail(grid, max_dim=220):
    """Return an (H, W, 3) uint8 RGB thumbnail of a single-channel occupancy
    grid image, nearest-neighbor subsampled so neither dimension exceeds
    max_dim.
    """
    height, width = grid.shape
    stride = max(1, int(np.ceil(max(height, width) / max_dim)))
    small = grid[::stride, ::stride]
    return np.stack([small, small, small], axis=-1).astype(np.uint8)
