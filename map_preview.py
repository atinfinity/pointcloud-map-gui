"""Downsample an occupancy grid image into a GUI thumbnail preview."""
import numpy as np


def downsample_to_thumbnail(grid, max_dim=220, alpha=None):
    """Return a uint8 thumbnail of a single-channel occupancy grid image,
    nearest-neighbor subsampled so neither dimension exceeds max_dim.

    With alpha=None the result is (H, W, 3) RGB. With an integer alpha
    (0-255) the result is (H, W, 4) RGBA with a constant alpha channel, so
    the preview can be overlaid translucently on the 3D view.
    """
    height, width = grid.shape
    stride = max(1, int(np.ceil(max(height, width) / max_dim)))
    small = grid[::stride, ::stride].astype(np.uint8)
    if alpha is None:
        return np.stack([small, small, small], axis=-1)
    alpha_plane = np.full_like(small, int(alpha))
    return np.stack([small, small, small, alpha_plane], axis=-1)
