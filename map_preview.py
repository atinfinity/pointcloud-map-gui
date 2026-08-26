"""Downsample an occupancy grid image into a GUI thumbnail preview."""
import numpy as np

# Padding added to square off the last block. Higher than every real grid value,
# so it can never win the reduction below.
_PAD_VALUE = 255


def downsample_to_thumbnail(grid, max_dim=220, alpha=None):
    """Return a uint8 thumbnail of a single-channel occupancy grid image,
    reduced so neither dimension exceeds max_dim.

    Each output pixel is the *minimum* over the block it covers, not the
    top-left sample of it. The grid's values run occupied (0) < unknown (205)
    < free (254), so the minimum keeps whatever matters most in that block:
    an obstacle anywhere in it shows as an obstacle, and unscanned ground
    beats scanned. Picking one sample per block instead loses a wall one cell
    wide whenever its index falls between samples -- on a 50 m site gridded at
    5 cm, that dropped about a third of the walls from the preview while the
    exported map had them all.

    With alpha=None the result is (H, W, 3) RGB. With an integer alpha
    (0-255) the result is (H, W, 4) RGBA with a constant alpha channel, so
    the preview can be overlaid translucently on the 3D view.
    """
    height, width = grid.shape
    stride = max(1, int(np.ceil(max(height, width) / max_dim)))
    if stride == 1:
        small = grid.astype(np.uint8)
    else:
        padded = np.pad(
            grid,
            ((0, -height % stride), (0, -width % stride)),
            constant_values=_PAD_VALUE,
        )
        blocks = padded.reshape(
            padded.shape[0] // stride, stride, padded.shape[1] // stride, stride
        )
        small = blocks.min(axis=(1, 3)).astype(np.uint8)
    if alpha is None:
        return np.stack([small, small, small], axis=-1)
    alpha_plane = np.full_like(small, int(alpha))
    return np.stack([small, small, small, alpha_plane], axis=-1)
