
import numpy as np


from pointcloud_map_gui.map_preview import downsample_to_thumbnail
from pointcloud_map_gui.occupancy_grid import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


def test_small_grid_is_unchanged():
    grid = (np.arange(20 * 30) % 256).astype(np.uint8).reshape(20, 30)
    thumb = downsample_to_thumbnail(grid, max_dim=220)
    assert thumb.shape == (20, 30, 3)
    assert np.array_equal(thumb[:, :, 0], grid)
    assert np.array_equal(thumb[:, :, 0], thumb[:, :, 1])
    assert np.array_equal(thumb[:, :, 0], thumb[:, :, 2])


def test_large_grid_is_downsampled_within_max_dim():
    grid = np.zeros((1000, 400), dtype=np.uint8)
    thumb = downsample_to_thumbnail(grid, max_dim=220)
    assert thumb.shape[0] <= 220
    assert thumb.shape[1] <= 220
    assert thumb.shape[2] == 3


def test_downsampled_values_are_a_subset_of_the_original():
    grid = np.random.randint(0, 256, size=(500, 500), dtype=np.uint8)
    thumb = downsample_to_thumbnail(grid, max_dim=100)
    unique_thumb = set(np.unique(thumb[:, :, 0]).tolist())
    unique_grid = set(np.unique(grid).tolist())
    assert unique_thumb.issubset(unique_grid)


def test_dtype_is_uint8():
    grid = np.full((50, 50), 205, dtype=np.uint8)
    thumb = downsample_to_thumbnail(grid)
    assert thumb.dtype == np.uint8


def test_alpha_adds_constant_alpha_channel():
    grid = np.random.randint(0, 256, size=(300, 200), dtype=np.uint8)
    thumb = downsample_to_thumbnail(grid, max_dim=100, alpha=128)
    assert thumb.shape[2] == 4
    assert thumb.dtype == np.uint8
    assert np.all(thumb[:, :, 3] == 128)
    assert np.array_equal(thumb[:, :, 0], thumb[:, :, 2])


def test_thin_obstacle_survives_downsampling_at_any_offset():
    """A one-cell wall must reach the preview whatever column it lands on.

    Sampling one cell per block used to drop it outright on odd columns, so
    walls went missing from the preview of any map larger than the thumbnail
    while the exported map still had them.
    """
    for column in range(6):
        grid = np.full((60, 60), FREE_VALUE, dtype=np.uint8)
        grid[:, column] = OCCUPIED_VALUE
        thumb = downsample_to_thumbnail(grid, max_dim=20)  # stride 3
        assert (thumb[:, :, 0] == OCCUPIED_VALUE).any(), f"wall at column {column} vanished"


def test_obstacles_win_over_free_and_unknown_in_a_block():
    grid = np.array(
        [
            [FREE_VALUE, UNKNOWN_VALUE],
            [UNKNOWN_VALUE, OCCUPIED_VALUE],
        ],
        dtype=np.uint8,
    )
    thumb = downsample_to_thumbnail(grid, max_dim=1)
    assert thumb[0, 0, 0] == OCCUPIED_VALUE


def test_unknown_wins_over_free_in_a_block():
    grid = np.array(
        [[FREE_VALUE, FREE_VALUE], [FREE_VALUE, UNKNOWN_VALUE]], dtype=np.uint8
    )
    thumb = downsample_to_thumbnail(grid, max_dim=1)
    assert thumb[0, 0, 0] == UNKNOWN_VALUE


def test_ragged_grid_is_padded_without_inventing_obstacles():
    """A grid that is not a whole number of blocks across still reduces, and
    the padding must not read as free space over a real obstacle."""
    grid = np.full((7, 7), FREE_VALUE, dtype=np.uint8)
    grid[6, 6] = OCCUPIED_VALUE
    thumb = downsample_to_thumbnail(grid, max_dim=3)  # stride 3, 7 -> 3 blocks
    assert thumb.shape[:2] == (3, 3)
    assert thumb[2, 2, 0] == OCCUPIED_VALUE
    assert set(np.unique(thumb[:, :, 0]).tolist()) <= {OCCUPIED_VALUE, FREE_VALUE}
