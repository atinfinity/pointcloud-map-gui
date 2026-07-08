import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from map_preview import downsample_to_thumbnail


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
