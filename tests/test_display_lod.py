import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from display_lod import select_display_indices


def _grid_cloud(n_side=60, extent=6.0):
    """A hollow box surface, i.e. the shape real scans have: points on
    surfaces, empty in between."""
    g = np.linspace(0.0, extent, n_side)
    xx, yy = np.meshgrid(g, g)
    floor = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    wall = np.column_stack([xx.ravel(), np.zeros(xx.size), yy.ravel()])
    return np.vstack([floor, wall])


def test_returns_none_when_cloud_already_fits():
    points = _grid_cloud(10)
    assert select_display_indices(points, points.shape[0]) is None
    assert select_display_indices(points, points.shape[0] + 1) is None


def test_zero_budget_disables_thinning():
    points = _grid_cloud(10)
    assert select_display_indices(points, 0) is None


def test_respects_budget_and_returns_valid_indices():
    points = _grid_cloud(80)
    budget = 1500
    index = select_display_indices(points, budget)
    assert index is not None
    assert index.shape[0] <= budget
    assert index.shape[0] > 0
    assert index.min() >= 0
    assert index.max() < points.shape[0]
    assert np.unique(index).shape[0] == index.shape[0]


def test_uses_a_useful_fraction_of_the_budget():
    """A thinning that returns 20 points when 1500 were allowed is technically
    within budget and useless to look at."""
    points = _grid_cloud(80)
    budget = 1500
    index = select_display_indices(points, budget)
    assert index.shape[0] >= budget // 4


def test_thinning_stays_spatially_uniform():
    """Voxel selection should keep the shape's extent, unlike a head-of-array
    truncation which would drop whole regions."""
    points = _grid_cloud(80)
    index = select_display_indices(points, 1000)
    kept = points[index]
    assert np.allclose(kept.min(axis=0), points.min(axis=0), atol=0.3)
    assert np.allclose(kept.max(axis=0), points.max(axis=0), atol=0.3)


def test_degenerate_cloud_of_identical_points():
    points = np.zeros((500, 3))
    index = select_display_indices(points, 10)
    assert index is not None
    assert index.shape[0] <= 10


def test_single_axis_cloud():
    points = np.column_stack([np.linspace(0, 10, 5000), np.zeros(5000), np.zeros(5000)])
    index = select_display_indices(points, 100)
    assert index is not None
    assert index.shape[0] <= 100
    assert index.max() < points.shape[0]
