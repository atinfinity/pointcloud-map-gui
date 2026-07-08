import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from colorize import height_colormap_colors


def test_colors_vary_with_height():
    points = np.array([[0, 0, 0.0], [0, 0, 5.0], [0, 0, 10.0]])
    colors = height_colormap_colors(points)
    assert not np.allclose(colors[0], colors[1])
    assert not np.allclose(colors[1], colors[2])
    assert not np.allclose(colors[0], colors[2])


def test_same_height_gives_same_color():
    points = np.array([[0, 0, 3.0], [5, -2, 3.0]])
    colors = height_colormap_colors(points)
    assert np.allclose(colors[0], colors[1])


def test_colors_stay_within_unit_range():
    z = np.linspace(0, 10, 50)
    points = np.stack([np.zeros(50), np.zeros(50), z], axis=1)
    colors = height_colormap_colors(points)
    assert colors.min() >= 0.0
    assert colors.max() <= 1.0


def test_single_height_does_not_crash():
    points = np.array([[0, 0, 2.0], [1, 1, 2.0]])
    colors = height_colormap_colors(points)
    assert colors.shape == (2, 3)
    assert np.allclose(colors[0], colors[1])
