import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from colorize import GRAY_OUT_COLOR, fade_to_background, height_colormap_colors


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


def test_fade_to_background_moves_color_toward_background():
    background = np.array([0.35, 0.35, 0.35])
    faded = fade_to_background(GRAY_OUT_COLOR, 0.15, background)
    # The point stays lighter than the background it sits on, but far closer
    # to it than the unfaded color is.
    assert np.all(faded > background * 0.4)
    assert np.all(faded < GRAY_OUT_COLOR)
    assert np.all(np.abs(faded - background) < np.abs(GRAY_OUT_COLOR - background))


def test_fade_to_background_alpha_endpoints():
    background = np.array([0.35, 0.35, 0.35])
    color = np.array([1.0, 0.3, 0.3])
    assert np.allclose(fade_to_background(color, 1.0, background), color, atol=1e-6)
    # At alpha 0 the point is exactly the background, whatever its own color.
    fully_faded = fade_to_background(color, 0.0, background)
    assert np.allclose(fully_faded, fade_to_background(GRAY_OUT_COLOR, 0.0, background))


def test_fade_to_background_keeps_hue_order():
    """A red marker must still read as red, not as neutral gray."""
    faded = fade_to_background(np.array([1.0, 0.3, 0.3]), 0.15, np.array([0.35, 0.35, 0.35]))
    assert faded[0] > faded[1]
    assert np.isclose(faded[1], faded[2])
