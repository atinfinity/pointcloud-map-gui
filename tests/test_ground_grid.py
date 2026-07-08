import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ground_grid import MARGIN_CELLS, build_ground_grid_lines, nice_step


def test_nice_step_rounds_to_1_2_5_sequence():
    assert nice_step(0.08) == 0.1
    assert nice_step(0.25) == 0.2
    assert nice_step(0.4) == 0.5
    assert nice_step(0.8) == 1.0
    assert nice_step(3.0) == 2.0
    assert nice_step(4.0) == 5.0
    assert nice_step(9.0) == 10.0


def test_extended_bounds_are_whole_multiples_of_step():
    # Deliberately misaligned bounds (not multiples of any nice step).
    _, _, (ext_min_x, ext_min_y, ext_max_x, ext_max_y, step) = build_ground_grid_lines(
        0.3, -0.2, 5.7, 4.9
    )
    assert np.isclose(ext_min_x % step, 0, atol=1e-9) or np.isclose(ext_min_x % step, step, atol=1e-9)
    assert np.isclose(ext_max_x % step, 0, atol=1e-9) or np.isclose(ext_max_x % step, step, atol=1e-9)
    assert np.isclose(ext_min_y % step, 0, atol=1e-9) or np.isclose(ext_min_y % step, step, atol=1e-9)
    assert np.isclose(ext_max_y % step, 0, atol=1e-9) or np.isclose(ext_max_y % step, step, atol=1e-9)


def test_extended_bounds_fully_contain_the_input_bounds_plus_margin():
    min_x, min_y, max_x, max_y = 0.3, -0.2, 5.7, 4.9
    _, _, (ext_min_x, ext_min_y, ext_max_x, ext_max_y, step) = build_ground_grid_lines(
        min_x, min_y, max_x, max_y
    )
    assert ext_min_x <= min_x
    assert ext_min_y <= min_y
    assert ext_max_x >= max_x
    assert ext_max_y >= max_y
    # Extension is a MARGIN_CELLS-wide border beyond the tightly-rounded
    # bounds, so it overshoots by at least that many whole cells but never
    # more than one extra cell (from the floor/ceil rounding step).
    margin = MARGIN_CELLS * step
    assert margin <= min_x - ext_min_x < margin + step
    assert margin <= min_y - ext_min_y < margin + step
    assert margin <= ext_max_x - max_x < margin + step
    assert margin <= ext_max_y - max_y < margin + step


def test_aligned_bounds_get_exactly_the_margin_added():
    # 0..6 is exactly 12 * 0.5, and 0.5 is what nice_step picks for this size,
    # so with no rounding needed the extension is exactly the margin.
    _, _, (ext_min_x, ext_min_y, ext_max_x, ext_max_y, step) = build_ground_grid_lines(
        0.0, 0.0, 6.0, 6.0
    )
    assert step == 0.5
    margin = MARGIN_CELLS * step
    assert ext_min_x == -margin
    assert ext_min_y == -margin
    assert ext_max_x == 6.0 + margin
    assert ext_max_y == 6.0 + margin


def test_every_line_spans_the_full_extended_rectangle():
    points, lines, (ext_min_x, ext_min_y, ext_max_x, ext_max_y, step) = build_ground_grid_lines(
        0.3, -0.2, 5.7, 4.9
    )
    points = np.array(points)
    for i0, i1 in lines:
        p0, p1 = points[i0], points[i1]
        if p0[0] == p1[0]:  # vertical line: must span full Y range
            assert {p0[1], p1[1]} == {ext_min_y, ext_max_y}
        else:  # horizontal line: must span full X range
            assert {p0[0], p1[0]} == {ext_min_x, ext_max_x}
