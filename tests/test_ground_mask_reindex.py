import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import reindex_ground_mask

TOTAL = 8


def _mask(*indices):
    m = np.zeros(TOTAL, dtype=bool)
    m[list(indices)] = True
    return m


def test_nothing_to_carry():
    assert reindex_ground_mask(None, None, _mask(1), TOTAL) is None


def test_noise_switched_on_drops_the_removed_points():
    """Ground was classified over every point; now two are being removed."""
    ground = _mask(0, 1, 2, 3)  # over all 8 points
    carried = reindex_ground_mask(ground, None, _mask(1, 5), TOTAL)
    # Surviving points are 0,2,3,4,6,7 -- of those, 0, 2 and 3 were ground.
    assert carried.tolist() == [True, True, True, False, False, False]


def test_noise_switched_off_puts_the_points_back_unclassified():
    previous = _mask(1, 5)
    ground = np.array([True, True, False, False, False, False])  # over 0,2,3,4,6,7
    carried = reindex_ground_mask(ground, previous, None, TOTAL)
    # 0 and 2 stay ground; 1 and 5 come back, not classified.
    assert carried.tolist() == [True, False, True, False, False, False, False, False]


def test_one_noise_mask_to_another():
    previous = _mask(7)
    ground = np.array([True, False, True, False, False, False, True])  # over 0..6
    carried = reindex_ground_mask(ground, previous, _mask(0), TOTAL)
    # Surviving points are 1,2,3,4,5,6,7 -- of those 2 and 6 were ground, so
    # positions 1 and 5 of the result; 7 is newly back and unclassified.
    assert carried.tolist() == [False, True, False, False, False, True, False]


def test_classification_survives_a_round_trip():
    ground = _mask(0, 3, 6)
    noise = _mask(2, 4)
    removed = reindex_ground_mask(ground, None, noise, TOTAL)
    restored = reindex_ground_mask(removed, noise, None, TOTAL)
    # Every point that was never removed keeps its answer.
    kept = ~noise
    assert np.array_equal(restored[kept], ground[kept])


def test_a_mask_of_the_wrong_length_is_refused():
    """Better to re-estimate than to silently shift every point's answer."""
    assert reindex_ground_mask(np.zeros(3, dtype=bool), None, None, TOTAL) is None
    assert reindex_ground_mask(np.zeros(TOTAL, dtype=bool), _mask(1), None, TOTAL) is None
