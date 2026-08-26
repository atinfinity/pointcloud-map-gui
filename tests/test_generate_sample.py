import numpy as np
import pytest

from pointcloud_map_gui.tools import generate_sample

SCENE_BUILDERS = [
    generate_sample.build_room_point_cloud,
    generate_sample.build_slope_point_cloud,
    generate_sample.build_noisy_slope_point_cloud,
    generate_sample.build_ramp_point_cloud,
]


def _is_float32_exact(points):
    return np.array_equal(points.astype(np.float32).astype(np.float64), points)


def test_point_cloud_helper_rounds_to_float32():
    """A coordinate float32 cannot hold must come back rounded, not carried."""
    awkward = np.array([[1 / 3, np.pi, 1e-9]])
    assert not _is_float32_exact(awkward)
    points = np.asarray(generate_sample._point_cloud(awkward).points)
    assert _is_float32_exact(points)


def test_helper_keeps_colours_untouched():
    colors = np.array([[0.1, 0.2, 0.3]])
    pcd = generate_sample._point_cloud(np.zeros((1, 3)), colors)
    assert np.array_equal(np.asarray(pcd.colors), colors)


@pytest.mark.parametrize("builder", SCENE_BUILDERS, ids=lambda b: b.__name__)
def test_scene_clouds_are_float32_exact(builder):
    """Every builder has to go through _point_cloud.

    PLY stores coordinates as double and PCD as float32, so a builder that
    keeps full float64 precision writes a .ply that only reproduces on the
    machine that made it: rng.normal reaches the platform's libm through the
    ziggurat's exp/log, and macOS arm64 and Linux x86_64 disagree by a few ULP.
    Rounding to float32 is what makes the two formats agree and the files
    regenerate to the same bytes anywhere.
    """
    points = np.asarray(builder().points)
    assert points.shape[0] > 0
    assert _is_float32_exact(points)
