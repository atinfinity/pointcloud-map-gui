import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ground_removal import (  # noqa: E402
    DEFAULT_PARAMS,
    METHODS,
    _bilinear_sample,
    _build_min_z_dem,
    _fill_nan,
    _morphological_open,
    default_params,
    estimate_ground_mask,
)


def _method_ids():
    return list(METHODS.keys())


def _skip_without_csf():
    """Skip if the CSF extension is missing -- without importing it.

    pytest.importorskip would load it here, ahead of ground_removal._import_csf,
    and the single-thread pin that makes CSF reproducible only takes on the
    first load. Importing it from a test would quietly cost the whole session
    that guarantee.
    """
    if importlib.util.find_spec("CSF") is None:
        pytest.skip("cloth-simulation-filter is not installed")


def _run(method, points, **overrides):
    if method == "csf":
        _skip_without_csf()
    params = default_params(method)
    params.update(overrides)
    return estimate_ground_mask(points, method, **params)


def make_floor(slope_x=0.0, noise=0.01, size=6.0, step=0.05, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(0, size, step), np.arange(0, size, step))
    xs, ys = xs.ravel(), ys.ravel()
    zs = slope_x * xs + rng.normal(0.0, noise, size=xs.size)
    return np.stack([xs, ys, zs], axis=1)


def make_wall(x, slope_x=0.0, size=6.0, height=2.0, step=0.05):
    ys, hs = np.meshgrid(np.arange(0, size, step), np.arange(0.3, height, step))
    ys, hs = ys.ravel(), hs.ravel()
    return np.stack([np.full(ys.size, x), ys, slope_x * x + hs], axis=1)


def make_box_top(x0, y0, side, slope_x=0.0, height=0.3, step=0.02):
    xs, ys = np.meshgrid(np.arange(x0, x0 + side, step), np.arange(y0, y0 + side, step))
    xs, ys = xs.ravel(), ys.ravel()
    return np.stack([xs, ys, slope_x * xs + height], axis=1)


@pytest.mark.parametrize("method", _method_ids())
def test_noisy_flat_floor_is_ground_wall_is_not(method):
    floor = make_floor()
    wall = make_wall(3.0)
    mask = _run(method, np.vstack([floor, wall]))
    assert mask[: floor.shape[0]].mean() > 0.95
    assert mask[floor.shape[0] :].mean() < 0.05


@pytest.mark.parametrize("method", _method_ids())
def test_sloped_floor_removed_box_kept(method):
    slope = 0.15
    floor = make_floor(slope_x=slope)
    box = make_box_top(1.0, 4.0, 0.6, slope_x=slope, height=0.3)
    wall = make_wall(5.5, slope_x=slope)
    mask = _run(method, np.vstack([floor, box, wall]))
    n_f, n_b = floor.shape[0], box.shape[0]
    assert mask[:n_f].mean() > 0.9, "sloped floor should be ground"
    assert mask[n_f : n_f + n_b].mean() < 0.05, "box top should not be ground"
    assert mask[n_f + n_b :].mean() < 0.05, "wall should not be ground"


@pytest.mark.parametrize("method", _method_ids())
def test_cloud_with_empty_regions_does_not_raise(method):
    floor = make_floor()
    center = np.array([3.0, 3.0])
    ring = floor[np.linalg.norm(floor[:, :2] - center, axis=1) > 1.5]
    mask = _run(method, ring)
    assert mask.shape == (ring.shape[0],)
    assert mask.mean() > 0.9


@pytest.mark.parametrize("method", _method_ids())
def test_invalid_inputs_raise(method):
    if method == "csf":
        _skip_without_csf()
    with pytest.raises(ValueError):
        estimate_ground_mask(np.empty((0, 3)), method, **default_params(method))
    first_param = next(iter(DEFAULT_PARAMS[method]))
    with pytest.raises(ValueError):
        _run(method, make_floor(), **{first_param: -1.0})


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        estimate_ground_mask(make_floor(), "nope")


def test_default_params_match_metadata():
    for method, spec in DEFAULT_PARAMS.items():
        assert set(default_params(method)) == set(spec)


# --- DEM helper unit tests -------------------------------------------------
def test_min_z_dem_takes_cell_minimum_and_marks_empty_cells_nan():
    pts = np.array([[0.1, 0.1, 5.0], [0.2, 0.2, 1.0], [2.5, 0.1, 3.0]])
    dem, origin = _build_min_z_dem(pts, 1.0)
    assert dem.shape == (1, 3)
    assert dem[0, 0] == 1.0
    assert np.isnan(dem[0, 1])
    assert dem[0, 2] == 3.0
    np.testing.assert_allclose(origin, [0.1, 0.1])


def test_fill_nan_propagates_neighbour_minimum():
    dem = np.array([[1.0, np.nan, np.nan], [np.nan, np.nan, 4.0]])
    filled = _fill_nan(dem)
    assert not np.isnan(filled).any()
    assert filled[0, 1] == 1.0
    assert filled[1, 2] == 4.0


def test_opening_removes_small_bump_keeps_plateau():
    dem = np.zeros((9, 9))
    dem[4, 4] = 1.0  # one-cell island
    dem[0:4, 0:4] = 2.0  # 4x4 plateau
    opened = _morphological_open(dem, 3)
    assert opened[4, 4] == 0.0
    assert opened[1, 1] == 2.0


def test_bilinear_sample_interpolates_between_cell_centres():
    dem = np.array([[0.0, 1.0]])
    origin = np.array([0.0, 0.0])
    # Cell centres at x=0.5 and x=1.5 -> midpoint x=1.0 gives 0.5.
    z = _bilinear_sample(dem, np.array([[0.5, 0.5], [1.0, 0.5], [1.5, 0.5], [9.0, 0.5]]), origin, 1.0)
    np.testing.assert_allclose(z, [0.0, 0.5, 1.0, 1.0])


def test_csf_import_leaves_the_environment_alone():
    """CSF is pinned to one thread only for its own load. Leaving the variable
    set would follow every child process and cost the noise methods, which do
    scale with threads, up to 2.7x."""
    import ground_removal as gr

    before = os.environ.get("OMP_NUM_THREADS")
    gr._import_csf()
    assert os.environ.get("OMP_NUM_THREADS") == before


def test_csf_returns_the_same_answer_every_time():
    """Without the pin the ground count wandered over a 90-point range on this
    cloud, all of it points sitting within class_threshold of the cloth."""
    _skip_without_csf()
    path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_ramp.ply")
    if not os.path.exists(path):
        pytest.skip("sample_ramp.ply has not been generated")
    import open3d as o3d

    import ground_removal as gr

    points = np.asarray(o3d.io.read_point_cloud(path).points)
    params = gr.default_params("csf")
    first = gr.estimate_ground_mask(points, "csf", **params)
    for _ in range(5):
        assert np.array_equal(gr.estimate_ground_mask(points, "csf", **params), first)
