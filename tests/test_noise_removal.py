
import numpy as np
import pytest


from pointcloud_map_gui.noise_removal import (  # noqa: E402
    DEFAULT_PARAMS,
    METHODS,
    default_params,
    estimate_noise_mask,
)


def _run(method, points, **overrides):
    params = default_params(method)
    params.update(overrides)
    return estimate_noise_mask(points, method, **params)


def make_structure(step=0.03):
    """Dense floor (4x4m) plus a wall, both sampled finely enough that every
    point has many neighbours within 0.1m."""
    xs, ys = np.meshgrid(np.arange(0, 4, step), np.arange(0, 4, step))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    ys, zs = np.meshgrid(np.arange(0, 4, step), np.arange(0, 2, step))
    wall = np.stack([np.full(ys.size, 4.0), ys.ravel(), zs.ravel()], axis=1)
    return np.vstack([floor, wall])


def make_isolated(n=200, seed=0):
    """Random points at least 0.5m above the floor and away from the wall."""
    rng = np.random.default_rng(seed)
    return rng.uniform([0.2, 0.2, 0.5], [3.5, 3.8, 3.0], size=(n, 3))


# scatter is excluded on purpose. It asks whether a neighbourhood fills a
# volume, and answers by agreement among neighbours -- which a handful of
# points scattered through a room cannot reach, because their own neighbours
# are too few and too mixed. That strictness is what keeps corners and edges,
# and it is deliberate: sparse scatter is what radius and cluster are for. See
# test_scatter.py, which pins the boundary from the other side.
DENSITY_METHODS = [m for m in METHODS if m != "scatter"]


@pytest.mark.parametrize("method", DENSITY_METHODS)
def test_isolated_points_removed_structure_kept(method):
    structure = make_structure()
    isolated = make_isolated()
    mask = _run(method, np.vstack([structure, isolated]))
    n_s = structure.shape[0]
    assert mask[:n_s].mean() < 0.01, "structure should be kept"
    assert mask[n_s:].mean() > 0.95, "isolated points should be removed"


@pytest.mark.parametrize("method", ["radius", "cluster"])
def test_tiny_cluster_removed(method):
    structure = make_structure()
    tiny = np.array([[2.0, 2.0, 1.5], [2.02, 2.0, 1.5], [2.0, 2.02, 1.5]])
    mask = _run(method, np.vstack([structure, tiny]))
    assert mask[-3:].all()


def make_specks(n_clumps=20, seed=3):
    """Dense clumps of 10-40 points floating in the room -- the 'specks' that
    per-point outlier tests let through."""
    rng = np.random.default_rng(seed)
    clumps = []
    for _ in range(n_clumps):
        center = rng.uniform([0.5, 0.5, 0.5], [3.5, 3.5, 2.5])
        clumps.append(center + rng.normal(0.0, 0.02, size=(int(rng.integers(10, 41)), 3)))
    return np.vstack(clumps)


def test_cluster_removes_speck_clumps_that_radius_keeps():
    structure = make_structure()
    specks = make_specks()
    pts = np.vstack([structure, specks])
    n_s = structure.shape[0]
    cluster_mask = _run("cluster", pts)
    assert cluster_mask[n_s:].mean() > 0.9
    assert cluster_mask[:n_s].mean() < 0.01
    radius_mask = _run("radius", pts)
    assert radius_mask[n_s:].mean() < 0.1  # documents why cluster is the default


def test_cluster_is_default_method():
    assert next(iter(METHODS)) == "cluster"
    assert next(iter(DEFAULT_PARAMS)) == "cluster"


@pytest.mark.parametrize("method", list(METHODS))
def test_invalid_inputs_raise(method):
    with pytest.raises(ValueError):
        estimate_noise_mask(np.empty((0, 3)), method, **default_params(method))
    first_param = next(iter(DEFAULT_PARAMS[method]))
    with pytest.raises(ValueError):
        _run(method, make_structure(), **{first_param: -1.0})


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        estimate_noise_mask(make_structure(), "nope")


def test_default_params_match_metadata():
    for method, spec in DEFAULT_PARAMS.items():
        assert set(default_params(method)) == set(spec)


def test_voxel_count_exact_threshold():
    # Three points share one voxel, one point sits alone.
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.03, 0.03, 0.03], [5.0, 5.0, 5.0]])
    mask = _run("voxel_count", pts, voxel_size=0.1, min_points=3)
    assert mask.tolist() == [False, False, False, True]
