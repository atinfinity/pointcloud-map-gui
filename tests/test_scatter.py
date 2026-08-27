import numpy as np
import pytest

from pointcloud_map_gui.noise_removal import estimate_noise_mask, noise_mask_scatter


def _surface(n=60, spacing=0.03, jitter=0.004, seed=0):
    """A flat sheet in z=0, roughened the way a scan is."""
    rng = np.random.default_rng(seed)
    g = np.arange(n) * spacing
    x, y = np.meshgrid(g, g)
    return np.stack([x.ravel(), y.ravel(), rng.normal(0.0, jitter, x.size)], axis=1)


def _blob(centre, n=1500, spread=0.2, seed=1):
    """Points filling a volume, as accumulated scans of a moving object do."""
    return np.asarray(centre) + np.random.default_rng(seed).normal(0.0, spread, (n, 3))


def _corner(n=60, spacing=0.03, jitter=0.004, seed=2):
    """Two sheets meeting at a right angle -- a wall against a floor."""
    rng = np.random.default_rng(seed)
    g = np.arange(n) * spacing
    a, b = np.meshgrid(g, g)
    floor = np.stack([a.ravel(), b.ravel(), rng.normal(0, jitter, a.size)], axis=1)
    wall = np.stack([rng.normal(0, jitter, a.size), b.ravel(), a.ravel()], axis=1)
    return np.vstack([floor, wall])


def test_a_surface_is_left_alone():
    assert not noise_mask_scatter(_surface()).any()


def test_a_volume_is_removed():
    surface = _surface()
    blob = _blob([0.9, 0.9, 0.5])
    mask = noise_mask_scatter(np.vstack([surface, blob]))
    assert mask[surface.shape[0] :].mean() > 0.9
    assert mask[: surface.shape[0]].mean() < 0.01


def test_a_corner_is_not_mistaken_for_haze():
    """Where two surfaces meet the neighbourhood spans both and scatters like a
    volume. It is a thin band, though: the neighbours of a corner point are
    mostly still flat, which is what `agreement` tests for."""
    points = _corner()
    assert noise_mask_scatter(points, agreement=0.9).mean() < 0.01


def test_agreement_is_what_saves_the_corner():
    """Pinned so the parameter cannot be quietly dropped: without it the corner
    goes."""
    points = _corner()
    without = noise_mask_scatter(points, agreement=0.0)
    with_it = noise_mask_scatter(points, agreement=0.9)
    # The band is thin -- 52 of 7,200 points on two sheets meeting at a right
    # angle -- but it is the join, and on a real cloud it is every join.
    assert without.sum() > 20
    assert not with_it.any()


def test_it_works_where_counting_neighbours_cannot():
    """The haze that survives to be a problem is as dense as the walls. This is
    the case the density methods cannot see at all."""
    from scipy.spatial import cKDTree

    surface = _surface()
    blob = _blob([0.9, 0.9, 0.5], n=6000, spread=0.18, seed=3)
    points = np.vstack([surface, blob])
    is_blob = np.zeros(points.shape[0], bool)
    is_blob[surface.shape[0] :] = True

    density = cKDTree(points).query_ball_point(points, 0.1, return_length=True)
    assert np.median(density[is_blob]) > np.median(density[~is_blob]), (
        "the blob has to be at least as dense as the sheet for this to be the "
        "case worth testing"
    )
    mask = noise_mask_scatter(points)
    assert mask[is_blob].mean() > 0.9
    assert mask[~is_blob].mean() < 0.01


def test_a_cloud_smaller_than_the_neighbourhood_is_left_alone():
    assert not noise_mask_scatter(_surface(n=4), knn=30).any()


@pytest.mark.parametrize(
    "params",
    [{"knn": 5}, {"max_scatter": 0.0}, {"max_scatter": 1.5}, {"agreement": -0.1},
     {"agreement": 1.1}],
)
def test_invalid_parameters_raise(params):
    with pytest.raises(ValueError):
        noise_mask_scatter(_surface(n=20), **params)


def test_reachable_through_the_registry():
    surface = _surface()
    points = np.vstack([surface, _blob([0.9, 0.9, 0.5])])
    mask = estimate_noise_mask(points, "scatter", knn=30, max_scatter=0.20, agreement=0.9)
    assert mask[surface.shape[0] :].mean() > 0.9


def test_a_thin_scattering_is_not_what_this_method_finds():
    """The boundary of the method, asserted so it stays deliberate.

    A few points strewn through a room do fill a volume, but too thinly to
    agree with each other, and `agreement` is what keeps corners. Loosening it
    to catch them would give the corners back. Sparse scatter is radius's and
    cluster's job -- pair them with this one.
    """
    rng = np.random.default_rng(4)
    surface = _surface()
    sparse = rng.uniform([0.1, 0.1, 0.2], [1.6, 1.6, 1.2], size=(120, 3))
    mask = noise_mask_scatter(np.vstack([surface, sparse]))
    assert mask[: surface.shape[0]].mean() < 0.01, "the surface is left alone"
    assert mask[surface.shape[0] :].mean() < 0.9, "thin scatter is not its business"
