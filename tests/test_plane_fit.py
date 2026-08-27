import numpy as np
import pytest

from pointcloud_map_gui.noise_removal import estimate_noise_mask, noise_mask_plane_fit


def _plane(n=40, spacing=0.05, jitter=0.0, seed=0):
    """A flat grid in z=0, optionally roughened along the normal."""
    rng = np.random.default_rng(seed)
    g = np.arange(n) * spacing
    x, y = np.meshgrid(g, g)
    z = rng.normal(0.0, jitter, x.size) if jitter else np.zeros(x.size)
    return np.stack([x.ravel(), y.ravel(), z], axis=1)


def test_a_point_lifted_off_the_surface_is_found():
    points = _plane(jitter=0.002, seed=1)
    points[500, 2] += 0.05
    mask = noise_mask_plane_fit(points, knn=20, nsigma=3.0)
    assert mask[500]
    assert mask.sum() < points.shape[0] * 0.05  # and it does not take the surface with it


def test_a_point_in_the_surface_is_kept():
    points = _plane(jitter=0.002, seed=2)
    mask = noise_mask_plane_fit(points, knn=20, nsigma=6.0)
    assert not mask.any()


def test_a_perfectly_flat_cloud_loses_nothing():
    """No spread means no evidence about what "too far" is. Judging against
    zero would condemn everything not exactly coplanar."""
    mask = noise_mask_plane_fit(_plane(), knn=20, nsigma=3.0)
    assert not mask.any()


def test_absolute_error_cuts_at_the_distance_given():
    points = _plane(jitter=0.001, seed=3)
    points[100, 2] += 0.04
    points[200, 2] += 0.20
    assert not noise_mask_plane_fit(points, max_error=0.10)[100]
    assert noise_mask_plane_fit(points, max_error=0.10)[200]
    assert noise_mask_plane_fit(points, max_error=0.02)[100]


def test_absolute_error_takes_over_from_nsigma():
    points = _plane(jitter=0.002, seed=4)
    points[300, 2] += 0.05
    # An nsigma that would catch it is ignored once max_error is set wide.
    assert noise_mask_plane_fit(points, nsigma=3.0)[300]
    assert not noise_mask_plane_fit(points, nsigma=3.0, max_error=1.0)[300]


def test_the_point_itself_is_excluded_from_its_own_fit():
    """Including it caps how far it can look from a plane it helped define --
    with knn+1 samples nothing can exceed (k)/sqrt(k+1) sigma, so nsigma=3
    would find nothing however bad the point is."""
    points = _plane(jitter=0.002, seed=5)
    points[400, 2] += 1.0  # wildly off; only findable if it is left out
    assert noise_mask_plane_fit(points, knn=8, nsigma=3.0)[400]


def test_a_cloud_smaller_than_the_neighbourhood_is_left_alone():
    assert not noise_mask_plane_fit(_plane(n=3), knn=20).any()


@pytest.mark.parametrize(
    "params", [{"knn": 2}, {"nsigma": 0.0}, {"nsigma": -1.0}, {"max_error": -0.1}]
)
def test_invalid_parameters_raise(params):
    with pytest.raises(ValueError):
        noise_mask_plane_fit(_plane(n=10), **params)


def test_reachable_through_the_registry():
    points = _plane(jitter=0.002, seed=6)
    points[50, 2] += 0.05
    assert estimate_noise_mask(points, "plane_fit", knn=20, nsigma=3.0, max_error=0.0)[50]


def test_scattered_points_are_not_what_this_method_finds():
    """The boundary of the method, asserted so it stays deliberate.

    A point alone in space has no surface to be off: its neighbours are other
    scattered points, a plane through them fits nothing, and the spread it is
    judged against is as large as the distance. Density is radius's and
    cluster's job -- pair them with this one rather than expecting it to
    cover both.
    """
    rng = np.random.default_rng(0)
    surface = _plane(jitter=0.002, seed=7)
    scatter = rng.uniform([0.2, 0.2, 0.5], [1.8, 1.8, 3.0], size=(200, 3))
    mask = noise_mask_plane_fit(np.vstack([surface, scatter]), knn=20, nsigma=3.0)
    # A few percent of a noisy surface always goes: nsigma is not a plain
    # z-score, see test_nsigma_is_stricter_than_the_normal_distribution_suggests.
    assert mask[: surface.shape[0]].mean() < 0.05, "the surface is mostly left alone"
    assert mask[surface.shape[0] :].mean() < 0.5, "scatter is not its business"


def test_nsigma_is_stricter_than_the_normal_distribution_suggests():
    """nsigma=3 does not mean "the 0.27% beyond three sigma".

    The neighbours' spread is measured against a plane fitted to those same
    neighbours, so their residuals are shrunk by the fitting. The point's own
    distance is measured against a plane it had no part in, so it carries the
    plane's estimation error as well. The bar ends up lower than it reads, and
    a few percent of a cleanly noisy surface crosses it. Worth knowing before
    reaching for nsigma to trade recall against precision.
    """
    surface = _plane(n=80, jitter=0.002, seed=12)
    removed = noise_mask_plane_fit(surface, knn=20, nsigma=3.0).mean()
    assert 0.005 < removed < 0.05
