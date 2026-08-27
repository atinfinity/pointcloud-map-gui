"""Isolated-noise point detection.

Four interchangeable methods, all sharing the signature

    method(points: (N, 3) float array, **params) -> (N,) bool noise mask

where True marks a point to *remove*. Unlike ground removal, noise points are
treated as if they never existed: they are dropped before display, ground
estimation and occupancy grid generation (so they cannot enlarge the map).

- ``radius``: fewer than ``min_neighbors`` points within ``radius`` -> noise.
- ``statistical``: mean k-NN distance beyond mean + std_ratio * std -> noise.
- ``voxel_count``: voxel holding fewer than ``min_points`` -> noise (numpy only).
- ``cluster``: DBSCAN; unlabeled points and clusters smaller than
  ``min_cluster_size`` -> noise.
- ``plane_fit``: distance to a plane fitted through the neighbours, against
  either their own spread or a fixed distance -> noise. CloudCompare's
  "noise filter"; the only one here that looks at *shape* rather than
  density, so it is the only one that can see a point sitting off a surface
  it still has plenty of neighbours on.
"""
import numpy as np

# Neighbour coordinates are materialised a slice at a time; see plane_fit.
_PLANE_FIT_CHUNK = 200_000
# Smallest spread plane_fit will judge against, as a fraction of the
# neighbourhood's radius. See where it is used.
_PLANE_FIT_FLAT_TOL = 1e-6

# name -> (default, min, max); ints are shown as integer fields in the GUI.
# Dict order is the GUI/CLI order; the first entry is the default method.
# ``cluster`` is first because per-point tests (radius/statistical/voxel)
# keep small dense clumps of 10-40 points, which then show up as specks on
# the map; only cluster size removes those.
DEFAULT_PARAMS = {
    "cluster": {
        "eps": (0.10, 0.01, 2.0),
        "min_points": (4, 1, 100),
        "min_cluster_size": (50, 1, 100000),
    },
    "radius": {
        "radius": (0.10, 0.01, 2.0),
        "min_neighbors": (8, 1, 100),
    },
    "statistical": {
        "nb_neighbors": (20, 2, 200),
        "std_ratio": (2.0, 0.1, 10.0),
    },
    "voxel_count": {
        "voxel_size": (0.10, 0.01, 2.0),
        "min_points": (4, 1, 100),
    },
    "plane_fit": {
        # 20 rather than a handful: on sample_range_noise, going from 8 to 20
        # neighbours lifts precision at nsigma=3 from 0.20 to 0.57 for the same
        # recall, because both the plane and the spread it is judged against
        # are estimated from more evidence.
        "knn": (20, 3, 200),
        "nsigma": (3.0, 0.1, 20.0),
        # 0 means "no absolute limit, use nsigma". Any positive value switches
        # to the fixed distance and nsigma stops mattering -- an absolute
        # threshold of zero would remove the whole cloud, so nothing is lost by
        # spending the value this way instead of on a separate switch.
        "max_error": (0.0, 0.0, 5.0),
    },
}

METHOD_LABELS = {
    "cluster": "DBSCAN cluster size",
    "radius": "Radius outlier",
    "statistical": "Statistical outlier",
    "voxel_count": "Voxel point count",
    "plane_fit": "Plane fit (CloudCompare noise filter)",
}


def default_params(method):
    return {name: spec[0] for name, spec in DEFAULT_PARAMS[method].items()}


def _validate_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty")
    return points


def _to_o3d(points):
    import open3d as o3d

    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def _mask_from_kept_indices(n, kept):
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(kept, dtype=np.int64)] = False
    return mask


def noise_mask_radius(points, radius=0.10, min_neighbors=8):
    points = _validate_points(points)
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be >= 1")
    _, kept = _to_o3d(points).remove_radius_outlier(int(min_neighbors), float(radius))
    return _mask_from_kept_indices(points.shape[0], kept)


def noise_mask_statistical(points, nb_neighbors=20, std_ratio=2.0):
    points = _validate_points(points)
    if nb_neighbors < 2:
        raise ValueError("nb_neighbors must be >= 2")
    if std_ratio <= 0:
        raise ValueError("std_ratio must be > 0")
    _, kept = _to_o3d(points).remove_statistical_outlier(int(nb_neighbors), float(std_ratio))
    return _mask_from_kept_indices(points.shape[0], kept)


def noise_mask_voxel_count(points, voxel_size=0.10, min_points=4):
    points = _validate_points(points)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    if min_points < 1:
        raise ValueError("min_points must be >= 1")
    ijk = np.floor(points / voxel_size).astype(np.int64)
    ijk -= ijk.min(axis=0)
    extent = ijk.max(axis=0) + 1
    key = (ijk[:, 0] * extent[1] + ijk[:, 1]) * extent[2] + ijk[:, 2]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inverse] < int(min_points)


def noise_mask_cluster(points, eps=0.10, min_points=4, min_cluster_size=50):
    points = _validate_points(points)
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if min_points < 1 or min_cluster_size < 1:
        raise ValueError("min_points and min_cluster_size must be >= 1")
    labels = np.asarray(_to_o3d(points).cluster_dbscan(float(eps), int(min_points)))
    noise = labels < 0
    valid = labels[~noise]
    if valid.size:
        sizes = np.bincount(valid)
        noise[~noise] = sizes[valid] < int(min_cluster_size)
    return noise


def noise_mask_plane_fit(points, knn=8, nsigma=3.0, max_error=0.0):
    """CloudCompare's noise filter: how far is each point off the surface its
    neighbours describe?

    Fit a plane through the `knn` nearest neighbours and measure the point's
    distance to it. With `max_error` at 0 that distance is judged against the
    neighbours' own spread about the plane, so the bar moves with the surface:
    strict where the scan is smooth, forgiving where it is rough. Give
    `max_error` a distance and it is judged against that instead, which is what
    you want when you know the scanner's accuracy.

    The point itself is left out of the fit. Including it bounds how far it can
    appear to be from a plane it helped define -- with 9 samples nothing can
    exceed 2.67 sigma, so nsigma=3 would remove nothing at all, whatever the
    data.

    `nsigma` is not a plain z-score, and reads stricter than it looks: the
    neighbours' spread is shrunk by having been fitted to, while the point's
    distance carries the plane's estimation error on top of its own noise. On a
    cleanly noisy surface nsigma=3 takes a few percent, not the 0.27% the
    normal distribution would suggest.
    """
    points = _validate_points(points)
    knn = int(knn)
    if knn < 3:
        raise ValueError("knn must be >= 3")  # three points define a plane
    if nsigma <= 0:
        raise ValueError("nsigma must be > 0")
    if max_error < 0:
        raise ValueError("max_error must be >= 0")
    if points.shape[0] <= knn:
        return np.zeros(points.shape[0], dtype=bool)  # nothing to compare against

    from scipy.spatial import cKDTree  # noqa: PLC0415 - keeps import cost off startup

    tree = cKDTree(points)
    n = points.shape[0]
    distance = np.empty(n)
    spread = np.empty(n)
    # The neighbour coordinates are n x knn x 3; on a few million points that
    # is gigabytes at once, so it is built a slice at a time.
    for start in range(0, n, _PLANE_FIT_CHUNK):
        stop = min(start + _PLANE_FIT_CHUNK, n)
        _, index = tree.query(points[start:stop], k=knn + 1, workers=-1)
        neighbours = points[index[:, 1:]]  # column 0 is the query point itself
        centre = neighbours.mean(axis=1)
        centred = neighbours - centre[:, None, :]
        covariance = np.einsum("mkj,mkl->mjl", centred, centred) / knn
        # eigh returns ascending eigenvalues, so the first vector is the
        # direction the neighbourhood varies least in: the plane's normal.
        normal = np.linalg.eigh(covariance)[1][:, :, 0]
        distance[start:stop] = np.abs(
            np.einsum("mj,mj->m", points[start:stop] - centre, normal)
        )
        offsets = np.einsum("mkj,mj->mk", centred, normal)
        # Exactly coplanar neighbours have no spread, and a threshold of zero
        # would condemn every point not coplanar to the last bit -- which on
        # synthetic data is all of them, and on real data none. Floor the
        # spread at a hair of the neighbourhood's own size: far below any real
        # roughness, far above the rounding error of a flat surface.
        scale = np.linalg.norm(centred, axis=2).mean(axis=1)
        spread[start:stop] = np.maximum(offsets.std(axis=1), scale * _PLANE_FIT_FLAT_TOL)

    if max_error > 0:
        return distance > float(max_error)
    return distance > float(nsigma) * spread


METHODS = {
    "cluster": noise_mask_cluster,
    "radius": noise_mask_radius,
    "statistical": noise_mask_statistical,
    "voxel_count": noise_mask_voxel_count,
    "plane_fit": noise_mask_plane_fit,
}


def estimate_noise_mask(points, method, **params):
    if method not in METHODS:
        raise ValueError(f"Unknown noise removal method '{method}'")
    return METHODS[method](points, **params)
