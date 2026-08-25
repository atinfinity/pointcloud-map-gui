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
"""
import numpy as np

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
}

METHOD_LABELS = {
    "cluster": "DBSCAN cluster size",
    "radius": "Radius outlier",
    "statistical": "Statistical outlier",
    "voxel_count": "Voxel point count",
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


METHODS = {
    "cluster": noise_mask_cluster,
    "radius": noise_mask_radius,
    "statistical": noise_mask_statistical,
    "voxel_count": noise_mask_voxel_count,
}


def estimate_noise_mask(points, method, **params):
    if method not in METHODS:
        raise ValueError(f"Unknown noise removal method '{method}'")
    return METHODS[method](points, **params)
