"""Ground (floor) point detection for point clouds whose floor is not flat.

Three interchangeable methods, all sharing the signature

    method(points: (N, 3) float array, **params) -> (N,) bool ground mask

- ``local_grid``: per-cell minimum-Z DEM, one 3x3 morphological opening,
  bilinear sampling, fixed thickness threshold. Pure numpy.
- ``pmf``: Progressive Morphological Filter (Zhang et al. 2003) on the same
  DEM with a growing opening window and a slope-aware height threshold.
  Pure numpy.
- ``csf``: Cloth Simulation Filter (Zhang et al. 2016) via the
  ``cloth-simulation-filter`` package.

The DEM helpers are pure numpy (no scipy) so this module stays testable
headless alongside occupancy_grid.py.
"""
import numpy as np

# Parameter metadata shared by the GUI and the CLI: name -> (default, min, max).
DEFAULT_PARAMS = {
    "local_grid": {
        "cell_size": (0.5, 0.05, 5.0),
        "thickness": (0.10, 0.0, 2.0),
    },
    "pmf": {
        "cell_size": (0.5, 0.05, 5.0),
        "max_window": (33, 3, 201),
        "slope": (0.3, 0.0, 5.0),
        "initial_distance": (0.10, 0.0, 2.0),
        "max_distance": (1.0, 0.0, 10.0),
    },
    "csf": {
        "cloth_resolution": (0.5, 0.05, 5.0),
        "rigidness": (2, 1, 3),
        "class_threshold": (0.10, 0.0, 2.0),
        "slope_smooth": (1, 0, 1),
    },
}

METHOD_LABELS = {
    "local_grid": "Local grid (min-Z + opening)",
    "pmf": "PMF (progressive morphological)",
    "csf": "CSF (cloth simulation)",
}


def default_params(method):
    return {name: spec[0] for name, spec in DEFAULT_PARAMS[method].items()}


# ----------------------------------------------------------------------
# DEM helpers
# ----------------------------------------------------------------------
def _validate_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty")
    return points


def _build_min_z_dem(points, cell_size):
    """Rasterize the minimum Z of every XY cell. Empty cells are NaN.

    Returns (dem, origin_xy) where dem has shape (rows, cols) indexed as
    dem[row(y), col(x)].
    """
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    origin = np.array([xs.min(), ys.min()])
    cols = np.floor((xs - origin[0]) / cell_size).astype(np.int64)
    rows = np.floor((ys - origin[1]) / cell_size).astype(np.int64)
    n_cols = int(cols.max()) + 1
    n_rows = int(rows.max()) + 1

    cell_ids = rows * n_cols + cols
    order = np.argsort(cell_ids, kind="stable")
    sorted_ids = cell_ids[order]
    sorted_z = zs[order]
    unique_ids, first = np.unique(sorted_ids, return_index=True)
    min_z = np.minimum.reduceat(sorted_z, first)

    dem = np.full(n_rows * n_cols, np.nan)
    dem[unique_ids] = min_z
    return dem.reshape(n_rows, n_cols), origin


def _min_filter(a, window):
    """Square min filter (edge-padded), implemented separably."""
    if window <= 1:
        return a
    half = window // 2
    p = np.pad(a, half, mode="edge")
    rows = np.lib.stride_tricks.sliding_window_view(p, window, axis=0).min(axis=-1)
    return np.lib.stride_tricks.sliding_window_view(rows, window, axis=1).min(axis=-1)


def _max_filter(a, window):
    return -_min_filter(-a, window)


def _morphological_open(a, window):
    return _max_filter(_min_filter(a, window), window)


def _fill_nan(dem):
    """Fill empty cells by propagating the minimum of valid neighbours outward."""
    dem = dem.copy()
    missing = np.isnan(dem)
    if not missing.any():
        return dem
    if missing.all():
        raise ValueError("DEM has no valid cells")
    work = np.where(missing, np.inf, dem)
    while np.isinf(work).any():
        grown = _min_filter(work, 3)
        still = np.isinf(work)
        work[still] = grown[still]
    return work


def _bilinear_sample(dem, xy, origin, cell_size):
    """Sample a DEM (cell-centred samples) at XY positions with bilinear
    interpolation, clamping outside the DEM extent."""
    n_rows, n_cols = dem.shape
    u = (xy[:, 0] - origin[0]) / cell_size - 0.5
    v = (xy[:, 1] - origin[1]) / cell_size - 0.5
    u = np.clip(u, 0.0, n_cols - 1)
    v = np.clip(v, 0.0, n_rows - 1)
    c0 = np.floor(u).astype(np.int64)
    r0 = np.floor(v).astype(np.int64)
    c1 = np.minimum(c0 + 1, n_cols - 1)
    r1 = np.minimum(r0 + 1, n_rows - 1)
    fu = u - c0
    fv = v - r0
    top = dem[r0, c0] * (1 - fu) + dem[r0, c1] * fu
    bottom = dem[r1, c0] * (1 - fu) + dem[r1, c1] * fu
    return top * (1 - fv) + bottom * fv


# ----------------------------------------------------------------------
# Method C: local grid
# ----------------------------------------------------------------------
def ground_mask_local_grid(points, cell_size=0.5, thickness=0.10):
    points = _validate_points(points)
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    if thickness < 0:
        raise ValueError("thickness must be >= 0")

    dem, origin = _build_min_z_dem(points, cell_size)
    dem = _fill_nan(dem)
    surface = _morphological_open(dem, 3)
    ground_z = _bilinear_sample(surface, points[:, :2], origin, cell_size)
    return (points[:, 2] - ground_z) <= thickness


# ----------------------------------------------------------------------
# Method E: progressive morphological filter
# ----------------------------------------------------------------------
def _pmf_windows(max_window, base=2):
    """Exponentially growing odd window sizes 3, 5, 9, 17, ... <= max_window."""
    windows = []
    k = 0
    while True:
        w = 2 * base**k + 1
        if w > max_window:
            break
        windows.append(w)
        k += 1
    if not windows:
        windows = [3]
    return windows


def ground_mask_pmf(
    points,
    cell_size=0.5,
    max_window=33,
    slope=0.3,
    initial_distance=0.10,
    max_distance=1.0,
):
    points = _validate_points(points)
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    if max_window < 3:
        raise ValueError("max_window must be >= 3")
    if slope < 0 or initial_distance < 0 or max_distance < 0:
        raise ValueError("slope, initial_distance and max_distance must be >= 0")

    dem, origin = _build_min_z_dem(points, cell_size)
    surface = _fill_nan(dem)

    prev_w = 1
    for w in _pmf_windows(int(max_window)):
        opened = _morphological_open(surface, w)
        dh = min(slope * (w - prev_w) * cell_size + initial_distance, max_distance)
        # Cells rising more than dh above the opened surface are objects:
        # replace them with the opened (ground) estimate. Cells within dh
        # keep their measured value so terrain detail survives.
        non_ground = (surface - opened) > dh
        surface = np.where(non_ground, opened, surface)
        prev_w = w

    ground_z = _bilinear_sample(surface, points[:, :2], origin, cell_size)
    return (points[:, 2] - ground_z) <= initial_distance


# ----------------------------------------------------------------------
# Method D: cloth simulation filter
# ----------------------------------------------------------------------
def ground_mask_csf(
    points,
    cloth_resolution=0.5,
    rigidness=2,
    class_threshold=0.10,
    slope_smooth=1,
):
    points = _validate_points(points)
    if cloth_resolution <= 0:
        raise ValueError("cloth_resolution must be > 0")
    if int(rigidness) not in (1, 2, 3):
        raise ValueError("rigidness must be 1, 2 or 3")
    if class_threshold < 0:
        raise ValueError("class_threshold must be >= 0")
    try:
        import CSF
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "CSF method requires the 'cloth-simulation-filter' package (uv sync)"
        ) from e

    csf = CSF.CSF()
    csf.params.bSloopSmooth = bool(slope_smooth)
    csf.params.cloth_resolution = float(cloth_resolution)
    csf.params.rigidness = int(rigidness)
    csf.params.class_threshold = float(class_threshold)
    csf.setPointCloud(np.ascontiguousarray(points))
    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx, False)  # no cloth_nodes.txt

    mask = np.zeros(points.shape[0], dtype=bool)
    mask[np.asarray(list(ground_idx), dtype=np.int64)] = True
    return mask


METHODS = {
    "local_grid": ground_mask_local_grid,
    "pmf": ground_mask_pmf,
    "csf": ground_mask_csf,
}


def estimate_ground_mask(points, method, **params):
    if method not in METHODS:
        raise ValueError(f"Unknown ground removal method '{method}'")
    return METHODS[method](points, **params)
