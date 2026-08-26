"""Pick a bounded subset of points to draw, so the 3D view stays responsive
on clouds far larger than the screen can resolve.

This only ever affects what is drawn. The occupancy grid and the exported map
are always computed from the full cloud, so thinning the view costs nothing in
map fidelity.

Selection returns *indices* rather than new points on purpose: every per-point
array the GUI already maintains (height colors, ground mask) can then be
sliced with the same indices and stay aligned.
"""
import numpy as np

# Stop once a pass lands within this fraction of the budget. Chasing the last
# few percent costs another full pass over the cloud and buys nothing visible.
_TOLERANCE = 0.85
_MAX_PASSES = 4
# Aim slightly under budget when correcting downward, so a pass that lands a
# fraction of a percent over does not need several more to inch below.
_SAFETY = 0.95
# The lookup table below is allocated per voxel, not per occupied voxel. Clouds
# are surfaces, so most voxels stay empty and the grid legitimately needs many
# more cells than the points it yields -- but not unboundedly many. This caps
# the table at 128 MB of int32; past it the voxel size is widened instead,
# which returns fewer points than asked for rather than exhausting memory.
_MAX_CELLS = 1 << 25


def _grid_shape(extent, voxel_size):
    return tuple(int(e / voxel_size) + 1 for e in extent)


def _voxel_representatives(points, origin, extent, voxel_size):
    """Index of one arbitrary point per occupied voxel, ordered by voxel.

    Scatters point indices into a dense per-voxel table rather than sorting
    the voxel keys: it is O(N) instead of O(N log N), and measured ~7x faster
    than np.unique on a five-million-point cloud.
    """
    inv = 1.0 / voxel_size
    _, ny, nz = _grid_shape(extent, voxel_size)
    # Points sit at or above `origin`, so truncation is already a floor, and
    # folding the axes in one at a time avoids materializing an (N, 3) integer
    # copy of the whole cloud.
    flat = ((points[:, 0] - origin[0]) * inv).astype(np.int64)
    flat *= ny
    flat += ((points[:, 1] - origin[1]) * inv).astype(np.int64)
    flat *= nz
    flat += ((points[:, 2] - origin[2]) * inv).astype(np.int64)

    table = np.full(int(np.prod(_grid_shape(extent, voxel_size))), -1, dtype=np.int32)
    # Later writes win, so each voxel keeps whichever of its points came last.
    # Any one of them represents the voxel equally well.
    table[flat] = np.arange(points.shape[0], dtype=np.int32)
    return table[table >= 0]


def _widen_to_cell_budget(voxel_size, extent, dims):
    """Raise `voxel_size` until the voxel grid fits in _MAX_CELLS."""
    cells = np.prod(_grid_shape(extent, voxel_size), dtype=np.float64)
    if cells <= _MAX_CELLS:
        return voxel_size
    # Scale by the cube (or square) root of the overshoot rather than stepping
    # blindly: a fixed step either overshoots -- costing points the budget
    # would have allowed -- or takes many iterations to clear the cap.
    voxel_size *= float((cells / _MAX_CELLS) ** (1.0 / dims.size))
    # The per-axis "+1" makes the true count exceed the closed form on thin
    # clouds, so confirm rather than trust it.
    while np.prod(_grid_shape(extent, voxel_size), dtype=np.float64) > _MAX_CELLS:
        voxel_size *= 1.1
    return voxel_size


def select_display_indices(points, max_points):
    """Return indices of at most `max_points` points, spread evenly in space,
    or None when the whole cloud already fits (draw it as-is).

    Uses voxel subsampling -- one point kept per occupied voxel -- which keeps
    a uniform spatial density instead of leaving sparse regions even sparser
    the way random thinning does.
    """
    n = points.shape[0]
    if max_points <= 0 or n <= max_points:
        return None

    origin = points.min(axis=0)
    extent = points.max(axis=0) - origin
    dims = extent[extent > 0.0]
    if dims.size == 0:  # every point coincides; one of them stands for all
        return np.arange(min(n, max_points))

    # Seed so the grid holds about `max_points` cells. Occupied cells are
    # fewer than that (real clouds are surfaces, not solids), so the search
    # below mostly refines downward from here -- usually once.
    voxel_size = float((dims.prod() / max_points) ** (1.0 / dims.size))
    best = None
    for _ in range(_MAX_PASSES):
        voxel_size = _widen_to_cell_budget(voxel_size, extent, dims)
        index = _voxel_representatives(points, origin, extent, voxel_size)
        kept = index.shape[0]
        if kept <= max_points:
            if best is None or kept > best.shape[0]:
                best = index
            if kept >= _TOLERANCE * max_points:
                break
        # Occupied-cell count moves with voxel_size**-dims.size, so this
        # converges from either side; the floor stops a near-empty pass from
        # collapsing the voxel size in one step.
        target = max_points if kept <= max_points else _SAFETY * max_points
        adjusted = voxel_size * max((kept / target) ** (1.0 / dims.size), 0.5)
        if abs(adjusted - voxel_size) < 1e-4 * voxel_size:
            break  # the cell cap is pinning the voxel size; more passes cannot help
        voxel_size = adjusted

    if best is None:
        # Never got under budget (pathological geometry). A stride cannot
        # overshoot, so fall back to one.
        return np.arange(0, n, int(np.ceil(n / max_points)))[:max_points]
    return best
