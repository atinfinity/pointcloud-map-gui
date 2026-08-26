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
_MAX_PASSES = 5
# Thinning a cloud that only just exceeds the budget costs a search and a copy
# to remove a rounding error's worth of points. Draw it as it is.
_SLACK = 1.05
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


def _count_exponent(previous, voxel_size, kept, n_dims):
    """How fast the kept-point count falls as the voxel grows.

    The count goes as voxel_size**-p. Assuming p is the dimension count only
    holds for solid clouds; real scans are surfaces sitting in a 3D box, where
    p is nearer 2 and the assumption leaves the budget badly underfilled. Once
    two passes exist, measure p from them instead of assuming it.
    """
    if previous is None:
        return n_dims
    last_size, last_kept = previous
    if last_kept < 1 or kept < 1 or abs(np.log(voxel_size / last_size)) < 1e-6:
        return n_dims
    measured = -np.log(kept / last_kept) / np.log(voxel_size / last_size)
    # Nonsense slopes come from passes that hit the cell cap rather than from
    # the geometry, so keep it inside what a 1D-to-3D structure can produce.
    return float(np.clip(measured, 1.0, 3.0))


def select_display_indices(points, max_points):
    """Return indices of at most `max_points` points, spread evenly in space,
    or None when the whole cloud already fits (draw it as-is).

    Uses voxel subsampling -- one point kept per occupied voxel -- which keeps
    a uniform spatial density instead of leaving sparse regions even sparser
    the way random thinning does.
    """
    n = points.shape[0]
    if max_points <= 0 or n <= max_points * _SLACK:
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
    previous = None  # (voxel_size, kept) of the pass before, for the exponent
    for _ in range(_MAX_PASSES):
        voxel_size = _widen_to_cell_budget(voxel_size, extent, dims)
        index = _voxel_representatives(points, origin, extent, voxel_size)
        kept = index.shape[0]
        if kept <= max_points:
            if best is None or kept > best.shape[0]:
                best = index
            if kept >= _TOLERANCE * max_points:
                break
        target = max_points if kept <= max_points else _SAFETY * max_points
        exponent = _count_exponent(previous, voxel_size, kept, dims.size)
        previous = (voxel_size, kept)
        # The floor stops a near-empty pass from collapsing the voxel size in
        # one step, which would blow the cell budget and get widened right back.
        adjusted = voxel_size * max((kept / target) ** (1.0 / exponent), 0.5)
        if abs(adjusted - voxel_size) < 1e-4 * voxel_size:
            break  # the cell cap is pinning the voxel size; more passes cannot help
        voxel_size = adjusted

    if best is None:
        # Never got under budget (pathological geometry). A stride cannot
        # overshoot, so fall back to one.
        return np.arange(0, n, int(np.ceil(n / max_points)))[:max_points]
    return best
