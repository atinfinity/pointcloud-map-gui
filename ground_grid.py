"""Ground-plane reference grid geometry (pure math, no Open3D GUI dependency)."""
import numpy as np

MARGIN_CELLS = 5


def nice_step(rough_step):
    """Round rough_step up to a "nice" 1/2/5 * 10^n grid unit."""
    if rough_step <= 0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(rough_step))
    residual = rough_step / magnitude
    if residual < 1.5:
        nice = 1.0
    elif residual < 3.5:
        nice = 2.0
    elif residual < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * magnitude


def build_ground_grid_lines(min_x, min_y, max_x, max_y):
    """Compute 2D grid line endpoints covering the given bounds.

    The bounds are extended outward to whole multiples of an auto-picked
    step -- so the result is always a complete rectangle of whole cells,
    never a partial/ragged one -- plus a fixed MARGIN_CELLS-wide border of
    extra whole cells on every side, so the grid reads clearly past the
    point cloud's own edge.

    Returns (points, lines, extended_bounds) where points/lines are plain
    lists suitable for o3d.utility.Vector*Vector, and extended_bounds is
    (ext_min_x, ext_min_y, ext_max_x, ext_max_y, step).
    """
    step = nice_step(max(max_x - min_x, max_y - min_y) / 10.0)
    margin = MARGIN_CELLS * step
    ext_min_x = np.floor(min_x / step) * step - margin
    ext_max_x = np.ceil(max_x / step) * step + margin
    ext_min_y = np.floor(min_y / step) * step - margin
    ext_max_y = np.ceil(max_y / step) * step + margin

    eps = step * 1e-6
    xs = np.arange(ext_min_x, ext_max_x + eps, step)
    ys = np.arange(ext_min_y, ext_max_y + eps, step)

    points = []
    lines = []
    for x in xs:
        points.append([x, ext_min_y])
        points.append([x, ext_max_y])
        lines.append([len(points) - 2, len(points) - 1])
    for y in ys:
        points.append([ext_min_x, y])
        points.append([ext_max_x, y])
        lines.append([len(points) - 2, len(points) - 1])

    return points, lines, (ext_min_x, ext_min_y, ext_max_x, ext_max_y, step)
