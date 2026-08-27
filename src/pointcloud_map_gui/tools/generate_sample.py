"""Generate synthetic point clouds (a flat room, a sloped-floor room, the
sloped room with isolated noise, and a room mixing flat floors with a ramp)
for manually exercising the GUI: loading, height-filter visualization, ground
removal, and map export. Produces both a .pcd and a .ply file.
"""
import argparse
import os

import numpy as np
import open3d as o3d

from .. import paths



def _wall_on_floor(floor_z, x0, y0, x1, y1, height=2.5, step=0.03):
    """Vertical wall from (x0, y0) to (x1, y1) standing on floor_z(x, y)."""
    n_len = int(np.hypot(x1 - x0, y1 - y0) / step)
    n_h = int(height / step)
    t = np.linspace(0, 1, n_len)
    h = np.linspace(0, height, n_h)
    tt, hh = np.meshgrid(t, h)
    xx = x0 + (x1 - x0) * tt
    yy = y0 + (y1 - y0) * tt
    zz = floor_z(xx, yy) + hh
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def _box_on_floor(floor_z, x0, y0, x1, y1, height, step=0.02):
    """Closed box (top + 4 sides) of the given height standing on floor_z."""
    bx, by = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    bx, by = bx.ravel(), by.ravel()
    top = np.stack([bx, by, floor_z(bx, by) + height], axis=1)
    sides = [
        _wall_on_floor(floor_z, *edge, height=height)
        for edge in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
    ]
    return np.vstack([top] + sides)


def _point_cloud(points, colors=None):
    """Build a cloud whose coordinates are exactly representable in float32.

    PLY stores coordinates as double, PCD as float32. Left at full float64
    precision the two formats disagree about the same cloud across machines:
    `rng.normal` goes through the ziggurat's exp/log, and those come from the
    platform's libm, so macOS arm64 and Linux x86_64 differ by one to four ULP
    -- 1e-16 m, invisible in the .pcd because float32 rounds it away, but baked
    into the .ply. Rounding here makes both formats carry the same numbers and
    regenerate to the same bytes anywhere. Coordinates are metres over a few
    tens of metres, so float32 leaves sub-micron precision; there is nothing to
    lose.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32).astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def build_room_point_cloud():
    rng = np.random.default_rng(42)

    # Floor: 6x6m plane at z=0.
    fx, fy = np.meshgrid(np.arange(0, 6, 0.03), np.arange(0, 6, 0.03))
    floor = np.stack([fx.ravel(), fy.ravel(), np.zeros(fx.size)], axis=1)
    floor_color = np.tile([0.6, 0.5, 0.4], (floor.shape[0], 1))

    def wall(x0, y0, x1, y1, z_max=2.5):
        n_len = int(np.hypot(x1 - x0, y1 - y0) / 0.03)
        n_h = int(z_max / 0.03)
        t = np.linspace(0, 1, n_len)
        z = np.linspace(0, z_max, n_h)
        tt, zz = np.meshgrid(t, z)
        xx = x0 + (x1 - x0) * tt
        yy = y0 + (y1 - y0) * tt
        return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    walls = np.vstack(
        [
            wall(0, 0, 6, 0),
            wall(0, 6, 6, 6),
            wall(0, 0, 0, 6),
            wall(6, 0, 6, 6),
        ]
    )
    wall_color = np.tile([0.8, 0.8, 0.85], (walls.shape[0], 1))

    # A floating obstacle (e.g. a shelf) between 1.0m and 1.8m height, so it can
    # be isolated by adjusting the min/max height sliders independently of the
    # floor and full walls.
    ox, oy = np.meshgrid(np.arange(2.5, 3.5, 0.02), np.arange(2.5, 3.5, 0.02))
    oz = rng.uniform(1.0, 1.8, size=ox.size)
    obstacle = np.stack([ox.ravel(), oy.ravel(), oz], axis=1)
    obstacle_color = np.tile([0.9, 0.2, 0.2], (obstacle.shape[0], 1))

    points = np.vstack([floor, walls, obstacle])
    colors = np.vstack([floor_color, wall_color, obstacle_color])

    return _point_cloud(points, colors)


def build_slope_point_cloud():
    """A room whose floor rises 15% along X with noise and gentle waves, plus
    a low box sitting on the floor. A plain height filter cannot separate
    the box from the higher end of the floor; ground removal can.
    """
    rng = np.random.default_rng(7)

    def floor_z(x, y):
        return 0.15 * x + 0.03 * np.sin(2.0 * np.pi * y / 3.0)

    fx, fy = np.meshgrid(np.arange(0, 6, 0.03), np.arange(0, 6, 0.03))
    fx, fy = fx.ravel(), fy.ravel()
    fz = floor_z(fx, fy) + rng.normal(0.0, 0.01, size=fx.size)
    floor = np.stack([fx, fy, fz], axis=1)
    floor_color = np.tile([0.6, 0.5, 0.4], (floor.shape[0], 1))

    walls = np.vstack(
        [_wall_on_floor(floor_z, *edge) for edge in [(0, 0, 6, 0), (0, 6, 6, 6), (0, 0, 0, 6), (6, 0, 6, 6)]]
    )
    wall_color = np.tile([0.8, 0.8, 0.85], (walls.shape[0], 1))

    # Low box (0.3m tall) standing on the floor near the low end: its top
    # (z ~= 0.45) is below the floor at the high end (z ~= 0.9).
    box = _box_on_floor(floor_z, 1.0, 4.0, 1.6, 4.6, height=0.3)
    box_color = np.tile([0.2, 0.7, 0.2], (box.shape[0], 1))

    # Floating obstacle, as in the flat room.
    ox, oy = np.meshgrid(np.arange(3.5, 4.5, 0.02), np.arange(2.0, 3.0, 0.02))
    ox, oy = ox.ravel(), oy.ravel()
    oz = floor_z(ox, oy) + rng.uniform(1.0, 1.8, size=ox.size)
    obstacle = np.stack([ox, oy, oz], axis=1)
    obstacle_color = np.tile([0.9, 0.2, 0.2], (obstacle.shape[0], 1))

    points = np.vstack([floor, walls, box, obstacle])
    colors = np.vstack([floor_color, wall_color, box_color, obstacle_color])

    return _point_cloud(points, colors)


def build_noisy_slope_point_cloud():
    """The sloped room plus isolated noise: 500 uniformly scattered points,
    30 tiny clusters of 2-5 points and 20 speck clumps of 10-40 points, for
    exercising noise removal."""
    rng = np.random.default_rng(11)
    base = build_slope_point_cloud()
    points = np.asarray(base.points)
    colors = np.asarray(base.colors)

    scatter = rng.uniform([0.0, 0.0, -0.5], [6.0, 6.0, 3.0], size=(500, 3))
    clusters = []
    for _ in range(30):
        center = rng.uniform([0.5, 0.5, 0.5], [5.5, 5.5, 2.5])
        size = rng.integers(2, 6)
        clusters.append(center + rng.normal(0.0, 0.01, size=(size, 3)))
    # Medium "speck" clumps of 10-40 points: dense enough to survive
    # per-point outlier tests, small enough to be dropped by cluster size.
    for _ in range(20):
        center = rng.uniform([0.5, 0.5, 0.5], [5.5, 5.5, 2.5])
        size = rng.integers(10, 41)
        clusters.append(center + rng.normal(0.0, 0.02, size=(size, 3)))
    noise = np.vstack([scatter] + clusters)
    noise_color = np.tile([1.0, 0.0, 1.0], (noise.shape[0], 1))

    return _point_cloud(np.vstack([points, noise]), np.vstack([colors, noise_color]))


def build_ramp_point_cloud():
    """A room mixing flat and sloped floor: a flat lower level (x < 2, z=0),
    a 35% ramp (2 <= x < 4) rising 0.7m, and a flat upper level (x >= 4,
    z=0.7), with slight noise. Low boxes stand on each flat level and one on
    the ramp itself, plus a floating obstacle. The flat/slope breaklines are
    where DEM-based ground removal tends to misclassify: too-large windows or
    thresholds either eat the box on the ramp or flag the ramp as an object.
    """
    rng = np.random.default_rng(23)

    def floor_z(x, y):
        return np.clip((x - 2.0) * 0.35, 0.0, 0.7)

    fx, fy = np.meshgrid(np.arange(0, 6, 0.03), np.arange(0, 6, 0.03))
    fx, fy = fx.ravel(), fy.ravel()
    fz = floor_z(fx, fy) + rng.normal(0.0, 0.005, size=fx.size)
    floor = np.stack([fx, fy, fz], axis=1)
    floor_color = np.tile([0.6, 0.5, 0.4], (floor.shape[0], 1))

    walls = np.vstack(
        [_wall_on_floor(floor_z, *edge) for edge in [(0, 0, 6, 0), (0, 6, 6, 6), (0, 0, 0, 6), (6, 0, 6, 6)]]
    )
    wall_color = np.tile([0.8, 0.8, 0.85], (walls.shape[0], 1))

    # 0.3m boxes: one on the lower flat, one on the ramp, one on the upper
    # flat. The lower box's top (z=0.3) is below the upper floor (z=0.7), so
    # a plain height filter cannot keep it while dropping the upper floor.
    boxes = np.vstack(
        [
            _box_on_floor(floor_z, 0.7, 1.0, 1.3, 1.6, height=0.3),
            _box_on_floor(floor_z, 2.7, 4.2, 3.3, 4.8, height=0.3),
            _box_on_floor(floor_z, 4.7, 2.2, 5.3, 2.8, height=0.3),
        ]
    )
    box_color = np.tile([0.2, 0.7, 0.2], (boxes.shape[0], 1))

    ox, oy = np.meshgrid(np.arange(1.0, 2.0, 0.02), np.arange(4.0, 5.0, 0.02))
    ox, oy = ox.ravel(), oy.ravel()
    oz = floor_z(ox, oy) + rng.uniform(1.0, 1.8, size=ox.size)
    obstacle = np.stack([ox, oy, oz], axis=1)
    obstacle_color = np.tile([0.9, 0.2, 0.2], (obstacle.shape[0], 1))

    points = np.vstack([floor, walls, boxes, obstacle])
    colors = np.vstack([floor_color, wall_color, box_color, obstacle_color])

    return _point_cloud(points, colors)


# --- large site ------------------------------------------------------------
# A 50 x 50 m outdoor yard, for exercising the tool at survey scale: sparser
# than an indoor scan but spread over 25x the area. Everything in it exists so
# the exported map can be checked by eye -- buildings are hollow with doorways,
# one shed was never entered, and two patches of ground sit in the occlusion
# shadow of a building, so `free`, `occupied` and `unknown` all have to appear
# in recognizable shapes.
SITE_SIZE = 50.0
SITE_FENCE_HEIGHT = 1.2
SITE_UNDULATION = 0.05  # gentle ground relief, enough to give ground removal work
# Ground spacing has to stay *below* the map resolution it will be gridded at.
# A lattice of spacing s puts a point in every cell of size c only while s < c;
# at s > c the cells it skips come out `unknown`, and the map fills with a mesh
# of speckle that looks like a bug. 0.045 covers the 0.05 m/cell default.
SITE_GROUND_STEP = 0.045
SITE_STRUCTURE_STEP = 0.05
# (x0, y0, x1, y1), wall height, and the doorway the scanner drove in through
# as (side, start along that side, width) -- None for a building it could only
# see from outside, whose interior therefore has to come out `unknown`.
# No roofs: a ground-level scanner sees walls, not the tops of buildings. It
# also matters for the map, because a roof point would mark the cells under it
# as scanned and the sealed shed would read as free, not unknown.
SITE_BUILDINGS = [
    ((5.0, 5.0, 17.0, 13.0), 4.0, ("S", 4.0, 2.0)),
    ((30.0, 28.0, 40.0, 38.0), 5.0, ("W", 4.0, 2.5)),
    ((35.0, 8.0, 40.0, 12.0), 3.0, None),  # closed shed, never entered
]
# Ground the scanner never saw: shadows cast by the buildings it drove past.
SITE_UNSCANNED = [
    (18.0, 5.0, 26.0, 13.0),
    (30.0, 20.0, 40.0, 26.0),
]
# Shipping containers: (x0, y0, x1, y1, height).
SITE_CONTAINERS = [
    (6.0, 30.0, 12.0, 32.4, 2.6),
    (6.0, 34.0, 12.0, 36.4, 2.6),
    (22.0, 42.0, 28.0, 44.4, 2.6),
]


def _rect_wall_segments(x0, y0, x1, y1, door):
    """The four walls of a rectangle, with a gap left in one of them.

    Without the gap the interior would be sealed off and read as unknown; the
    doorway is what makes the room show up as scanned, hollow space.
    """
    side, start, width = door if door else (None, 0.0, 0.0)
    corners = {
        "S": ((x0, y0), (x1, y0)),
        "E": ((x1, y0), (x1, y1)),
        "N": ((x1, y1), (x0, y1)),
        "W": ((x0, y1), (x0, y0)),
    }
    segments = []
    for name, ((ax, ay), (bx, by)) in corners.items():
        if side is None or name != side:
            segments.append((ax, ay, bx, by))
            continue
        length = np.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / length, (by - ay) / length
        segments.append((ax, ay, ax + ux * start, ay + uy * start))
        end = start + width
        segments.append((ax + ux * end, ay + uy * end, bx, by))
    return segments


def build_large_site_point_cloud():
    """50 x 50 m yard with buildings, a fence, containers and occlusion gaps.

    Density is fixed rather than solved for a point count: the ground has to be
    sampled finer than the map resolution (see SITE_GROUND_STEP), and that
    constraint, not a target size, is what sets it.

    No colors, `.pcd` only: the GUI colors points by height, and a second copy
    of a cloud this size is not worth the repository space.
    """
    step = SITE_STRUCTURE_STEP
    size = SITE_SIZE

    def floor_z(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        # Smooth and deterministic -- a scanned yard is never dead flat, and
        # sine relief keeps the fixture reproducible without an RNG.
        return SITE_UNDULATION * np.sin(x / 7.0) * np.cos(y / 9.0)

    gx, gy = np.meshgrid(
        np.arange(0.0, size, SITE_GROUND_STEP), np.arange(0.0, size, SITE_GROUND_STEP)
    )
    gx, gy = gx.ravel(), gy.ravel()
    keep = np.ones(gx.shape, dtype=bool)
    holes = list(SITE_UNSCANNED)
    holes += [footprint for footprint, _, door in SITE_BUILDINGS if door is None]
    # Nothing sees the ground under a container, so those cells stay unknown
    # and the container reads as the solid obstacle it is instead of a hollow
    # outline with free space inside it.
    holes += [(x0, y0, x1, y1) for x0, y0, x1, y1, _ in SITE_CONTAINERS]
    for x0, y0, x1, y1 in holes:
        keep &= ~((gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1))
    parts = [np.stack([gx[keep], gy[keep], floor_z(gx[keep], gy[keep])], axis=1)]

    for x0, y0, x1, y1 in [
        (0.0, 0.0, size, 0.0),
        (size, 0.0, size, size),
        (size, size, 0.0, size),
        (0.0, size, 0.0, 0.0),
    ]:
        parts.append(
            _wall_on_floor(floor_z, x0, y0, x1, y1, height=SITE_FENCE_HEIGHT, step=step)
        )

    for (x0, y0, x1, y1), height, door in SITE_BUILDINGS:
        for seg in _rect_wall_segments(x0, y0, x1, y1, door):
            parts.append(_wall_on_floor(floor_z, *seg, height=height, step=step))

    for x0, y0, x1, y1, height in SITE_CONTAINERS:
        parts.append(_box_on_floor(floor_z, x0, y0, x1, y1, height, step=step))

    return _point_cloud(np.vstack(parts))


# --- benchmark fixture -----------------------------------------------------
# A warehouse floor plan, not a scene to look at: it exists so
# benchmark_display.py can build a cloud of any size on demand, which is how
# the five-million-point figures in the README are reproduced. Unlike the
# clouds above it is not written to disk by main() -- at that size the file
# would cost the repository more than regenerating it costs anyone.
# Rooms are still hollow and joined by doorways, so a map exported from it can
# be sanity-checked the same way: a solid block of "occupied" means something
# broke.
BENCHMARK_SIZE = 40.0
BENCHMARK_WALL_HEIGHT = 3.0
BENCHMARK_PILLAR_HEIGHT = 2.5
# Interior partitions, with gaps left between segments for doorways.
BENCHMARK_PARTITIONS = [
    (0.0, 20.0, 16.0, 20.0),
    (24.0, 20.0, 40.0, 20.0),
    (20.0, 20.0, 20.0, 32.0),
    (20.0, 36.0, 20.0, 40.0),
    (12.0, 0.0, 12.0, 8.0),
    (12.0, 12.0, 12.0, 20.0),
    (28.0, 0.0, 28.0, 13.0),
]
BENCHMARK_PILLARS = [(7.0, 26.0), (30.0, 30.0), (20.0, 8.0)]
BENCHMARK_PILLAR_SIDE = 1.0
# A room the scanner never entered, seen only through its doorway: no floor
# points fall inside it, so the map has to come out `unknown` there rather than
# `free`. Without it every cell in the grid is scanned and the fixture cannot
# tell the two apart.
BENCHMARK_UNSCANNED = (28.0, 0.0, 40.0, 13.0)


def _benchmark_step(n_points):
    """Sample spacing that lands the scene near `n_points`.

    Every surface below is sampled on a fixed grid, so the point count is the
    total area over step**2 -- solve that for the step rather than building the
    scene repeatedly to search for it.
    """
    outer = 4.0 * BENCHMARK_SIZE * BENCHMARK_WALL_HEIGHT
    partitions = sum(
        np.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in BENCHMARK_PARTITIONS
    ) * BENCHMARK_WALL_HEIGHT
    pillar = BENCHMARK_PILLAR_SIDE * (
        BENCHMARK_PILLAR_SIDE + 4.0 * BENCHMARK_PILLAR_HEIGHT
    )
    ux0, uy0, ux1, uy1 = BENCHMARK_UNSCANNED
    floor = BENCHMARK_SIZE**2 - (ux1 - ux0) * (uy1 - uy0)
    area = floor + outer + partitions + len(BENCHMARK_PILLARS) * pillar
    return float(np.sqrt(area / n_points))


def build_benchmark_point_cloud(n_points=2_000_000):
    """Large warehouse-like cloud for benchmarking the 3D view.

    No colors: the GUI colors points by height and ignores any the file
    carries, and dropping them keeps the committed file a third smaller.
    """
    step = _benchmark_step(n_points)
    size = BENCHMARK_SIZE

    def floor_z(x, y):
        return np.zeros_like(np.asarray(x, dtype=float))

    fx, fy = np.meshgrid(np.arange(0.0, size, step), np.arange(0.0, size, step))
    fx, fy = fx.ravel(), fy.ravel()
    ux0, uy0, ux1, uy1 = BENCHMARK_UNSCANNED
    scanned = ~((fx >= ux0) & (fx <= ux1) & (fy >= uy0) & (fy <= uy1))
    parts = [np.stack([fx[scanned], fy[scanned], np.zeros(int(scanned.sum()))], axis=1)]

    outer = [
        (0.0, 0.0, size, 0.0),
        (size, 0.0, size, size),
        (size, size, 0.0, size),
        (0.0, size, 0.0, 0.0),
    ]
    for x0, y0, x1, y1 in outer + BENCHMARK_PARTITIONS:
        parts.append(
            _wall_on_floor(floor_z, x0, y0, x1, y1, height=BENCHMARK_WALL_HEIGHT, step=step)
        )

    half = BENCHMARK_PILLAR_SIDE / 2.0
    for cx, cy in BENCHMARK_PILLARS:
        parts.append(
            _box_on_floor(
                floor_z, cx - half, cy - half, cx + half, cy + half,
                BENCHMARK_PILLAR_HEIGHT, step=step,
            )
        )

    return _point_cloud(np.vstack(parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where to write the clouds (default: the repository's sample_data/)",
    )
    args = parser.parse_args(argv)
    out_dir = args.out_dir or str(paths.require_sample_data_dir())
    os.makedirs(out_dir, exist_ok=True)
    for name, builder in (
        ("sample_room", build_room_point_cloud),
        ("sample_slope", build_slope_point_cloud),
        ("sample_slope_noisy", build_noisy_slope_point_cloud),
        ("sample_ramp", build_ramp_point_cloud),
    ):
        pcd = builder()
        pcd_path = os.path.join(out_dir, f"{name}.pcd")
        ply_path = os.path.join(out_dir, f"{name}.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Wrote {len(pcd.points)} points to:\n  {pcd_path}\n  {ply_path}")

    # .pcd only: a .ply of a cloud this size would be tens of MB more in the
    # repository for a file nothing reads.
    pcd = build_large_site_point_cloud()
    pcd_path = os.path.join(out_dir, "sample_large_site.pcd")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Wrote {len(pcd.points)} points to:\n  {pcd_path}")


if __name__ == "__main__":
    main()
