"""Generate synthetic point clouds (a flat room, and a sloped-floor room) for
manually exercising the GUI: loading, height-filter visualization, and map
export. Produces both a .pcd and a .ply file.
"""
import os

import numpy as np
import open3d as o3d

OUT_DIR = os.path.dirname(__file__)


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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


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

    def wall(x0, y0, x1, y1, height=2.5):
        n_len = int(np.hypot(x1 - x0, y1 - y0) / 0.03)
        n_h = int(height / 0.03)
        t = np.linspace(0, 1, n_len)
        h = np.linspace(0, height, n_h)
        tt, hh = np.meshgrid(t, h)
        xx = x0 + (x1 - x0) * tt
        yy = y0 + (y1 - y0) * tt
        zz = floor_z(xx, yy) + hh
        return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    walls = np.vstack([wall(0, 0, 6, 0), wall(0, 6, 6, 6), wall(0, 0, 0, 6), wall(6, 0, 6, 6)])
    wall_color = np.tile([0.8, 0.8, 0.85], (walls.shape[0], 1))

    # Low box (0.3m tall) standing on the floor near the low end: its top
    # (z ~= 0.45) is below the floor at the high end (z ~= 0.9).
    bx, by = np.meshgrid(np.arange(1.0, 1.6, 0.02), np.arange(4.0, 4.6, 0.02))
    bx, by = bx.ravel(), by.ravel()
    top = np.stack([bx, by, floor_z(bx, by) + 0.3], axis=1)
    sides = []
    for x0, y0, x1, y1 in [(1.0, 4.0, 1.6, 4.0), (1.6, 4.0, 1.6, 4.6), (1.6, 4.6, 1.0, 4.6), (1.0, 4.6, 1.0, 4.0)]:
        sides.append(wall(x0, y0, x1, y1, height=0.3))
    box = np.vstack([top] + sides)
    box_color = np.tile([0.2, 0.7, 0.2], (box.shape[0], 1))

    # Floating obstacle, as in the flat room.
    ox, oy = np.meshgrid(np.arange(3.5, 4.5, 0.02), np.arange(2.0, 3.0, 0.02))
    ox, oy = ox.ravel(), oy.ravel()
    oz = floor_z(ox, oy) + rng.uniform(1.0, 1.8, size=ox.size)
    obstacle = np.stack([ox, oy, oz], axis=1)
    obstacle_color = np.tile([0.9, 0.2, 0.2], (obstacle.shape[0], 1))

    points = np.vstack([floor, walls, box, obstacle])
    colors = np.vstack([floor_color, wall_color, box_color, obstacle_color])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    for name, builder in (("sample_room", build_room_point_cloud), ("sample_slope", build_slope_point_cloud)):
        pcd = builder()
        pcd_path = os.path.join(OUT_DIR, f"{name}.pcd")
        ply_path = os.path.join(OUT_DIR, f"{name}.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Wrote {len(pcd.points)} points to:\n  {pcd_path}\n  {ply_path}")


if __name__ == "__main__":
    main()
