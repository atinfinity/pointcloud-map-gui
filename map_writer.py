"""Write ROS2 map_server / nav2 compatible PGM + YAML files."""
import os

import yaml


def write_pgm(path, grid):
    """Write a uint8 2D numpy array as a binary (P5) PGM file."""
    height, width = grid.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(grid.tobytes())


def write_yaml(path, pgm_filename, resolution, origin_x, origin_y):
    data = {
        "image": pgm_filename,
        "resolution": float(resolution),
        "origin": [float(origin_x), float(origin_y), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


def export_map(basepath, occupancy_result):
    """Write <basepath>.pgm and <basepath>.yaml for an OccupancyGridResult."""
    pgm_path = basepath + ".pgm"
    yaml_path = basepath + ".yaml"
    write_pgm(pgm_path, occupancy_result.grid)
    write_yaml(
        yaml_path,
        os.path.basename(pgm_path),
        occupancy_result.resolution,
        occupancy_result.origin_x,
        occupancy_result.origin_y,
    )
    return pgm_path, yaml_path
