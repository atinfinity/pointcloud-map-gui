"""Point cloud file I/O via Open3D (supports .pcd and .ply)."""
import os

import open3d as o3d

SUPPORTED_EXTENSIONS = (".pcd", ".ply")


def load_point_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Expected .pcd or .ply")

    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Loaded point cloud from '{path}' is empty")
    return pcd
