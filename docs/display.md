# Display thinning

How the 3D view keeps up with clouds far larger than the screen can
resolve, and what that costs.

Drawing every point of a multi-million-point cloud costs far more than it
shows -- the screen cannot resolve that many points anyway -- so the 3D view
is capped at "Max display points" (default 1,000,000). Above that, points are
voxel-subsampled for drawing: one point per voxel, which keeps a uniform
spatial density rather than leaving sparse regions sparser the way random
thinning would.

**This only affects what is drawn.** The occupancy grid, its preview, and the
exported map are always computed from every point, so thinning costs nothing
in map fidelity. Raise the limit to inspect fine detail, or set it to 0 to
disable thinning entirely.

Measured on `sample_data/sample_large_site.pcd` (1,554,944 points), dragging a
height slider:

| Max display points | Points drawn | Height-filter update |
|---|---:|---:|
| 500,000 | 474,242 | 7.0 ms (144 fps) |
| 1,000,000 (default) | 491,181 | 7.1 ms (142 fps) |
| 0 (no limit) | 1,554,944 | 21.1 ms (47 fps) |

Before the geometry was made updatable in place, the same drag cost 163 ms per
tick (6.1 fps) on that cloud, and 786 ms (1.3 fps) on a five-million-point one.

Fewer points are drawn than the limit allows because the voxel grid only
divides so finely before its lookup table would outgrow the memory budget; the
panel always reports the count actually being drawn.

Reproduce with:

```bash
uv run python -m pointcloud_map_gui.tools.benchmark_display                              # the committed cloud
uv run python -m pointcloud_map_gui.tools.benchmark_display --generate --points 5000000  # a generated one
```

Check out the commit before this change and run the same script to get the
"before" column; it detects which version of the code it is on.
