# Tuning guide: which method, which knob

Practical notes on choosing a noise-removal / ground-removal method and
which parameter to reach for when the result is wrong. Numbers quoted below
were measured on the bundled `sample_data/` clouds with
`pointcloud_map_gui.tools.compare_noise_removal` /
`...compare_ground_removal` (see [cli.md](cli.md)); re-run them on
your own data before trusting a rule of thumb.

Processing order is noise removal -> ground removal -> height filter ->
occupancy grid -> map cleanup, so fix problems in that order: a stray point
that survives noise removal can drag the DEM down, and a floor that is not
removed cannot be fixed by the height filter.

## 1. Decide what you actually need

| Symptom on the map | Fix at this stage |
|---|---|
| Map is far larger than the room; isolated `occupied`/`free` cells far from anything | Noise removal (scattered points stretch the bounds) |
| Small specks of `occupied` inside open floor, 1-4 cells wide | Noise removal `cluster`, or Map cleanup as a last resort |
| Whole floor area is `occupied` on one side of the room | Ground removal (sloped/uneven floor) |
| Low obstacle (box, step, pallet) disappears when the floor disappears | Ground removal, then tighten the thickness threshold |
| Walls look thin/broken | Height filter min too high, or noise removal too aggressive on sparse walls |
| Everything fine but an obstacle of a few cells vanishes | Map cleanup `min blob` too large |

## 2. Noise removal

### Which method

| Situation | Method | Why |
|---|---|---|
| Default / unsure | `cluster` | The only method that removes small dense clumps (10-40 points), which per-point tests keep. On `sample_slope_noisy` it removes 95% of tiny clusters *and* 95% of specks while deleting only 2 real points. |
| Only scattered single points, huge cloud, need speed | `radius` | 2x faster than `cluster`, catches 90% of scatter, but keeps 100% of the 10-40 point specks and clips ~400 real points on sparse walls. |
| Point density varies a lot across the scan (near vs far from the sensor) | `statistical` | Threshold adapts to the cloud-wide mean neighbour distance; less wall damage than `radius` (54 real points vs 436). Still keeps specks. |
| Millions of points, just want the gross outliers gone fast | `voxel_count` | Pure numpy, ~10x faster. Grid-aligned, so it eats thin structures: 2,891 real points removed on the sample. Use a coarse voxel and low `min_points`. |

### Which parameter

- Specks still on the map after `cluster` -> raise `min_cluster_size`
  (default 50). Every clump smaller than this is deleted, so look at the
  speck size in the 3D view first. Thin real structures (railings, cables,
  chair legs) disappearing -> lower it.
- `cluster` splits a real wall into small pieces and deletes them -> raise
  `eps` (default 0.10 m) to roughly 2-3x the point spacing on the walls.
| Points sit *on* the surfaces but scattered off them -- scanner range noise, a fuzzy wall | `plane_fit` | The only method that measures shape. On `sample_range_noise`, where 2.5% of points were displaced along the surface normal: `plane_fit` at the defaults removes 2,727 and `statistical` 2,038, but of what they remove `plane_fit` is right 57% of the time against `statistical`'s 63% at similar recall -- and with `max_error` 0.03 (a known scanner accuracy) `plane_fit` is right 98% of the time. `cluster` and `radius` find almost none of it: 3% and 8% recall. |
| Both kinds at once | `plane_fit` **and** `cluster` | They see different things and barely overlap -- IoU 0.03 on that cloud. Run noise removal twice, or pick the one matching the dominant problem. |

- `plane_fit` `nsigma` reads stricter than a z-score: the neighbours' spread
  is shrunk by having been fitted to, while the point's distance carries the
  plane's estimation error too. nsigma 3 takes a few percent of a cleanly
  noisy surface, not 0.27%. Raise `knn` before lowering `nsigma` -- 8 -> 20
  neighbours lifted precision from 0.20 to 0.57 at the same recall.
- `plane_fit` cannot see a point alone in space, by construction: its
  neighbours are then other scattered points, which describe no surface. Pair
  it with `radius` or `cluster` rather than expecting it to cover both.
- `radius`/`statistical` deleting wall points -> lower `min_neighbors` /
  raise `std_ratio`, or increase `radius` / `nb_neighbors` so sparse but
  real surfaces have enough neighbours.
- `cluster` too slow -> it is DBSCAN; a larger `eps` or pre-downsampling
  the cloud helps far more than `min_points`.
- "All points classified as noise" is rejected by the GUI; it usually means
  `eps`/`radius` is smaller than the point spacing.

## 3. Ground removal

### Which method

| Situation | Method | Why |
|---|---|---|
| Default, indoor, ramps up to ~30%, boxes/pallets on the floor | `pmf` | Best object rejection: 0% of box tops flagged as ground on every sample. Handles objects larger than a DEM cell (a 3x3 opening alone cannot). |
| Perfectly flat or gently sloped floor, want the fastest thing | `local_grid` | Same DEM, one opening; on the 15% slope it is within 1% of `pmf`. Two parameters only. |
| Steep ramps, hilly outdoor terrain, few objects on the ground | `csf` | Follows slopes best: 100% of the 35% ramp with defaults, where the DEM methods lose 8-10%. But it drapes over low wide objects: 17% of a box top on `sample_ramp`, 58% on `sample_slope`. |
| Objects on the floor keep being flagged as ground | not `csf` | Switch to `pmf`; `class_threshold` only partially fixes it (17% -> 4%). |

### Steep slopes with `pmf` / `local_grid`: what actually helps

The DEM stores the *minimum* Z per cell, so on a slope the estimated floor
sits below the true floor by up to `slope x cell_size`. Points near the top
edge of each cell then exceed the ground threshold and are kept as objects.
Measured on `sample_ramp` (35% ramp, `pmf`):

| Change | Ramp points detected as ground | Box tops flagged as ground |
|---|---|---|
| defaults | 92% | 0% |
| `slope` 0.3 -> 0.5 or 1.0 | 92% (no effect) | 0% |
| `cell_size` 0.5 -> 0.25 | 100% | 0% |
| `initial_distance` 0.10 -> 0.20 | 100% | 0% |
| `initial_distance` 0.10 -> 0.35 | 100% | 100% (boxes lost) |

So:

- Ramp/slope showing up as `occupied` -> **halve `cell_size`** first. This
  shrinks the within-cell height offset without loosening the object
  threshold. Only go smaller than the point spacing x ~5 or the DEM fills
  with holes.
- Still losing slope points -> raise `initial_distance` (`pmf`) or
  `thickness` (`local_grid`), but keep it **below the height of the lowest
  obstacle you must keep**. 0.2 m keeps a 0.3 m box; 0.35 m does not.
- `slope` in `pmf` is *not* the slope of your floor. It scales how much a
  cell may rise between opening windows before it is called an object; it
  only matters for large objects (bigger than a cell) and for the final
  ground threshold it does nothing. Leave it unless big flat-topped objects
  (tables, vehicles) are being absorbed into the ground -- then *lower* it.
- Large flat objects (a table top, a vehicle roof) become ground -> raise
  `max_window` so the opening is wider than the object, or lower `slope` /
  `max_distance`.
- Floor detail (steps, kerbs) being smoothed away -> lower `max_window`.

### `csf`

- `rigidness` 1/2/3 made no measurable difference on the samples; it only
  matters on real hilly terrain. 1 = cloth follows steep ground, 3 = stiff.
- Low objects absorbed into the ground -> lower `class_threshold` (default
  0.10). A finer `cloth_resolution` makes it *worse* (17% -> 42% of a box
  top) because the cloth can drape over the object.
- `slope_smooth` on (default) is what makes ramps work; turn it off only for
  flat floors with sharp steps.
- Results are reproducible: `csf` is loaded with its OpenMP threads pinned to
  one, because with them free the order they accumulate in decides which
  points fall on either side of `class_threshold` and the same cloud gives a
  different answer every run (a 90-point spread on `sample_ramp`). Pinning
  costs nothing -- `csf` gets no speedup from threads. If your own code
  imports `CSF` before this project does, the pin does not take and that
  reproducibility is lost.

## 4. Height filter and map cleanup

- Set the Min height just above the floor thickness (~0.05-0.10 m) even
  with ground removal on; it removes the residual floor points that any
  method leaves at slope breaklines.
- Set Max height to the robot's height. Overhanging structure above it
  (lamps, ducts) is not an obstacle for navigation.
- Map cleanup `min occupied blob` is the last resort for specks. It erases
  real obstacles of that size too, so keep it at 2-4 cells and prefer
  noise removal `cluster` when the specks come from point clumps.
- Map bounds are computed from *all* non-noise points regardless of height,
  so a single far-away point that survives noise removal enlarges the map;
  check the `X/Y` bounds shown under the file name after loading.

## 5. When the 3D view feels slow

The view is capped at "Max display points" (default 1,000,000) and thins
larger clouds by voxel subsampling. If interaction still drags:

- Lower "Max display points" (e.g. 300,000). Drawing cost scales with it
  almost linearly, and the map is unaffected -- it is always computed from
  every point.
- If the *map preview* is what lags behind, that is the occupancy grid being
  recomputed over the full cloud. Coarsen the resolution while tuning, then
  set the final value before exporting. The preview runs off the UI thread,
  so it lags rather than blocks.
- Noise and ground removal run over the full cloud too, and neither is
  affected by the display limit. Neither blocks the window either -- ground
  removal on a thread, noise removal in its own process -- so a slow method
  costs you the wait, not the interaction. See `voxel_count` in section 2 for
  the cheapest noise method on huge clouds: 0.2 s where `cluster` takes 3 s on
  1.5 million points.

## 6. Quick recipes

| Scene | Noise | Ground | Height |
|---|---|---|---|
| Flat indoor room, clean scan | off, or `cluster` defaults | off | min 0.10, max = robot height |
| Indoor with ramps / uneven concrete | `cluster` | `pmf`, `cell_size` 0.25 | min 0.10 |
| Steep ramp (>30%) with few objects | `cluster` | `csf` | min 0.10 |
| Outdoor, hilly, sparse far points | `statistical` | `csf`, or `pmf` with `max_window` 65 | min 0.15 |
| Huge cloud, need it now | `voxel_count`, `voxel_size` 0.2 | `local_grid` | -- |

Verify any recipe with the compare scripts and look at the `_<method>.ply`
outputs (noise = red, ground = gray) before trusting the map:

```bash
uv run python -m pointcloud_map_gui.tools.compare_ground_removal sample_data/sample_ramp.pcd --out-dir out/ \
    --min-height 0.1 --max-height 1.5 --param pmf.cell_size=0.25
```
