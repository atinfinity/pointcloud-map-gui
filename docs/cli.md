# Command-line tools

Four tools ship alongside the GUI. They are package modules, so they run with
`python -m` and need no path setup:

```bash
uv run python -m pointcloud_map_gui.tools.<name> [options]
```

Every one of them accepts `--help`. This page covers what they are for and how
the options fit together; `--help` is the authority on spelling and defaults.

| Tool | Needs a display | What it is for |
|---|---|---|
| [`compare_ground_removal`](#compare_ground_removal) | no | Run every ground-removal method on one cloud and compare them |
| [`compare_noise_removal`](#compare_noise_removal) | no | The same, for noise removal |
| [`generate_sample`](#generate_sample) | no | Rebuild the bundled sample clouds |
| [`benchmark_display`](#benchmark_display) | **yes** | Time what a height-filter drag costs in the 3D view |

---

## compare_ground_removal

Runs each method over the same cloud, prints how much each called ground and
how long it took, and writes a coloured `.ply` plus the occupancy grid each one
produces. Use it to answer "which method, and which parameter" without clicking
through the GUI.

```bash
uv run python -m pointcloud_map_gui.tools.compare_ground_removal INPUT [options]
```

| Option | Default | Meaning |
|---|---|---|
| `INPUT` | required | Point cloud to run every method on (`.pcd` / `.ply`) |
| `--out-dir DIR` | `ground_removal_out` | Where the `.ply` and map files go |
| `--methods M [M ...]` | all | Any of `pmf`, `local_grid`, `csf` |
| `--param METHOD.NAME=VALUE` | — | Override one parameter. Repeatable |
| `--min-height M` | the cloud's minimum | Bottom of the height filter, in metres |
| `--max-height M` | the cloud's maximum | Top of the height filter, in metres |
| `--resolution M` | `0.05` | Occupancy grid resolution, metres per cell |
| `--min-blob-cells N` | `0` | Map cleanup: erase occupied blobs smaller than this. 0 disables it |

```bash
uv run python -m pointcloud_map_gui.tools.compare_ground_removal \
    sample_data/sample_ramp.pcd --out-dir out/ \
    --min-height 0.1 --max-height 1.5 --param pmf.cell_size=0.25
```

```
sample_ramp: 113,981 points, height filter [0.100, 1.500], resolution 0.05
baseline (no removal): 9,908 occupied cells

method      params                                      ground   ratio  time[s]  occupied
pmf         cell_size=0.25, max_window=33, slope=0.3…    43,227   0.379     0.01     1,382
local_grid  cell_size=0.5, thickness=0.1                 41,265   0.362     0.01     2,481
csf         cloth_resolution=0.5, rigidness=2…           44,004   0.386     0.09     1,504

pairwise IoU of ground masks:
  pmf vs local_grid: 0.955
  pmf vs csf: 0.966
  local_grid vs csf: 0.922
```

Per method it writes `<name>_<method>.ply` (ground grey, everything else in the
height colormap) and `<name>_<method>.pgm` / `.yaml`, plus `<name>_none.*` as
the no-removal baseline. Open the `.ply` files before trusting a number: the
IoU tells you two methods disagree, not which one is right.

Parameter names for `--param` are whatever the method takes -- the printed
`params` column lists them all.

## compare_noise_removal

The same idea for noise removal, with one addition: `--ground-method` also runs
ground removal, so the maps reflect the full pipeline rather than noise removal
alone.

```bash
uv run python -m pointcloud_map_gui.tools.compare_noise_removal INPUT [options]
```

| Option | Default | Meaning |
|---|---|---|
| `INPUT` | required | Point cloud to run every method on (`.pcd` / `.ply`) |
| `--out-dir DIR` | `noise_removal_out` | Where the `.ply` and map files go |
| `--methods M [M ...]` | all | Any of `cluster`, `radius`, `statistical`, `voxel_count` |
| `--param METHOD.NAME=VALUE` | — | Override one parameter. Repeatable |
| `--ground-method M` | off | Also remove ground with `pmf`, `local_grid` or `csf` |
| `--min-height M` | the cloud's minimum | Bottom of the height filter, in metres |
| `--max-height M` | the cloud's maximum | Top of the height filter, in metres |
| `--resolution M` | `0.05` | Occupancy grid resolution, metres per cell |
| `--min-blob-cells N` | `0` | Map cleanup: erase occupied blobs smaller than this |

```bash
uv run python -m pointcloud_map_gui.tools.compare_noise_removal \
    sample_data/sample_slope_noisy.pcd --out-dir out/ \
    --methods cluster voxel_count --ground-method pmf \
    --min-height 0.1 --max-height 1.5
```

```
sample_slope_noisy: 111,729 points, resolution 0.05, ground removal: pmf
baseline (no removal): map 121x121, 12,401 occupied cells

method       params                                       noise   ratio  time[s]   map size  occupied
cluster      eps=0.1, min_points=4, min_cluster_size=50   1,032  0.0092     0.19    121x121       805
voxel_count  voxel_size=0.1, min_points=4                 3,494  0.0313     0.01    121x121       744

pairwise IoU of noise masks:
  cluster vs voxel_count: 0.146
```

A low IoU like that is the point of the tool: these two methods remove almost
disjoint sets of points, so "how many" says very little on its own.

## generate_sample

Rebuilds the clouds under `sample_data/`. They are committed, so this is only
needed after changing the generator -- or to write a copy somewhere else.

```bash
uv run python -m pointcloud_map_gui.tools.generate_sample [--out-dir DIR]
```

| Option | Default | Meaning |
|---|---|---|
| `--out-dir DIR` | the repository's `sample_data/` | Where to write the clouds |

Output is deterministic: regenerating in place leaves the `.pcd` files
byte-identical. See [sample-data.md](sample-data.md) for what each cloud is.

## benchmark_display

Times what one height-filter change costs in the 3D view, per display budget.
**This one opens the real window** -- the timings come from the live renderer,
so it needs a display.

```bash
uv run python -m pointcloud_map_gui.tools.benchmark_display [INPUT] [options]
```

| Option | Default | Meaning |
|---|---|---|
| `INPUT` | `sample_data/sample_large_site.pcd` | Cloud to measure |
| `--generate` | off | Build a warehouse scene in memory instead of reading a file |
| `--points N` | `5000000` | Size of the generated cloud |
| `--write PATH` | — | Write the generated cloud and exit, without opening a window |
| `--budgets N [N ...]` | `1000000 2000000 0` | "Max display points" values to measure. `0` means no thinning |
| `--ticks N` | `10` | Updates to time per budget; the fastest is reported |

```bash
uv run python -m pointcloud_map_gui.tools.benchmark_display --budgets 500000 1000000 0
```

```
sample_large_site.pcd (1,554,944 points)
height-filter update: _update_display_buffers

load + first upload : 0.89 s

            budget       drawn       rebuild              tick
           500,000     474,242      507.6 ms      5.4 ms (184.7 fps)
         1,000,000     491,181      535.2 ms      5.3 ms (188.8 fps)
          no limit   1,554,944      278.7 ms     15.5 ms (64.3 fps)
```

`rebuild` is the cost of re-picking the drawn subset, which happens only when
the point set changes; `tick` is what every slider movement costs. The fastest
of `--ticks` runs is reported, because the slow ones are the process losing the
CPU rather than the code getting slower.

`--generate --points 5000000` reproduces the five-million-point figures quoted
in [display.md](display.md); that cloud is 60 MB as a file, which is why it is
built on demand rather than committed. `--write` saves one if you want it.

To compare against an older version of the drawing code, check out the earlier
commit and run the tool there. It looks the height-filter update up by name and
prints which one it found, so it still works across the versions where that
method was called something else.
