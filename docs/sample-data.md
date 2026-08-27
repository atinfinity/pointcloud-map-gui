# Sample data

The clouds under `sample_data/` are generated, not recorded. Each one exists
to exercise a particular thing, and the generator reproduces them byte for
byte on any machine, so a measurement quoted against one of them can be
repeated.

Coordinates are rounded to float32 before they are written. PLY stores
coordinates as double and PCD as float32, and at full float64 precision the
two disagreed about the same cloud: `rng.normal` reaches the platform's libm
through the ziggurat's exp/log, so macOS arm64 and Linux x86_64 differed by a
few ULP -- around 1e-16 m, invisible in the `.pcd` because float32 rounds it
away, but baked into the `.ply`. Rounding makes both formats carry the same
numbers, and the `.pcd` and `.ply` of one cloud now load as identical arrays.

```bash
uv run python -m pointcloud_map_gui.tools.generate_sample
```

Generates five synthetic point clouds in `sample_data/`. The first four are
small scenes for exercising one feature at a time, written as both `.pcd` and
`.ply`; the last is large and written as `.pcd` only, without colors (the GUI
colors points by height and ignores any the file carries).

- `sample_room`: a flat floor, walls, and a floating obstacle (height
  1.0-1.8m), useful for exercising the height filter.
- `sample_slope`: a floor rising 15% along X with noise and gentle waves, a
  0.3m-tall box standing on it, walls and a floating obstacle. The box top is
  lower than the far end of the floor, so only ground removal can keep the
  box while dropping the floor.
- `sample_slope_noisy`: `sample_slope` plus 500 uniformly scattered noise
  points, 30 tiny clusters of 2-5 points and 20 speck clumps of 10-40
  points, for exercising noise removal.
- `sample_ramp`: flat and sloped floor mixed -- a flat lower level (z=0), a
  35% ramp rising 0.7m, and a flat upper level (z=0.7) -- with a 0.3m box on
  each level and one on the ramp, plus a floating obstacle. The lower box's
  top is below the upper floor, and the flat/slope breaklines are where
  DEM-based ground removal is most likely to misclassify.
- `sample_haze`: a 6 x 6 m room with an operator smeared through it, as a SLAM
  map records someone who walked with the robot -- see
  [koide3/glim#240](https://github.com/koide3/glim/issues/240). The haze is
  coloured red and sits at the end of the cloud, so a comparison can score
  against it. Its density is about a quarter of the structure's -- 5 neighbours
  within 10 cm against 21 -- which is where a real export was reported to sit,
  and where the methods start to disagree: `radius` finds three quarters of it,
  `scatter` all but a tenth. Thinner and any of them would do; denser and only
  `scatter` would.
- `sample_large_site`: a 50 x 50 m outdoor yard at 1,554,944 points -- survey
  scale, sparser than an indoor scan but spread over 25x the area. A perimeter
  fence, three buildings, three shipping containers and gently undulating
  ground (+/-0.05m, enough to give ground removal something to do). Everything
  in it is there so the map can be checked by eye: two buildings were entered
  through their doorways and come out hollow and `free`; the third was only
  seen from outside and comes out `unknown` inside its `occupied` walls; two
  patches of ground sit in a building's occlusion shadow, and nothing sees the
  ground under a container, so those come out `unknown` too. Its ground is
  sampled at 0.045 m, just under the 0.05 m/cell default resolution -- coarser
  than the grid and the map fills with `unknown` speckle.

`python -m pointcloud_map_gui.tools.benchmark_display --generate --points N` builds a sixth, a warehouse floor
plan of any size, without writing it to disk; `--write PATH` saves one. It is
what the five-million-point figures above come from, and is not committed
because at that size the file is 60 MB.
