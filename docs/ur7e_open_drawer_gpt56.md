# UR7e image-mesh drawer comparison

`ur7e_open_drawer_gpt56` uses
`/home/ubuntu/playground/rr_ur/miniworkflow_gptsol_drawer.usd`
through the existing `miniworkflow_gptsol_drawer_arena.usda` overlay beside it. The overlay
disables the asset's PhysicsScene and anchors its Body articulation root.

The comparison factory inherits the original workcell builder and changes only
the drawer asset. Both environments use x = [0.05, 0.18] m,
y = [0, 0.15] m, table-top z, and yaw = -90 +/- 10 degrees. The robot,
calibrated 640x480 D435, scene camera, and reset distribution are shared.
The driver defaults to 0.03 rad initial robot joint jitter and a 0.10 m pull.
Use matching seeds and driver settings for comparisons; matching distributions
alone do not establish that every sampled episode is paired.

The replacement has a `DrawerSlide` joint with -0.155..0 m travel and a knob
on the `Drawer` body at local (0, -0.1001, 0.0405) m. The shared success rule
is greater than 50% travel: 77.5 mm for this asset versus 75 mm for the original.
The knob is 7.5 mm higher than on the original; the driver reads each asset's
knob frame and joint sign without changing the grasp policy.

Run the automatic pipeline (200 successful openings by default):

```bash
TARGET_SUCCESS=200 bash /home/ubuntu/playground/datasets/collect_ur7e_open_drawer_gpt56_v2.sh
```

The repository entry point is
`isaaclab_arena_cumotion/scripts/collect_ur7e_open_drawer_gpt56.sh`.
Two cuMotion workers run batches of eight attempts until exactly 200 successes
can be merged. `--stop-after-pull` ends and judges the episode after the 100 mm
pull settles, while the gripper remains commanded closed. Release, retreat and
return home are neither executed nor recorded. Failed attempts are discarded.
The original driver's default remains the full demonstration when this flag is absent.

There is no need to render/convert a v1 dataset first. The pipeline records
states once, then uses two workers to re-render them with the original v2 visual
randomizer (lights, HDRI dome, table/floor materials and up to six non-colliding
distractors). Drawer position/joint jitter remain part of the physical recording;
visual randomization is deterministic per demo and does not change those states.

Outputs are separated from the original drawer and earlier qualification data:

- Raw HDF5, provenance, per-worker logs, randomization records and checkpoints:
  `/home/ubuntu/playground/datasets/rr_sim2real_raw/open_drawer_gpt56_v2/`.
- LeRobot v2.1 with 7D joints and UR-base 9D EEF:
  `/home/ubuntu/playground/datasets/rr_sim2real/miniworkflow_gptsol_open_drawer/`.
- Diffusion-policy replay buffer:
  `/home/ubuntu/playground/datasets/rr_sim2real/miniworkflow_gptsol_open_drawer_dp.zarr`.
- Preview:
  `/home/ubuntu/playground/datasets/rr_sim2real/miniworkflow_gptsol_open_drawer_preview.png`.

D435 videos stay at 640x480 / 15 fps. The separate diffusion-policy zarr uses
320x240 images and row-convention rot6d / 10D actions, matching the original v2
processing script. This resize does not change LeRobot videos or camera calibration.

Relaunch with the same settings to resume; completed worker files and camera
clips are reused. Settings are checked against `pipeline.json`, and a file lock
prevents duplicate orchestrators. Incomplete conversions are retained under a
timestamped name. `MAX_ROUNDS=40` bounds retries; reaching it before the target
fails rather than publishing a partial dataset. `SEED_BASE=0` and
`RANDOMIZE_SEED=0` are the default seeds; workers get distinct consecutive seeds.
The `--raw` and `--final` arguments select separate outputs for smoke checks.
Keep the GPU free of training during rendering.

The stop-after-pull smoke check (seeds 0 and 1, two attempts each) exported 4/4
successes, all ending with the gripper commanded closed and the drawer beyond
77.5 mm. The complete randomized pipeline produced four LeRobot episodes,
1000 training rows, 640x480 / 15 fps videos, a validated 10D-action DP zarr,
and a contact sheet under the corresponding `open_drawer_gpt56_v2_smoke` paths.
The final zarr check registers diffusion_policy's vendored JPEG2000 codec before
opening compressed image arrays, as the training loader does.

For an independent physics probe or camera preview:

```bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH=/home/ubuntu/code/IsaacLab-Arena-tasks
.venv/bin/python isaaclab_arena_cumotion/scripts/ur7e_open_drawer_cumotion.py \
  --env ur7e_open_drawer_gpt56 --headless --device cuda:0 \
  --probe-only --num-demos 3 --seed 0
.venv/bin/python isaaclab_arena_cumotion/scripts/ur7e_open_drawer_cumotion.py \
  --env ur7e_open_drawer_gpt56 --headless --device cuda:0 --seed 0 --stop-after-pull \
  --video /home/ubuntu/playground/rr_ur/sim_renders/open_drawer_gpt56.mp4
```

The probe deliberately returns the drawer to closed, so its printed task success
predicate is false; inspect the 50 mm hold and 0 mm reset measurements instead.

## Earlier qualification with release and return on 2026-09-10

Seed 0, five randomized attempts, default driver settings, states-only recording:

| Attempt | After pull (mm) | After release and return (mm) | Exported |
| --- | ---: | ---: | --- |
| 1 | 99.7 | 91.6 | Yes |
| 2 | 99.9 | 53.6 | No |
| 3 | 99.8 | 38.9 | No |
| 4 | 99.7 | 78.1 | Yes |
| 5 | 100.1 | 58.0 | No |

All five planned and pulled successfully, but only 2/5 passed the final task
predicate. The loss of opening happens after the pull, during release/return;
these measurements do not identify the precise physical cause. This small batch
does not establish a reliable collection success rate. No grasp or material
parameters were tuned to improve the comparison result.

The two successful recordings contain 389 and 380 steps (769 total), with
7-dimensional actions and `drawer_rr_gpt56` articulation states, under
`/home/ubuntu/playground/datasets/rr_sim2real_raw/open_drawer_gpt56/qualification_seed0/`.

Both episodes were re-rendered through D435 and scene_cam and converted to
`qualification_seed0/open_drawer_gpt56/lerobot/`. The existing converter aligns
next-step actions by dropping one row per episode, yielding 767 training rows.
Validation confirmed finite 7D joint states/actions, finite 9D end-effector fields,
and 640x480 H.264 D435 video at 15 fps. URDF FK versus recorded TCP maximum errors
were 2.82 and 2.87 mm (below the existing 3 mm gate).
