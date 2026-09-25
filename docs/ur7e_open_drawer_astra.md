# miniworkflow Astra open-drawer qualification

Source-host qualification on 2026-09-11. This is the fourth drawer baseline;
USDcraft, Articraft and miniworkflow GPTSOL already have collected datasets.

## Asset and minimum compatibility change

- Source: `/home/ubuntu/playground/rr_ur/miniworkflow_astra_drawer.usd`.
- SHA-256: `f28dee0a983af5cc3bdd8c4a650de59e93e172391ba8f73ee8266ea7c52c389e`.
- Unscaled bounds: 0.199 x 0.229 x 0.078 m, including knob.
- Joint `DrawerSlide`: Y axis, limits [-0.150, 0] m; success above 75 mm opening.
- Knob belongs to `Drawer`, local center (0, -0.103, 0.043) m.
- Environment: `ur7e_miniworkflow_astra_open_drawer`.
- HDF5 articulation key: `miniworkflow_astra_drawer`.

The raw USD fails Isaac Lab fixed-root spawning: `PhysicsArticulationRootAPI`
is on plain `/DrawerBox`, which lacks `RigidBodyAPI`. The minimal
`miniworkflow_astra_drawer_arena.usda` overlay moves that API to `/DrawerBox/Housing`.
No geometry, scale, material, collision, joint-limit, friction or damping edits.
The overlay template is committed under `tools/rr_sim2real/asset_overlays/`;
copy it next to the original payload. `ARENA_DRAWER_ASTRA_USD` overrides its path.

With the overlay, the joint probe holds exactly 50 mm for 30 physics steps and
returns to 0 mm. Its final task predicate is intentionally false because the
probe ends closed. Logs: `/tmp/astra_drawer_raw_probe_retry.log` (raw failure),
`/tmp/astra_drawer_arena_probe.log` (successful motion probe).

## Harness fixes discovered during qualification

Method aliases require distinct config dataclasses: reusing legacy config types
caused a registry assertion before any asset spawned. Legacy environment names
remain registered for old HDF5 files.

The shared openable reset sets joint position but leaves joint velocity intact.
In the first three-attempt run, attempt 2 inherited -0.01890831 m/s from attempt 1.
It slid open before grasping (126.4 mm knob/TCP separation); the task's final
opening predicate incorrectly made this look like a successful grasp demo.
Attempts 1 and 3 genuinely pulled 98.9 and 98.8 mm. Do not use this diagnostic
HDF5 for training: `rr_sim2real_raw/miniworkflow_astra_open_drawer/qualification_seed0/`.

The cuMotion driver now explicitly zeros slide velocity after every reset,
before stepping/recording. This applies equally to all drawer methods and does
not modify their passive physical properties or success threshold. A final
opening predicate alone remains insufficient to audit grasp quality.

## Corrected result

Three of three seed-0 randomized attempts completed real knob grasps and pulls.
Final openings were 98.878, 98.955 and 98.874 mm. Recordings contain 248, 256
and 248 frames (752 total). Initial slide speeds were below 6e-9 m/s, initial
openings below 1e-9 m, and recorded fixed-body translation drift was zero.
Gripper/knob proximity and approximately 98 mm knob displacement during the pull
were checked in the log, in addition to the final task predicate. This is a
three-attempt qualification, not an estimate of large-scale collection yield.

Outputs under `/home/ubuntu/playground`:

- D435: `rr_ur/sim_renders/miniworkflow_astra_drawer_cumotion_reset_fixed.mp4`.
- Scene: `rr_ur/sim_renders/miniworkflow_astra_drawer_cumotion_reset_fixed_scene.mp4`.
- HDF5 and `cumotion.log`:
  `datasets/rr_sim2real_raw/miniworkflow_astra_open_drawer/qualification_seed0_reset_fixed/`.

D435 metadata verified at 640x480, 15 fps, 752 frames. Preview confirms grasped
opening. The original qualification clip still has white fingertips. A subsequent
visual-only fix binds the inner-finger meshes/face subsets directly and was
verified by replaying demo 0; see the black-gripper section of the handoff README
for the corrected preview/video paths. The original Astra payload hash
is unchanged. Modified Python files pass pre-commit checks.

## Reproduce the corrected run

From this worktree, use the native runtime exports in
[`rr_sim2real/README.md`](rr_sim2real/README.md). Use fresh output paths per run:

```bash
.venv/bin/python isaaclab_arena_cumotion/scripts/ur7e_open_drawer_cumotion.py \
  --env ur7e_miniworkflow_astra_open_drawer --device cuda:0 \
  --stop-after-pull --pull-distance 0.10 --num-demos 3 --seed 0 --fps 15 \
  --video /home/ubuntu/playground/rr_ur/sim_renders/miniworkflow_astra_drawer_cumotion_reset_fixed.mp4 \
  --record-dir /home/ubuntu/playground/datasets/rr_sim2real_raw/miniworkflow_astra_open_drawer/qualification_seed0_reset_fixed \
  --dataset-name miniworkflow_astra_open_drawer
```

Uses the unchanged shared placement range, 0.03 rad robot start jitter, and
default black-gripper material override. D435 video is 640x480 at 15 fps;
the second clip has `_scene.mp4` suffix. Stop occurs after pulling, before
release/retreat/home. Background/distractor randomization is a later replay
stage, not enabled for these motion-qualification clips. No 200-demo collection,
LeRobot conversion or DP training dataset has been produced for Astra yet.
