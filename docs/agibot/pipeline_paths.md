# Teleop, record and cuMotion: what is shared and what differs

Measured on `agibot_stack_bowls` after the Isaac Lab 3.0 GA sync (2026-09-05), by building the
environment exactly the way each runner does and dumping what reaches the simulator, then
launching each runner for real and sampling the GPU. Re-run with
`scratchpad/dump_stack_bowls_cfg.py --mode teleop|record|cumotion --device ...` (kept in the
session scratchpad; the table below is the durable record).

## Identical across all three paths

| item | value |
| --- | --- |
| physics | dt 1/120 s, decimation 8, control 15 Hz, render interval 2, Fabric on, TGS solver, same iteration counts |
| robot | same actuator config everywhere: arms at the shipped `joint_effort_limit` 1000-2000 (task-level 300 removed 2026-09-06), stock stiffness/damping; grippers 100 N m / 10 rad/s both sides; same rest pose (lift 0.1995, pitch 0.6025) |
| reset events | `bowl0..2`, `jitter_bowls`, `reset_robot_to_default_pose`, `robot_reset_pose`, plus upstream's new `reset_background_physics` (a no-op on our table and room) |
| observations | `policy`: actions, joint_pos, joint_vel, eef_pos, eef_quat, left/right_gripper_pos |
| recorded state | `states/` and `initial_state/` for the robot and every object, `obs/*` as above, one row per control step |
| sensors | `ee_frame` only (cameras off; images are re-rendered offline for cuMotion data) |

## What differs

| item | teleop (`teleop_se3_agent`) | record (`record_demos`) | cuMotion (`*_cumotion.py --record-dir`) |
| --- | --- | --- | --- |
| arm action term | `SurfaceGuardedRMPFlowAction` x2 (relative EE delta, dim 6, target hold + surface guard) | same | `SmoothJointPositionAction` x2 (absolute joint targets, dim 7, first-order hold) |
| gripper action term | `RampedBinaryJointPositionAction` (dim 1, 1.67 s ramp) | same | `JointPositionAction` (dim 3, constant target; the executor ramps it over 1.67 s) |
| action vector | 14 (`[L pose 6, L grip 1, R pose 6, R grip 1]`) | 14 | 20 (`[L arm 7, L grip 3, R arm 7, R grip 3]`) |
| who drives | human keys -> device -> `env.step` | same | planner -> `EnvActionExecutor` -> `env.step` (or `ArmExecutor` direct joint writes when not recording) |
| terminations | `success` only (time_out removed) -> success auto-resets the scene | none (success held aside and judged by the script; time_out removed) | `time_out` at 600 s (success judged by the script) |
| recorder | none | `AgibotDemoRecorderManagerCfg` (Arena's + `joint_pos_target`), registered by the env, export succeeded only | same recorder, camera term off (images re-rendered offline) |
| recorded actions | -- | `actions (T,14)` = EE deltas + binary gripper; `processed_actions (T,18)`; **`joint_pos_target (T,20)`** | `actions (T,20)` = absolute joint targets; `processed_actions (T,20)`; **`joint_pos_target (T,20)`** (== actions) |
| pacing | as fast as the Kit loop renders | `RateLimiter` at `--step_hz` (we pass 15) | as fast as the sim steps (headless) |
| device | `cuda:0` (was `cpu` until 2026-09-05, inherited from upstream's XR docs) | `cuda:0` | `cuda:0` |
| viewer | Kit GUI via noVNC, head-view viewport | same | headless |

## Why the two recordings could not be mixed, and how they are unified

The raw `actions` of the two paths are different quantities. A teleop step's 14 numbers are
*how far the operator asked each end-effector to move this control step* (dx, dy, dz in metres and
an axis-angle rotation in radians, scaled by the sensitivity) plus a binary open/close per hand.
Nothing in them says where the joints went: RMPFlow, inside the simulator, turns the delta into a
target pose and solves joint position targets for the 7 arm joints at every physics substep, and
the binary gripper is expanded into a ramped target for the 3 gripper joints. A cuMotion step's 20
numbers *are* those joint position targets (7 + 3 per arm, radians, absolute). A policy trained on
one cannot emit the other, so the two datasets were not mixable as recorded.

What both paths share underneath is the joint position target the drives were actually given at
the end of each control step. Since 2026-09-05 every Agibot recording writes it as
`joint_pos_target (T, 20)` in `AgibotDualArmJointActionsCfg` order
(`isaaclab_arena/embodiments/agibot/demo_recorders.py`, installed by `install_agibot_control_stack`
as the env's demo recorder, and used by the cuMotion drivers). Measured on `agibot_stack_bowls`:

| check | result |
| --- | --- |
| cuMotion demo: `joint_pos_target` vs `actions` | identical, max difference 0.000000 rad on arms and grippers |
| teleop-path demo: arm `joint_pos_target[t]` vs `states/joint_position[t+1]` | 95th percentile 1.8 deg; 7 of 129 steps above 5 deg, all at the first steps after reset (RMPFlow's first-step jump) and at gripper contact |
| teleop-path demo: gripper column while the raw command is a bare -1 | the recorded target ramps 0.994 -> 0 over 1.67 s, i.e. exactly what the fingers were driven with |
| merge of one teleop-path and one cuMotion demo | `merge_demos.py --drop_mismatched` keeps `joint_pos_target`, `obs/*`, `states/*`, `initial_state/*` and drops the raw `actions`, `obs/actions`, `processed_actions` (recorded in the file's `dropped_keys` attribute) |

The LeRobot configs (`isaaclab_arena_gr00t/lerobot/config/agibot_*`) now read
`action_name_sim: joint_pos_target`, so a mixed file converts to one dataset with a 20-dim
joint-space action for every frame; `observation.state` (34 joints) is unchanged. The raw
per-path files keep their own `actions` for replay and debugging. End-to-end result (2026-09-05): the merged file re-rendered its three camera streams from the
recorded states (`rerender_demo_cameras.py`, both demos) and converted into **one LeRobot dataset
with 2 episodes, 1148 frames, a 20-dim joint-space action on every frame, 15 fps and 6 videos**
-- one episode driven through the teleop path, one by cuMotion, indistinguishable downstream.

Standing procedure for a mixed dataset: record human demos with `record_demos.py` (the env's
recorder writes `joint_pos_target`), collect cuMotion demos with the driver, `merge_demos.py
--drop_mismatched` the raw files, `rerender_demo_cameras.py` on the merged file, then
`convert_hdf5_to_lerobot.py`. Recordings made before 2026-09-05 have no `joint_pos_target`:
cuMotion ones can use `processed_actions` (identical values); teleop ones cannot be converted to
the joint-space label after the fact and would have to be re-recorded.

## GPU or CPU?

Not everything ran on the GPU before this check: teleop and recording used CPU physics because the
upstream recording docs pass `--device cpu` (for the XR path) and we copied it; cuMotion, the
probes and Phase 1 run on `cuda:0`. Measured on this box (L40S, one Isaac Sim process):

| launch | physics device | GPU util (mean over 40-90 s) | GPU memory | process CPU |
| --- | --- | --- | --- | --- |
| teleop, Kit GUI on Xvfb | cpu | 20 % (rendering only) | 3.6 GB | 319 % |
| teleop, Kit GUI on Xvfb | cuda:0 | 37 % | 5.6 GB | 271 % |
| record, Kit GUI, 15 Hz | cpu | 17 % | 3.3 GB | 327 % |
| cuMotion recording, headless | cuda:0 | 16 % | 5.2 GB | 483 % (planner) |

GPU physics works for teleop on GA (device instantiates, loop runs, no rendering warnings beyond
the ones the CPU run also prints). The mid-air pinch probe gives **identical numbers** on CPU and
GPU physics for the billet (close peak 0.56 m/s, swing 0.59 m/s, pad gap 45.3 mm, HELD/HELD), so
the device is not a hidden source of behavioural difference for a pinch. Two facts still argue for
knowing which device you are on: the CPU pipeline reports errors the GPU pipeline swallows
(velocity writes to kinematic bodies, zero-joint articulations), and GPU memory per process is
2 GB higher, which matters against the two-process ceiling of this box.

Decision (user, 2026-09-05): **everything runs on `cuda:0`** -- teleop, recording, cuMotion, probes,
every new task. `--device cpu` is kept only as the debugging mode that surfaces silent errors. The
commands in `ops.md` and the playbook say `cuda:0`.

## Things checked while measuring

- The teleop path keeps `success` as a termination, so solving the task resets the scene under the
  operator; `record_demos` removes it and judges success itself (needs `--num_success_steps`).
- Arena only registers its own demo recorder (`demo_recorder_config`) when the embodiment has
  cameras enabled; a camera-less `record_demos` run falls back to Isaac Lab's recorder. Both write
  the same `obs/`, `states/`, `actions` layout, so downstream tools do not notice.
- `tidy_workbench_teleop/teleop_demos.hdf5` (96 bytes) never received an exported demo: the file
  was created 2026-09-04 07:15 and untouched until the session was stopped 25 h later.
