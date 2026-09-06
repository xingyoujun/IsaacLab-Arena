# dev.md -- what this fork changes in Isaac Lab-Arena, and what an agent must do on top of upstream

This is the one document to read before working on the Agibot benchmark in this repository.
It answers two questions:

1. **Where does this repo differ from `origin/main`** (NVIDIA's Isaac Lab-Arena), and why.
2. **What must an agent do when it brings the Agibot -- or any new task -- into Arena**, i.e. the
   improvements upstream does not have and we must apply every time.

The detailed process documents live in `docs/agibot/` (start at `docs/agibot/README.md`); this file
is the map. Last updated 2026-09-06, after the Isaac Lab 3.0 GA sync (`b93d614d2`), the reset
transient fix (`01eb41fdd`) and the removal of the task-level arm effort limit.

---

## Part 1 -- Improvements upstream does not have (apply these; do not re-derive)

Everything below was found by measurement and cost days each. Upstream Arena ships the Agibot A2D
as a demo robot for two place tasks; it was never driven by a human, never recorded at scale, and
never run with a planner. Bringing it into a benchmark surfaced the following, in the order an
agent will hit them.

### 1.1 The robot: the rest pose is wrong on two counts

| defect in upstream | symptom | our fix | where |
| --- | --- | --- | --- |
| Gripper four-bar linkage authored inconsistently: drivers open (`*_hand_joint1`, `*_Support_Joint` = 0.994) while the coupled loop joints (`*_Right_1_Joint`, `*_RevoluteJoint`) are 0. A closed linkage cannot be there. | Every `env.reset()` makes PhysX snap the loop shut in one 8 ms substep; the impulse swings the arm **15-18 cm for 0.5 s**, on every controller and every gain. Every recorded demo started with it. Franka has no loop and no transient. | `CONSISTENT_OPEN_GRIPPER_LINKAGE_JOINT_POS`: `Right_1_Joint = -0.994`, `RevoluteJoint = +0.994`. Swing 172.5 mm -> 0.2 mm. | `isaaclab_arena/embodiments/agibot/agibot.py`; root cause in `docs/agibot/control_paths_research.md` section 10 |
| Left wrist not mirrored (`left_arm_joint6/7` = +1.47 / -0.16 against the right's -0.7 / 0.0). | Hands start 160 mm out of symmetry; the left one tucked over the body, out of the head view. | `MIRRORED_LEFT_WRIST_JOINT_POS` = right wrist mirrored onto the left; the Lula yamls' `default_q` and `cspace_to_urdf_rules` patched to match (local copies under `embodiments/agibot/rmpflow/`). | same file |

Both are Isaac Lab asset issues (`isaaclab_assets/robots/agibot.py`), not Arena's. Not yet reported
upstream (user decision: hold).

### 1.2 The robot: grippers and reset event

| defect in upstream | symptom | our fix |
| --- | --- | --- |
| Left gripper ships with 10x/100x lower drive ceilings than the right (effort 10 / velocity 2 vs 100 / 10). | Left hand cannot grasp anything: 0/15 lifts. | Left gripper = verbatim copy of the right (effort 100, velocity 10). Cross-matrix measured: velocity is the only live knob. |
| No robot-only reset event; the generic one reset **all** articulation joints to 0. | Hands at y = +/-1.12 m after reset, RMPFlow unusable (Lula's fixed `joint_lift_body` / `joint_body_pitch` no longer describe the robot). | `AgibotEventCfg.reset_robot_to_default_pose` -> `reset_joint_position_and_velocity_to_defaults` (positions **and** drive targets). |
| `OffsetCfg.rot` for the left arm written wxyz in an xyzw field. | Left arm runs 270-450 mm after reset. | `_LEFT_ARM_BODY_OFFSET_ROT_XYZW = (0, -0.7071, 0, 0.7071)`. Isaac Lab 3.0 quaternions are **xyzw** everywhere in Arena; cuMotion's API is wxyz. |

The arm PD and effort limits are the stock Isaac Lab values (stiffness 1e7 / 2e4, damping 0,
effort 1000-2000). A gains x controller matrix (2026-09-05) showed soft gains lose cuMotion demos
to in-hand slip, and on 2026-09-06 the task-level 300 N m ceiling was dropped as well: every
path runs the embodiment defaults, `apply_arm_gains` is a no-op unless a measurement sets a knob. **The robot is frozen;
tasks adapt to it.** See `docs/agibot/agibot_embodiment.md` for the refuted-fixes list.

### 1.3 The standard control stack (teleop and recording)

Upstream's RMPFlow relative-delta action terms are unusable for a human on a dual-arm robot as
shipped. Every Agibot environment installs, in this order, via
`install_agibot_control_stack(env_cfg, cfg, surface_z)` in
`isaaclab_arena_environments/agibot_tabletop_common.py`:

1. `install_arm_target_hold` (`isaaclab_arena/utils/arm_target_hold.py`) -- relative RMPFlow
   rebuilds the target as current pose + delta, so a zero-commanded arm walks 88 mm in 0.7 s in
   dual-arm mode. The hold keeps the previous target.
2. `install_surface_guard` (`isaaclab_arena/utils/surface_guard.py`) -- an action term that stops
   the pads being commanded below the work surface; the gripper-table collision was **commanded**,
   not a physics bug, and damping makes it 4x worse.
3. `install_ramped_gripper` (`isaaclab_arena/utils/ramped_gripper.py`) -- rate-limits the binary
   open/close so the pinch does not fling the object (one of four measured fling causes). The ramp
   time is the embodiment's `AGIBOT_GRIPPER_RAMP_SECONDS` (0.5 s since 2026-09-06), shared with the
   cuMotion executor.
4. `apply_arm_gains` -- no-op by default since 2026-09-06 (measurement knob only).

It also sets `env_cfg.demo_recorder_config = agibot_demo_recorder_cfg(...)` (section 1.5).

### 1.4 Teleoperation device and display

- Upstream deleted Arena's own teleop/record scripts (`4a97a0cce`, `0ed4b3f2a`) with no migration
  note. Use Isaac Lab's `record_demos.py` (every operator session records; `teleop_se3_agent.py` is a driver that writes nothing) with the Arena registration
  callback and **both** device flags (`--arena_teleop_device dual_arm_keyboard --teleop_device
  dual_arm_keyboard`); giving only the Arena one silently falls back to the 7-value keyboard and
  dies with `Invalid action shape, expected: 14, received: 7`.
- `DualArmSe3Keyboard` (`isaaclab_arena/devices/dual_arm_keyboard.py`, cfg in
  `dual_arm_keyboard_cfg.py` with a string `class_type` so the cfg imports without Kit; registered in
  `assets/device_library.py` and `assets/retargeter_library.py`). Tab switches arms; emits the
  Agibot's 14-value layout directly, no retargeter.
- Head view: `AgibotEmbodiment.HEAD_VIEW_*` and `utils/cameras.get_viewer_cfg_from_robot_body`;
  the Kit viewport FOV is fixed at +/-30 deg so the standoff goes straight up.
- Display: noVNC `:6080` over Xvfb `:99`; `--viz kit --device cuda:0`. `--headless` is gone in GA.
  Commands in `docs/agibot/ops.md`.

### 1.5 One action label for human and planner demos

Human demos are 14-dim RMPFlow deltas; cuMotion demos are 20-dim absolute joint targets
(`AgibotDualArmJointActionsCfg`). They must land in one training set, so both paths record
`joint_pos_target` (T, 20) post-step via `PostStepJointPositionTargetRecorder`
(`isaaclab_arena/embodiments/agibot/demo_recorders.py`), and the LeRobot configs read
`action_name_sim: "joint_pos_target"`. `merge_demos.py --drop_mismatched` merges HDF5 sets;
`rerender_demo_cameras.py` re-renders cameras into sidecar mp4s so the HDF5 need not carry images
(`dataset_config.sidecar_camera_streams`). Verified end to end 2026-09-05 (mixed 2-episode set).

### 1.6 Device: everything on `cuda:0`

User decision 2026-09-05: teleop, recording, cuMotion, probes and tests all run with
`--device cuda:0`. `--device cpu` is a debugging mode only (it surfaces PhysX errors the GPU
pipeline swallows, e.g. velocity writes to fixed-base roots -- hence `_velocity_is_writable` in
`terms/events.py`).

### 1.7 The process for a new task or asset

`docs/agibot/new_task_playbook.md` -- eight gated stages, each with the probe that passes it
(`isaaclab_arena_cumotion/scripts/probe_{pinch,reach,drop_settle,staged_success}.py`). Rules that
are not obvious from the code:

- Only set what RoboDojo sets when porting a RoboDojo task; never tune the asset or the robot.
- `MassPropertiesCfg` silently no-ops without `UsdPhysics.MassAPI` on the prim.
- Asset origins are not geometry centres; pads sit 17 mm behind the tool frame; grasps default to
  the flipped wrist.
- Success terminations auto-reset the env: a "flung" object one step after staging is the harness.
- Graspable objects sit in the work band x 0.15-0.30 (`REACH_X_BAND_M`, re-measured 2026-09-06);
  the robot cannot get closer to the table, so layouts move to it.
- Control rate stays 15 Hz (matches the training data; upstream #1016).
- Triage by layer before touching anything: `docs/agibot/triage.md`.

The same process is exposed to coding agents as the `agibot-benchmark-task` skill
(`skills/developer/agibot-benchmark-task/SKILL.md`, alias in `.agents/skills/`).

---

## Part 2 -- Every difference from `origin/main`

State on 2026-09-05: `main` is `origin/main` (Isaac Lab 3.0 GA, `bb0c8e1b9` submodule) plus
21 local commits, 88 files, +13 715 / -34 lines. Nothing pushed. Two kinds of change: **new
files** (safe on every sync) and **edits to upstream-owned files** (the merge-conflict surface,
listed first).

### 2.1 Edits to upstream-owned files (watch these on every sync)

| file | what changed | why |
| --- | --- | --- |
| `isaaclab_arena/embodiments/agibot/agibot.py` | +428 lines: `AGIBOT_ARENA_A2D_CFG` (deep copy of the Lab cfg with the linkage fix, mirrored wrist, left gripper = right), Arena RMPFlow cfgs pointing at patched Lula yamls, `AgibotCameraCfg` + wrist camera refresh, dual-arm scene/action cfgs, `AgibotDualArmJointActionsCfg` (20-dim), `AgibotEventCfg`, observations incl. gripper state, `AgibotMimicEnv`, head view. Uses GA names `joint_effort_limit` / `joint_velocity_limit`. | sections 1.1-1.5 |
| `isaaclab_arena/terms/events.py` | `_velocity_is_writable` guard in `set_object_pose`; new `reset_joint_position_and_velocity_to_defaults`. | CPU-mode PhysX errors; robot-only reset |
| `isaaclab_arena/environments/isaaclab_arena_environment.py` | `env_cfg_callback` typed `Callable[[cfg], cfg]`, documented to return the cfg. | upstream bug 3 (callback result dropped) |
| `isaaclab_arena/environments/arena_env_builder.py` | asserts the callback returned a cfg. | same |
| `isaaclab_arena/assets/registries.py` | imports `local_objects` so local assets register. | local assets |
| `isaaclab_arena/assets/device_library.py`, `retargeter_library.py` | register `dual_arm_keyboard` + a no-op Agibot retargeter. | section 1.4 |
| `isaaclab_arena/tasks/task_library.py` | registers the five Agibot tasks. | tasks |
| `isaaclab_arena/tasks/predicates/spatial.py` (+308), `predicate_utils.py` | `objects_upright`, `lowest_object_upright`, `objects_stacked`, `objects_at_rest`, `objects_upright_about_any_axis`, `any_object_near_body`, `*_in_frame_box`, `get_root_quat_w`. | task success checks |
| `isaaclab_arena/utils/cameras.py` | `get_viewer_cfg_from_robot_body`. | head view |
| `isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py`, `config/dataset_config.py` | sidecar camera mp4 path (`sidecar_camera_streams`, `sidecar_camera_dir`). | section 1.5 |
| `pyproject.toml` | adds `isaaclab_arena_cumotion*` to the packages. | cuMotion package |
| `docs/source/environment/environment_definition.rst`, `docs/conf.py` | callback doc line; `docs/agibot` excluded from the Sphinx build. | docs |

### 2.2 New files by area

| area | files |
| --- | --- |
| robot | `embodiments/agibot/rmpflow/agibot_{left,right}_arm_gripper.yaml` (patched Lula descriptions), `embodiments/agibot/demo_recorders.py`, `embodiments/common/smooth_joint_actions.py` (first-order hold for joint targets) |
| control stack | `utils/arm_target_hold.py`, `utils/surface_guard.py`, `utils/ramped_gripper.py` |
| teleop device | `devices/__init__.py`, `devices/dual_arm_keyboard.py`, `devices/dual_arm_keyboard_cfg.py` |
| assets | `assets/local_objects.py` (RoboDojo ports, USDCraft `agibot_assets_v0`, re-centred Factory gear, peg/sleeve); USDs live outside the repo in `/home/ubuntu/playground/objects/` |
| tasks | `tasks/{stack_bowls,make_toast,handover_toast,sleeve_on_peg,tidy_workbench}_task.py`, `tasks/predicates/joints.py`, `tasks/stack_bowls_trace.py` (diagnostic CSV) |
| environments | `isaaclab_arena_environments/agibot_tabletop_common.py` (shared stage, `AgibotTabletopEnvironmentCfg`, `install_agibot_control_stack`), `agibot_{stack_bowls,make_toast,handover_toast,sleeve_on_peg,tidy_workbench}_environment.py` |
| cuMotion package | `isaaclab_arena_cumotion/` -- `planner.py`, `grasps.py`, `pick_place.py`, `executor.py`, `robot_description.py` (XRDF from the Lula files), `cumotion_embodiment_cfg.py`, `embodiment_cumotion_registry.py` |
| drivers and tools | `isaaclab_arena_cumotion/scripts/{stack_bowls,handover_toast,make_toast,sleeve_on_peg}_cumotion.py`, `merge_demos.py`, `rerender_demo_cameras.py`, `fix_eef_9d_rotations.py` |
| probes | `isaaclab_arena_cumotion/scripts/probe_{common,pinch,reach,drop_settle,staged_success,gripper_axes,gripper_span,tool_orientation,make_toast}.py` |
| GR00T / LeRobot | `isaaclab_arena_gr00t/embodiments/agibot/` (`20dof_joint_space.yaml`, `34dof_joint_space.yaml`, `gr00t_20dof_joint_space.yaml`, `modality.json`, `info.json`), `lerobot/config/agibot_*_config.yaml` |
| process docs and skill | `docs/agibot/*.md`, `skills/developer/agibot-benchmark-task/SKILL.md` (+ `.agents/skills/` alias), this file |
| tests | Agibot-specific tests under `isaaclab_arena/tests/` (Phase 1 baseline 1114 passed on GA) |

### 2.3 Things we deliberately did **not** change

- No `docker/`, `.github/workflows/`, `.pre-commit-config.yaml` or `submodules/` edits.
- No gripper stiffness/damping, arm PD, RMPFlow parameters or control rate changes beyond the
  effort ceiling -- all refuted by measurement (`docs/agibot/triage.md`, refuted-fixes table).
- No asset physics tuning of USDCraft/RoboDojo assets (hinge friction, mass) -- stock values are
  tuned.
- Old assets and datasets stay under `/home/ubuntu/playground`, never in the repo.

---

## Part 3 -- Keeping this mergeable

Procedure and the last impact snapshot: `docs/agibot/upstream_sync.md`. Short form:

1. `git fetch origin && git log --oneline main..origin/main -- submodules/IsaacLab
   isaaclab_arena/embodiments/agibot isaaclab_arena/terms isaaclab_arena/tasks/predicates
   isaaclab_arena/assets pyproject.toml` -- anything here touches section 2.1.
2. Trial-merge in a scratch worktree; resolve; `uv sync --extra dev`; if `pxr` imports break,
   `uv sync --extra dev --reinstall-package usd-exchange`.
3. Smoke: `probe_reach` + `probe_pinch` on `agibot_stack_bowls`, then Phase 1
   (`.venv/bin/python -m pytest -m 'not with_cameras and not with_subprocess' isaaclab_arena/tests`).
4. A teleop session through noVNC, then commit locally with `git commit -s` (user identity, no AI
   attribution). Pushing is the user's call.

Upstream's tests do pass; when Phase 1 fails after a sync the cause has so far always been local
(a corrupted `usd-exchange` install, a module-level `carb` import in one of our files).

## Part 4 -- Open decisions (user)

- Push `main` / open the team baseline PR.
- Report the two Isaac Lab asset issues (section 1.1) upstream -- on hold.
- Re-record or trim the demos recorded before 2026-09-05 (they start with the reset transient).
- make_toast handover point; tidy_workbench layout; sleeve_on_peg insertion approach; a
  joint-space teleop path (Pink IK candidate) -- see `docs/agibot/README.md`.
