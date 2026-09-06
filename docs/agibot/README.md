# Agibot benchmark development in Arena

Working notes of the team building new Agibot (A2D, dual-arm) benchmark tasks in Isaac Lab-Arena.
Our deliverable per task is the whole chain, in this order:

1. **Scene** -- the task exists as a registered Arena environment with a validated success check.
2. **Teleoperation + human recording** -- a person can solve it from the head view with the
   dual-arm keyboard, and `record_demos.py` produces clean HDF5 demonstrations.
3. **cuMotion collection** -- a scripted pick-and-place driver produces demonstrations at scale.
4. **Training + closed-loop deployment** -- LeRobot conversion, a fine-tuned policy (on another
   machine), evaluation back in the same environment.

Every stage has bitten us with "the object flies", "the arm cannot reach", "success never fires",
and every time the first hour went to finding out *which layer* was wrong. These documents fix the
process so that does not happen again:

| document | read it when |
| --- | --- |
| [../../dev.md](../../dev.md) | first -- the map of every difference from upstream Arena and the improvements an agent must apply on top of it |
| [new_task_playbook.md](new_task_playbook.md) | starting any new task or asset -- the mandatory gate sequence, with the tool for each gate |
| [triage.md](triage.md) | something misbehaves -- one decision tree to name the layer at fault, plus the refuted-fixes list |
| [asset_guide.md](asset_guide.md) | choosing, sizing or registering an object -- the gripper envelope and the asset intake rules |
| [agibot_embodiment.md](agibot_embodiment.md) | anyone proposes touching the robot -- the frozen default config and why each knob is where it is |
| [cumotion_guide.md](cumotion_guide.md) | writing or tuning a scripted driver, or running a collection |
| [ops.md](ops.md) | running anything on this host -- environment, display, commands, data layout, limits |
| [pipeline_paths.md](pipeline_paths.md) | wondering how teleop, recording and cuMotion differ -- the measured side-by-side, and the CPU/GPU question |
| [control_paths_research.md](control_paths_research.md) | asking why teleop uses RMPFlow, whether it causes the hard contacts, and how RoboDojo drives its robot -- research notes and an experiment plan |
| [upstream_sync.md](upstream_sync.md) | keeping up with `origin/main` -- the sync procedure and the impact snapshot of the current upstream delta |
| [upstream_bug_report_2026-08.md](upstream_bug_report_2026-08.md) | the three Arena defects we fixed locally and reported (historical) |

The same process is available to coding agents as the `agibot-benchmark-task` skill
(`skills/developer/agibot-benchmark-task/SKILL.md`, aliased from `.agents/skills/`), which points here.

## The one rule

**The robot is frozen. Tasks adapt to it.** The Agibot's default configuration in
`isaaclab_arena/embodiments/agibot/agibot.py` and the standard control stack in
`isaaclab_arena_environments/agibot_tabletop_common.py` are the same for every task, and were each
settled by measurement (see [agibot_embodiment.md](agibot_embodiment.md)). When a task does not work,
the fix is in the asset, the layout, the grasp or the task code -- never in an actuator gain, a
gripper limit or a controller flag. Every gripper- or arm-side "fix" we tried made things worse.

## Where the code is

| layer | files |
| --- | --- |
| embodiment (frozen default) | `isaaclab_arena/embodiments/agibot/agibot.py`, `isaaclab_arena/embodiments/agibot/rmpflow/*.yaml` (patched Lula descriptions), `isaaclab_arena/embodiments/common/smooth_joint_actions.py` |
| standard control stack | `isaaclab_arena/utils/arm_target_hold.py`, `isaaclab_arena/utils/surface_guard.py`, `isaaclab_arena/utils/ramped_gripper.py`; installed by `install_agibot_control_stack` |
| shared stage + base config | `isaaclab_arena_environments/agibot_tabletop_common.py` (table, room, light, robot pose, `AgibotTabletopEnvironmentCfg`) |
| teleop device | `isaaclab_arena/devices/dual_arm_keyboard.py`, registered in `assets/device_library.py` + `assets/retargeter_library.py` |
| local assets | `isaaclab_arena/assets/local_objects.py` (RoboDojo ports, USDCraft `agibot_assets_v0`, Factory re-centred gear); USDs under `/home/ubuntu/playground/objects/` |
| tasks | `isaaclab_arena/tasks/{stack_bowls,make_toast,handover_toast,sleeve_on_peg,tidy_workbench}_task.py`; shared predicates in `tasks/predicates/spatial.py` and `tasks/predicates/joints.py` |
| environments | `isaaclab_arena_environments/agibot_*_environment.py` |
| cuMotion package | `isaaclab_arena_cumotion/` (planner, grasps, pick_place, executor, robot_description, per-arm registry) |
| scripted drivers | `isaaclab_arena_cumotion/scripts/{stack_bowls,handover_toast,make_toast,sleeve_on_peg}_cumotion.py` |
| measurement probes | `isaaclab_arena_cumotion/scripts/probe_*.py` -- see the table in the playbook |
| dataset tooling | `isaaclab_arena_cumotion/scripts/rerender_demo_cameras.py`, `fix_eef_9d_rotations.py`; LeRobot configs `isaaclab_arena_gr00t/lerobot/config/agibot_*.yaml`, embodiment metadata `isaaclab_arena_gr00t/embodiments/agibot/` |
| diagnostics | `isaaclab_arena/tasks/stack_bowls_trace.py` (per-step CSV, on only with `ARENA_STACK_BOWLS_TRACE=<path>`) |

## Task status board (2026-09-05)

| task | env | scene | teleop | human demos | cuMotion driver | collection | dataset | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stack bowls | `agibot_stack_bowls` | done | done (stack installed 09-05) | a few, 08-03 era | done, ~90 % success | 200 demos | `agibot_arena_v0/stack_bowls` (+ eef_9d, fixed) | jaw spin pinned per hand; `--jitter 0.04` |
| handover toast | `agibot_handover_toast` | done | done (stack installed 09-05) | 3 | done, ~55-60 % | 200 demos | `agibot_arena_v0/handover_toast` (+ eef_9d, fixed) | rounds 1-2 of the first 50 recorded under a pulsing grip |
| make toast (full) | `agibot_make_toast` | done | done (stack installed 09-05) | -- | step 1 (pick) stable; step 2 (chest handover) **paused** | -- | -- | **user decision pending**: handover point (`--chest-x/--chest-z`) |
| sleeve on peg | `agibot_sleeve_on_peg` | done (uncommitted) | done | 1 (right arm, 09-03) | grasps like the human (pitched 55-80 deg); insertion not yet reliable | -- | -- | SDF sleeve kept; convex/ring variants beside it |
| tidy workbench | `agibot_tidy_workbench` | done (uncommitted) | done | in progress (`tidy_workbench_teleop/`) | wrench lift 3/5 (IK-limited); no full driver | -- | -- | bearing replaced by `small_gear_centred`; far-side containers need the 75-deg wrist |

Archived, not on `main`: `push_T`, `press_button`, `store_laptop_and_headphones` ports live on the
branch `chuanruiz/feature/robodojo-tasks` (pre-baseline code; the Agibot config there is stale).

## Open decisions for the user

- make_toast step 2: choose the chest handover point, then resume `make_toast_cumotion.py`.
- Push the prepared Arena bug-fix branch (`chuanruiz/fix/agibot_left_arm_and_env_cfg_callback`)
  upstream, or fold it into the team baseline PR.
- tidy_workbench: keep the far-side container layout (looks right, thin IK) or move containers in.
- sleeve_on_peg: cuMotion insertion reliability vs regenerating the sleeve with a looser bore.

## Data and artifacts

Everything large lives on the host under `/home/ubuntu/playground/` (never in the repo, never in
`/tmp`): `datasets/` (raw HDF5 per task in `<task>_v0_raw/`, converted LeRobot v2.1 in
`agibot_arena_v0/<task>/`, teleop sessions in `<task>_teleop/`), `objects/` (USD assets),
`videos/`, `make_toast/` (collection logs). Details and the clean-up candidates are in
[ops.md](ops.md).
