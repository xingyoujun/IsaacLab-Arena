# New Agibot task playbook

The fixed sequence for bringing a new benchmark task from "an idea and some USD files" to a
collected dataset. Each stage has a **gate**: a measurement with a pass criterion and the tool that
produces it. Do not start the next stage until the gate passes, and do not tune anything outside
the stage you are in. The whole point is that when something fails you already know which layer it
is in.

All probes run headless from the repo root with the uv environment:

```bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
.venv/bin/python isaaclab_arena_cumotion/scripts/<probe>.py --env <env> [...]
```

Every probe accepts `--env-arg FIELD=VALUE` (repeatable) to override the environment's typed
config, e.g. `--env-arg billet_scale=1.2`.

## The gates at a glance

| stage | gate | tool | pass |
| --- | --- | --- | --- |
| 0 asset intake | origin convention, collider, mass known | `probe_drop_settle.py` | resting height explained; tilt 0; noise floor recorded |
| 1 graspability | the gripper holds the object at all | `probe_pinch.py` | every repeat HELD after close and swing; peak < 1 m/s |
| 2 layout | every grasp and release point is reachable | `probe_reach.py` | >= ~10 poses at lean 30-50 for grasps; > 0 for releases |
| 3 task | success accepts a solved scene and only that | `probe_staged_success.py` | True on the staged goal; survives 2 s of physics; False at reset |
| 4 teleop | a person solves it; nothing flies | Kit teleop via noVNC | 3 clean solves in a row |
| 5 recording | demos are complete and replay | `record_demos.py` + `rerender_demo_cameras.py` | success flag set, states replay, cameras render |
| 6 cuMotion | the scripted driver reaches success | `<task>_cumotion.py --video` | one clean video the user has watched |
| 7 collection | throughput and success rate hold | collection driver | >= 15 % success, workers stable, disk budgeted |
| 8 conversion / training / eval | dataset loads, policy deploys | `convert_hdf5_to_lerobot.py`, GR00T server + `policy_runner.py` | conversion clean; closed-loop episode runs |

## Stage 0 -- asset intake

Inputs: a USD (USDCraft export, RoboDojo asset, Arena library asset) and its intended role
(graspable part, fixture, container, articulated tool).

1. Register it in `isaaclab_arena/assets/local_objects.py` as a `LibraryObject` with **measured
   constants** as class attributes (half extents, rim height, origin offset) and a docstring that
   says where the origin is. Fixtures get `spawn_cfg_addon` with `kinematic_enabled=True`.
2. If the USD holds several rigid bodies under one prim, split it with `active = false` override
   layers referencing the whole asset (keeps material bindings), one `.usda` per body -- see
   `peg_platform.usda` / `sleeve.usda`.
3. Single-link "articulations" with zero joints (Factory peg, gears) must be loaded as RIGID: strip
   `PhysicsArticulationRootAPI` in the override layer, or the CPU device (Kit teleop) crashes in
   `Articulation._create_buffers`. GPU builds fine, which is why it only shows up in teleop.
4. Put it in a scratch layout of the target environment and run `probe_drop_settle.py`.

Gate: the "origin above surface" reading matches the asset's documented origin convention (0 mm =
bottom origin, half height = centre origin). Anything else means the geometry is off its origin
(the Factory gear was 50 mm off) -- re-centre in the override layer before placing by origin. Record
the at-rest speed: that is the noise floor a rest threshold must clear (a lone body reads ~0.001
m/s; a body in a contact stack reads 0.03-0.05 m/s and never decays).

Do **not** change colliders, friction, mass or joint friction on an asset the benchmark uses as-is
unless the source (RoboDojo loader, USDCraft metadata) does the same. Report the geometry problem
instead. See [asset_guide.md](asset_guide.md) for the reasons and the exceptions.

## Stage 1 -- graspability

Run `probe_pinch.py` on every object the robot must hold, at the scale you intend to use:

```bash
.venv/bin/python isaaclab_arena_cumotion/scripts/probe_pinch.py \
    --env agibot_tidy_workbench --object metal_billet --centre_offset 0.010 --repeats 3
```

`--centre_offset` is origin-to-pinch-point along the object's z (half the height for a
bottom-origin part). Try `--object_roll_deg 90` for the other in-plane side, and `--arm left`.

Gate: HELD/HELD on every repeat with a close peak well under 1 m/s (a clean hold is 0.1-0.35 m/s;
an ejection is 2-7 m/s). This probe has no table and no approach, so a failure is purely the
object's geometry against the gripper's: longer than the ~80 mm pad region (the finger linkage
inboard of the pads strikes it), wider than the 104 mm jaw opening, a grip band shorter than ~12
mm over a step, or a wide flange under a narrow neck. **No gripper or arm parameter fixes this**
(all were tried -- [triage.md](triage.md)). Resize (billet 1.0 passes, 1.2 marginal, 1.5 impossible),
pick a different grasp feature, or regenerate the asset. Before scaling parts up "for looks",
run this probe.

## Stage 2 -- layout

Lay the scene out from the shared constants in `agibot_tabletop_common.py`: robot at
`ROBOT_POSITION_XYZ`, work surface at `TABLE_TOP_Z` (0.6232, measured), graspable objects inside
`REACH_X_BAND_M` (0.35-0.45). Then run the reach scan on every grasp point and every release point:

```bash
.venv/bin/python isaaclab_arena_cumotion/scripts/probe_reach.py --env <env> \
    --point 0.43,-0.16,0.66 --point 0.52,0.30,0.83 --tilts 0,30,50,75 [--horizontal]
```

Gate: a grasp spot should read a dozen or more reachable poses at lean 30-50; a release spot needs
more than zero at some lean. Known shape of the workspace: pure top-down (tilt 0-20) is unreachable
at table height; the two arms' workspaces do not overlap over the table (make_toast's handover is
structural); the far side (x >= 0.50) is reachable only with the 75-degree near-horizontal wrist;
the left arm is not the mirror of the right (joint 3 axis flipped, joint 7 range halved) -- scan
both. Horizontal side grasps become reachable only with the surface raised 10-15 cm.

Also decide jitter here: clamp x into the reach band, reject draws that violate pairwise
separation, and reject draws whose yawed footprint enters any fixture footprint (a part spawned
into a container collider is ejected off the table). Keep a per-object "keep-out" list.

## Stage 3 -- task and success check

Write the task (`TaskBase` subclass) with predicates from `tasks/predicates/spatial.py`
(`objects_upright`, `objects_stacked` with `max_z_gap`, `objects_at_rest`, `any_object_in_frame_box`,
`count_objects_in_frame_box`, `any_object_near_body`) and `tasks/predicates/joints.py`
(`joint_past_travel_fraction`). Rules that each cost a session:

- `TerminationsCfg.time_out` needs `time_out=True` (a repo test sweeps every task for it).
- Reset events on the embodiment touch only the robot. Object resets are the task's or the
  environment's job, and remember that success triggers an **automatic reset**: an event that
  tramples objects fires the moment the task succeeds.
- `Pressable.is_pressed` inverts any joint whose lower limit is negative. For a RoboDojo-faithful
  threshold use `joint_past_travel_fraction`, and verify the polarity by driving the joint to each
  limit and watching the link.
- Rest thresholds: calibrate against the contact configuration the task ends in (stacked bowls
  need 0.1 m/s, 4x the resting artefact), never against a lone body.
- Per-episode latching state lives on the env (`get_env(env)` pattern), not on `self`: Arena
  deep-copies the task into several manager configs.
- Kinematic fixtures log two `Body must be non-kinematic!` errors per reset on the CPU device;
  cosmetic, but use `_velocity_is_writable` in your own events to avoid it.

Gate: `probe_staged_success.py` writes the goal poses and reads `is_success` before and after
physics:

```bash
.venv/bin/python isaaclab_arena_cumotion/scripts/probe_staged_success.py \
    --env agibot_sleeve_on_peg --pose peg_sleeve=0.40,0.00,0.6432 --physics_seconds 2
```

Pass: False before staging, True on the staged pose, still True after 2 s (or you have found that
the goal pose is not a resting state -- e.g. `place_upright()` sinks a lying object's base into the
table and PhysX pops it out). If the predicate needs several objects, stage them all.

## Stage 4 -- environment and teleoperation

The environment subclasses `AgibotTabletopEnvironmentCfg`, builds its stage with
`build_tabletop_stage`, its robot with `build_agibot`, and its `env_cfg_callback` starts with
`install_agibot_control_stack(env_cfg, cfg)` -- target hold, surface guard, ramped gripper, arm
gains, in that order. That stack is the standard; every task has it (a template is the
`agibot_stack_bowls` environment). Then add the task's own events after it.

Teleop through Isaac Lab's script with **both** device flags
(see [ops.md](ops.md) for the display set-up):

```bash
DISPLAY=:99 ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  .venv/bin/python submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
  --viz kit --device cpu \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task <env> --arena_teleop_device dual_arm_keyboard --teleop_device dual_arm_keyboard
```

Gate: three clean solves in a row by a person, from the head view, with no object thrown. If an
object flies, go to [triage.md](triage.md) -- run the pinch probe first, do not touch the gripper.

## Stage 5 -- human recording

`record_demos.py` builds the env through the same callback, so the control stack carries over;
it removes the success termination itself. `teleop_se3_agent.py` records nothing.

```bash
DISPLAY=:99 ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  .venv/bin/python submodules/IsaacLab/scripts/tools/record_demos.py --viz kit --device cpu \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task <env> --arena_teleop_device dual_arm_keyboard --teleop_device dual_arm_keyboard \
  --step_hz 15 --num_demos 0 --num_success_steps 10 \
  --dataset_file /home/ubuntu/playground/datasets/<task>_teleop/teleop_demos.hdf5
```

Record at **15 Hz** (Arena's default; training data is 15 Hz -- never "fix" a failure by going to
50 Hz). `obs/eef_pos` records only the left end-effector; replay joint states for the right tool.

Gate: the HDF5 has `success` set on each demo and the states replay
(`rerender_demo_cameras.py --hdf5 ... --env <env>` renders head and wrist views from the states).

## Stage 6 -- cuMotion driver

Write `isaaclab_arena_cumotion/scripts/<task>_cumotion.py` after the pattern of
`stack_bowls_cumotion.py` (grasp generator -> `PickAndPlace` -> `EnvActionExecutor` when
recording). The rules that are not obvious are in [cumotion_guide.md](cumotion_guide.md); the ones
that cost the most: the tool frame sits 16.7 mm past the pads; aim at the bounding-box centre, not
the origin; `pick()` descends wide open and sorts by joint travel; never `env.step()` while the
executor drives; the left tool frame is relabelled 90 degrees in the generated description (the
planner measures the correction); default wrist roll is `flipped`.

Gate: one video per change, sent to the user, who decides the next step
(`--video /home/ubuntu/playground/videos/<name>.mp4`). Do not iterate alone on a failed run.

## Stage 7 -- collection

Use the driver pattern in `/home/ubuntu/playground/datasets/collect_agibot_arena_v0.sh` (two
workers per round, adaptive round size, merge, trim, re-render, convert). Seeds from the epoch and
incremented per round; kill stale `<task>_cumotion` processes before restarting; two Isaac Sim
processes is this host's ceiling. Budget: ~6 h per 100 bowls demos, ~12 h per 100 handover demos.

Gate: success rate >= 15 % and no repeated worker crashes; below that, stop and triage.

## Stage 8 -- conversion, training, evaluation

Conversion: `convert_hdf5_to_lerobot.py --yaml_file isaaclab_arena_gr00t/lerobot/config/agibot_<task>_v0_config.yaml`
(LeRobot v2.1, 15 fps, sidecar camera mp4s). `observation.state` is relative to the reset pose and
`action` is absolute joint targets -- a user-accepted convention. End-effector poses (`eef_9d`) must
be produced from the xyzw quaternion correctly; `fix_eef_9d_rotations.py` repairs the known
scramble.

Training is **not** run on this host (it froze the machine on 2026-09-02). Evaluation uses the
two-container GR00T pattern (server container + `policy_runner.py` client over ZMQ, port 5757).

## Recording the outcome

When a stage's measurement settles a question (a scale, a threshold, a layout), put the number and
the reason in the code (field docstring or module comment) and, if it changes the process, in
these documents. Session memory is for what the repo cannot hold: pending user decisions and
working preferences.
