# G2 stack bowls

The `g2_stack_bowls` environment uses the G2 omnipicker with both arms enabled.
The table is 0.75 m high, and the bowls are arranged along its near edge: bowl_1
on the right, bowl_2 in the centre, bowl_3 on the left. Stack bowl_1 into bowl_2,
then bowl_3 onto bowl_1. The room includes a floor, walls, cabinets and shelves.

Run inside this clone's Arena container (the Genie Sim assets must be available
at `/datasets/GenieSimAssets`, or set `GENIESIM_ASSETS_DIR`):

```bash
cd /workspaces/isaaclab_arena
/isaac-sim/python.sh submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
  --device cpu --viz kit --num_envs 1 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task g2_stack_bowls --teleop_device keyboard --arena_teleop_device keyboard
```

Click the viewport to focus keyboard input. The selected arm starts on the right;
Tab switches arms and prints the selection in the terminal. Release and press a
movement key again after switching. Each gripper retains its own open/closed state.

| Keys | Action on the selected arm |
| --- | --- |
| Tab | Switch right/left arm |
| W / S | Forward / backward |
| A / D | Left / right |
| Q / E | Up / down |
| Z / X | Roll |
| T / G | Pitch |
| C / V | Yaw |
| K | Toggle gripper |
| L | Reset keyboard state: right arm selected, both grippers open |
| R | Reset the environment (teleop script binding) |

`--table_height_m` changes the floor-to-tabletop height while keeping the tabletop
at world z=0. The default robot base follows the floor automatically; an explicit
`--robot_position x y z` overrides this. `--bowl_positions` takes six values: x/y
for bowl_1, bowl_2 and bowl_3. `--arm_mode right` or `--arm_mode left` retains the
single-arm, seven-value action interface.

Dual-arm actions have 14 values: right delta pose (6), right gripper (1), left
delta pose (6), left gripper (1). Existing `eef_pos`/`eef_quat` observations describe
the right TCP; `left_eef_pos`/`left_eef_quat` describe the left TCP. G2 retains its
own differential IK, TCP frames and omnipicker drives. Agibot A2D's RMPFlow and
frame corrections are not used. Dual-arm Mimic data generation is not configured.

## Experimental cuRobo collection

`isaaclab_arena_examples/g2_stack_bowls_curobo.py` uses cuRobo MotionGen (the
motion-planning backend, not a ROS cuMotion server) to plan and physically execute
right-arm then left-arm rim grasps. Use the optional cuRobo container image
(`./docker/run_docker.sh -c`). The first build installs CUDA and compiles cuRobo.
The regular WebRTC teleoperation container does not require these dependencies.

Supply the Genie Sim G2 collision-sphere YAML
`source/data_collection/config/curobo/configs/robot/G2_omnipicker_fixed_dual.yml`
and matching fixed-base URDF
`source/data_collection/config/robot_cfg/G2/G2_omnipicker_fixed_dual.urdf`.
Make these external files available inside the container; they are not bundled
here. The similarly named benchmark G2 URDF has different kinematics and must not
be substituted. The collector checks its TCP against the simulator before planning.

```bash
/isaac-sim/python.sh isaaclab_arena_examples/g2_stack_bowls_curobo.py \
  --headless --device cpu \
  --robot_yaml /tmp/g2_curobo_vendor.yml \
  --robot_urdf /tmp/g2_curobo_fixed.urdf \
  --output_dir /datasets/g2_stack_bowls_curobo/pilot_001 --attempts 1
```

The output directory must be new. `report.json` records parameters, failure phase,
lift measurements and final bowl positions. Arena HDF5 recording separates
successful and failed episodes; a successful plan alone is not a successful demo.
The task predicate must remain true after both grippers release and the stack
settles. Add `--enable_cameras` to record both wrist views and a fixed 640x480
overview. Add `--record_video --enable_cameras` to also save a live overview MP4 per
attempt at simulation speed. `--table_height_m` defaults to 0.75 m and updates
both the physical scene and the planner's room obstacles.
Collection uses an external overview instead of the native head camera;
keyboard camera settings are not changed. The native head view was also verified
at the 0.75 m table height using the three-view renderer below.
Grasp parameters `--rim_offset`, `--grasp_height`, `--grasp_yaw` and
`--stack_release_offset` are experimental.

Collection uses **16 absolute joint-position actions**, ordered right arm (7),
right gripper (1), left arm (7), left gripper (1), with +1 open / -1 closed.
This differs from the keyboard environment's 14 delta-pose actions: replay or
training consumers must set `description.embodiment.action_config` to
`G2JointPositionActionsCfg()` from `isaaclab_arena.embodiments.g2.g2` before building
the environment. HDF5 metadata identifies this action configuration.
The default scene and keyboard controller are unchanged. The planner locks the
waist and inactive arm to their measured posture, checks collision spheres against
the room and other bowls, and represents the held bowl with attached collision
spheres. It parks the first arm outside the shared placement region before the
second arm places its bowl.
Planning queries tolerate at most 0.0001 rad of measured hard-stop overshoot;
larger joint-limit violations abort, and recorded physical states are never clipped.
Source-bowl contact is allowed for grasping; destination-bowl contact is allowed
during placement. At the previous 0.90 m table height, a headless pilot completed the full physical stack and exported
a successful 1208-step state/action episode. This is a pilot result, not a measured
success rate over varied scenes; validate small batches before bulk collection.
A second successful 1208-step episode includes RGB observations. Its two wrist
views are original capture; the overview was re-rendered from the recorded
pre-action states after correcting camera quaternion ordering. That pilot's HDF5
attributes and dataset README retain this provenance and preserve the invalid
original overview under a separate diagnostics group.

A subsequent 0.75 m table trial completed the full stack in 1208 steps, with
right/left bowl lifts of 0.15465/0.15426 m and maximum TCP position error 0.00676 m.
Its 80.53-second MP4 is live sensor capture; all three RGB streams are original.
This verifies the lower setup for one trial, not a batch success rate.

To export synchronized native head / left wrist / right wrist MP4s and a labeled
three-panel video from a collected episode:

```bash
/isaac-sim/python.sh isaaclab_arena_examples/g2_render_three_views.py \
  --headless --device cpu --enable_cameras --table_height_m 0.75 \
  --dataset /datasets/g2_stack_bowls_curobo/table075_video_002/episodes.hdf5 \
  --output_dir /datasets/g2_stack_bowls_curobo/table075_video_002/three_views
```

The output directory must be new. Head images are re-rendered from recorded
pre-action states using the authored robot camera; wrist images are original
recordings. No source data, camera extrinsics, actions or physical states are
modified, and no physics is stepped during replay. Provenance is saved in
`render_metadata.json`.

For seeded bowl-position randomization, permanent image-free raw, offline native
three-camera rendering and resumable LeRobot v2.1 collection of 200 successful
demos, see [DATA_COLLECTION.md](DATA_COLLECTION.md).
