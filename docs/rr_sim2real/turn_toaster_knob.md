# Four toaster dials on two boxx supports

Environment: `ur7e_usdcraft_turn_toaster_knob`.

For the matched 200-demo randomized datasets, use the
[bulk collection pipeline](collect_toaster_knob.md), not the fixed-pose command below.
Bulk support boxes are now kinematic rigid bodies so reset/replay can move and
record them; the source geometry and fixed support behavior during execution
are unchanged. Earlier qualification notes below describe the original static setup.

This new task reuses USDcraft toast, the UR7e robot with black fingertips, calibrated
D435, table, lights, and the press-toaster yaw (0 degrees, controls facing the robot).
The drawer pedestal is not present. Two full-size `boxx.usdc` boxes are stacked
below the toaster. No source mesh or USD has been resized or overwritten.

- Box dimensions: 0.3225 x 0.258133 x 0.1178 m.
- Stack height: 0.2356 m; tabletop z=0.75 m; toaster base z=0.9856 m.
- Fixed placement x=-0.16, y=0.12 m; no placement randomization yet.
- Dial center: approximately (-0.162, 0.0069, 1.0336) m in world coordinates.
- Dial diameter approximately 30 mm; joint `browning_rotation`, original range
  0–246.3719 degrees. The source dial has a working revolute joint.
- Success: dial angle magnitude exceeds 2 degrees from its zero-angle reset.
  This small nonzero threshold rejects numerical noise; there is no target level.
- Dial and unused carriage positions/velocities are reset. The source has no
  available negative range, although the movement predicate accepts either sign.

`tools/rr_sim2real/asset_overlays/boxx_support.usda` references the external original
box and makes its body/lid static while retaining visual geometry and collisions.
The lid stays closed; these are supports, not a lid-opening task. The toaster
housing remains fixed as in the pressing task. The overlay currently uses the
source-host absolute asset path; preserve the original box texture dependencies
when transferring and adapt that reference on another host.

## Qualification

Driver: `isaaclab_arena_cumotion/scripts/ur7e_turn_toaster_knob_cumotion.py`.
It checks passive drift, plans an approach using cuMotion, physically closes the
gripper on the dial and turns the wrist about the dial axis. It does not write the
dial joint during the manipulation. Output directories must be fresh.

```bash
cd /home/ubuntu/code/IsaacLab-Arena-tasks
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  PYTHONPATH=/home/ubuntu/code/IsaacLab-Arena-tasks:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab \
  .venv/bin/python isaaclab_arena_cumotion/scripts/ur7e_turn_toaster_knob_cumotion.py \
  --output outputs/toaster_knob_new_trial
```

2026-09-13 first physical qualification succeeded: passive angle 0 degrees,
angle after closing the gripper 0 degrees, final dial rotation 2.02366 degrees.
Tool tilt/spin was 75/0 degrees. Wrist command reached 3 degrees before the dial
passed the 2-degree success threshold. This is one fixed-pose trial, not a
collection-ready robustness result. Planner obstacles currently include table
and supports; full toaster-housing obstacle qualification remains follow-up work.

Evidence: `outputs/toaster_knob_qualification_20260913/{result.json,d435.mp4,scene.mp4}`.
The D435 shows the dial, but the top of the raised toaster is cropped; camera
calibration has not been changed. Third-person preview shows the complete stack.
No training dataset or DP has been collected/trained for this task yet.

## Self-collision audit (2026-09-13)

Repeat the command with `--audit-self-collision` and a fresh output directory to
check measured arm joints after every physics step, without changing the motion.
The fixed-pose repeat checked 1,467 steps across initial hold, gripper opening,
planned standoff, straight approach, closing, turning and settling: zero detected
self-collision frames; final dial rotation was again 2.02366 degrees.
Evidence: `outputs/toaster_knob_self_collision_20260913/` (`result.json`,
`self_collision_audit.json`, and both camera videos).

This is a configured cuMotion sphere-model check, not exact mesh-contact
certification. The shared robot config has `enabled_self_collisions=False`;
PhysX therefore does not provide robot self-contact response. The checker keeps
the existing ignored link pairs, and approximates the whole gripper using four
fixed spheres on `tool0`, without tracking finger opening. No collision settings,
ignore pairs or trajectory targets were changed for this diagnostic run.

## Comparison environments and recording

The same stack, pose, yaw, cameras, black gripper and 2-degree threshold are used
for all four methods. Existing toast payloads/overlays are reused without new
geometry changes; Articraft keeps its previously normalized size.

| Method | Environment | Dial joint | Dial body |
| --- | --- | --- | --- |
| USDcraft | `ur7e_usdcraft_turn_toaster_knob` | `browning_rotation` | `browning_dial` |
| Articraft | `ur7e_articraft_turn_toaster_knob` | `housing_to_browning_dial` | `browning_dial` |
| miniworkflow GPTSOL | `ur7e_miniworkflow_gptsol_turn_toaster_knob` | `DialRevolute` | `BrowningDial` |
| miniworkflow Astra | `ur7e_miniworkflow_astra_turn_toaster_knob` | `BrowningDial` | `Dial` |

Each dial resets to zero **radians**, not its lower joint limit. Both object
joint velocities reset to zero; the unused lever resets to 0.004 m for GPTSOL,
0 m for the other three. Body-frame grasp offsets account for each asset's dial
geometry and Articraft's rotated dial body. The wrist rotates about outward -Y;
the actual signed angle can differ across assets. Success uses angle magnitude.

Run one fixed-pose qualification recording for each comparison:

```bash
bash tools/rr_sim2real/qualify_toaster_knobs.sh
```

Or run one environment directly using the native runtime variables above:

```bash
.venv/bin/python isaaclab_arena_cumotion/scripts/ur7e_turn_toaster_knob_cumotion.py \
  --env ur7e_articraft_turn_toaster_knob --output outputs/knob_articraft_new_trial \
  --record-demo --audit-self-collision
```

`--record-demo` uses `EnvActionExecutor` and the existing UR7e recorder: 15 Hz,
7 absolute joint actions `[6 arm, finger_joint]`, `joint_pos_target`, full scene
states and initial states. The HDF5 carries the registered environment name.
It exports successful episodes only, ending at the first successful control
sample during turning, without retreat or settling. `demo.hdf5`, `result.json`,
`self_collision_audit.json`, `d435.mp4` and `scene.mp4` are saved together.
Recording audits measured joints at **control-step** resolution; non-recording
audits remain per physics step. Neither certifies exact mesh contact.

These are single-demo qualification artifacts under `outputs/`, not formal
training datasets. Placement/background randomization, bulk retry orchestration,
and LeRobot/DP conversion are not enabled by this qualification command. Replay
can use `rerender_embodiment_cameras.py --env <same environment> --hdf5 <demo.hdf5>`;
use a fresh sidecar directory and preserve the calibrated 640x480 D435 resolution.

### Verified comparison recordings (2026-09-13)

Final code, one fixed-pose recording per method (not a randomized success rate):

| Method | Success | Final signed angle | HDF5 samples | Detected self-collision samples |
| --- | --- | --- | --- | --- |
| Articraft | yes | +2.09792 degrees | 204 | 0 |
| miniworkflow GPTSOL | yes | -2.92306 degrees | 187 | 0 |
| miniworkflow Astra | yes | +4.12172 degrees | 186 | 0 |

Evidence directories: `outputs/knob_<method>_verified_20260913/`, with method
spelled as in the environment names. All three HDF5 files were checked for one
successful episode, correct environment metadata, finite `(N, 7)` actions and
targets, zero initial dial angle, correct lever rest, and only the final sample
exceeding the 2-degree threshold. Passive drift and dial motion after closing
the gripper were both zero. All selected the 75-degree tilt / 0-degree spin
candidate. D435 final frames were visually inspected for the black fingertips.
Earlier `knob_*_collect_20260913` artifacts are development trials; the verified
directories supersede them (GPTSOL initially retained one extra success sample).

USDcraft regression with the same final recording driver also passed: 183 samples,
+2.53906 degrees, zero detected self-collision samples, and first threshold crossing
at the last HDF5 sample. Evidence: `outputs/knob_usdcraft_verified_20260913/`.
