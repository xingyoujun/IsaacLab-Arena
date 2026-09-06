# cuMotion drivers and collection

How the scripted demonstration path works, the facts that decide whether a grasp holds, and how a
collection run is operated. Read [new_task_playbook.md](new_task_playbook.md) stages 6-7 first for
where this sits in the process.

## The package

`isaaclab_arena_cumotion/` wraps Isaac Sim 6.0's bundled cuMotion (`isaacsim.robot_motion.cumotion`,
not the open-source cuRobo API and not RoboDojo's vendored one -- three incompatible surfaces share
the name):

| module | role |
| --- | --- |
| `embodiment_cumotion_registry.py` | per-arm `CumotionEmbodimentCfg` for the Agibot: Lula description from **Arena's patched yamls** (mirrored rest pose), URDF from the Isaac asset cache, tool frame, joint names, gripper open/closed (0.994 / 0.0), self-collision ignore list. `get_embodiment_cumotion_cfg(embodiment, arm=...)` |
| `robot_description.py` | builds the XRDF (format 2.0) from the Lula yaml + URDF at run time |
| `planner.py` | `CumotionArmPlanner`: world/obstacle model, `plan_pose`, `plan_best_pose`, `ik_reachable`, candidate filtering on joint-limit margin (>= 0.30 rad) and travel (<= 2.6 rad), and the measured left tool-frame correction |
| `grasps.py` | grasp generators: `rim_grasps` (bodies of revolution), `slab_grasps` (boxes/slices, from a **bounding box**), `annotated_point_grasps`; canonical `DOWN_FACING_ROTATION`; quaternions here are **wxyz** |
| `pick_place.py` | `PickAndPlace.pick / place`: candidate ordering, pregrasp gap, attempts, retargeted release |
| `executor.py` | `ArmExecutor` (direct joint writes + `env.sim.step`) and `EnvActionExecutor` (drives through `env.step` with `AgibotDualArmJointActionsCfg` so recorders see every step); gripper ramp 0.5 s (`AGIBOT_GRIPPER_RAMP_SECONDS`), settles in **seconds** |

## Facts a grasp author must know

1. **Aim at the bounding-box centre, not the origin.** `slab_grasps(bbox_min_m, bbox_max_m)`.
   The bread's origin is its bottom face; aiming at it straddled the slab 5.6 mm off-centre and one
   pad jammed the linkage while the other had 14.7 mm to go. Fixing it took lift 84 -> 120 mm.
2. **The tool frame is 16.7 mm past the pads** (`approach_offset_m=0.017`). Re-measure per arm.
3. **`pick()` descends wide open (104.5 mm).** Use `pregrasp_gripper_pos` (0.25 -> 30 mm) in
   clutter; slices 25-33 mm apart were ploughed otherwise.
4. **`pick()` sorts candidates by joint travel, not grasp quality.** Decisive when only the first
   candidate executes. Do not offer silly candidates (edge grasps) rather than changing the sort.
5. **`max_attempts=1` when a failed attempt disturbs the scene**; unlimited is right for bowls,
   which a miss leaves untouched. A place failure retires the grasp, not the object: set it down
   and pick again.
6. **Default wrist roll is `flipped`** (upright rolled 180 degrees about the approach) -- keeps the
   wrist camera looking out and makes runs comparable. Pin it in the generator call.
7. **Jaw spin is pinned per hand** (stack_bowls: left +90, right -90) for camera orientation; spin
   has zero effect on IK reachability and 0/180 makes the fingers close tangentially on a rim.
8. **The left tool frame is relabelled 90 degrees** in the generated description relative to the
   USD (right is identity). `CumotionArmPlanner._measure_tool_correction` measures it at
   construction; if you compute tool poses outside the planner, apply it yourself. The two arms
   are not mirror images (joint 3 axis flipped, right joint 7 range halved); joint limits are.
9. **Never call `env.step()` while `ArmExecutor` is driving.** Control returns to the action
   manager, the arm goes limp mid-air and drops the object directly below the staging pose.
   Hold with `executor.step(arm_target=q_hold, steps=n)`; `env.step` is only safe before the
   executor first drives.
10. **Release aims from a fresh measurement**: a rim-held bowl slips tens of mm in the carry;
    re-measure at the stand-off and replan the last descent (`place(retarget=...)`). Open the
    gripper **half way** to let go (full span hooks the far wall) and leave **straight up**.
11. **Speeds and settles are frozen values**: trajectory speed 0.35 of time-optimal (the stiff arm
    lags above that and the gripper closes on air); handover climb 0.10, cross 0.06, lower 0.06,
    reach 0.10, close-in 0.06, retreat 0.12; settle 4 s wall clock. A 3x speed-up shed the slice.
12. **Plans that succeed but do not execute**: wrist parked at a joint limit (filter margin), IK
    branch flips sweeping through the torso (cap travel), unmodelled obstacles (add the other
    tabletop objects as boxes at their live poses). Cross-check `kin.pose(q)` against
    `body_pos_w` -- and the **orientation** (`probe_tool_orientation.py`), since positions agree
    even when frames do not.
13. **Ramp the gripper.** A stepped close discharges the closed-loop linkage into the
    zero-damping arm actuators. The executor does this (`DEFAULT_GRIPPER_RAMP_SECONDS`).

## Tuning loop (user directive, 2026-08-26)

One change -> one video (to a user-specified directory under `/home/ubuntu/playground/`) -> report
the outcome and the key log lines -> wait for the user's call. Do not watch the video yourself to
pick the next change; do not fix-and-retry on a failed run. The only exception is a run that
produces no video at all (crash at start-up), which may be fixed until the video exists. Gripper
configuration is frozen.

Recording from the **head camera** (`AgibotEmbodiment.HEAD_BODY_NAME` + view constants with the
Kit viewport's pinhole, focal 18.147 / aperture 20.955) is the standard for these videos;
hand-placed cameras cost a run each time they were wrong.

## Recording mode

Drivers take `--record-dir --num-demos --seed --dataset-name`. In recording mode the arms are
driven through `env.step` (`EnvActionExecutor`) so Isaac Lab's recorder hooks fire; the success
termination is removed and demos are exported manually (`record_pre_reset` / `set_success` /
`export`), only when the task's own predicate accepts the final state. Recording is states-only
(camera observation recorder terms set to None); images are re-rendered afterwards with
`rerender_demo_cameras.py --hdf5 <file> --env <env> --demo-range a b` into a `<hdf5>.cameras/`
sidecar of mp4s (writing images into the HDF5 needs ~330 GB for 200 demos). The rerender replays
states kinematically with `scene.reset_to` + `sim.step(render=True)` per frame (`render()` alone
leaves rigid bodies frozen).

## Collection operations

Reference driver: `/home/ubuntu/playground/datasets/collect_agibot_arena_v0.sh` -- per task,
rounds of two workers until `TARGET_SUCCESS`, adaptive round size, merge readable files only, trim
to exactly N, re-render cameras in two halves, convert to LeRobot, refresh the upload directory.

- **Throughput** (L40S + 8 vCPU, 2 workers, 512^2 head cam, 15 Hz):
  `time ~= demos x per-demo sim time / (2 x success rate) + rounds x 5 min start-up`.
  stack_bowls: ~3.5 min per success, ~90 % -> ~6 h per 100. handover_toast: ~7 min per success,
  ~55-60 % -> ~12 h per 100. Two Isaac Sim processes is the ceiling on this box (8 vCPUs saturate;
  a third instance freezes the re-render workers). A 256^2 camera would cut 30-50 %.
- **Seeds**: base from the epoch, incremented per round. Two workers with the same seed produce
  byte-identical demos.
- **Interruptible**: killing the driver loses only the two HDF5s being written (merge skips
  unreadable files). `pkill -f <task>_cumotion` before restarting or four instances crush the CPU.
- **Health line**: stop and triage below 15 % success or on repeated worker crashes. Disk: 20-40 GB
  of raw HDF5 per 100 demos.
- **Per-task flags are settled**: handover none; stack_bowls `--jitter 0.04` (outer bowls +/-40
  mm, centre bowl +/-20 mm from the env default).
- Launch detached (`setsid nohup ... &`) with `PYTHONUNBUFFERED=1` (stdout is block-buffered under
  redirection and the planner init looks hung for minutes otherwise). Never `source` a driver to
  check it; `bash -n`.

## Datasets delivered

`agibot_arena_v0/{stack_bowls,handover_toast}`: 200 demos each, LeRobot v2.1, 15 fps, three views
(ego + two wrists), joint-space state/action (34-dof state in articulation order -- the arms
interleave left/right; 20-dof action in action-term order; `modality.json` slices 0-7 / 7-10 /
10-17 / 17-20). `observation.state` is relative to the reset pose (zeros at t=0), `action` is
absolute joint targets -- a user-accepted convention. `*_with_ee_pose` variants carry `eef_9d`
(xyz + first two rotation-matrix **columns**, base frame; GR00T's rot6d wants **rows** -- transpose
offline). The rotations in the colleague-generated variant were scrambled by the quaternion-order
bug and repaired by `fix_eef_9d_rotations.py`. Rounds 1-2 of the first 50 handover demos were
recorded under the pulsing-grip bug (constant grip since); noted, not re-collected.
