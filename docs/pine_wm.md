# pine_wm calibrated workcell

`pine_wm` is a task-free Arena scene imported from the user's
`isim_scene_20260925.tar.gz` (Isaac Sim 6.0.1). It is separate from the original
`ur7e_workcell` and RR task environments. Add task objects and a task to this
scene for new data-generation workflows.

## Components and coordinates

- Embodiment: `pine_wm_ur7e`, collected UR7e + Robotiq 2F-85 with the source's
  -90 degree gripper mounting orientation, plus a 13 mm gripper spacer.
- Robot root: world `(0, -0.425, 0.75)` m; calibration pose `pose_a`.
- Slotted aluminium table: 1 x 1 m, top at 0.74 m, four legs and static collision.
  The 25 profiles use 40 mm pitch, 4.5 mm open grooves and 7 mm recess depth.
- Black mounting plate: 0.20 x 0.20 x 0.01 m between table and robot base.
- D405 housings, the spacer cylinder and photo-derived black printed camera
  brackets now have physical collisions (conservative convex hulls for printed parts),
  added during first20 task qualification. An annular collar, windowed
  support arms, rear mounting plates and fasteners connect both cameras to tool0.
  Bracket dimensions are approximate from the user photo, not measured CAD;
  camera cables are not modeled. Optical calibration and spacer physics are unchanged.
- Gripper appearance uses the same black material override as RR, including the
  directly bound inner fingertips.
- Following the user's reference photo, the table uses silver metallic material
  and the lighting uses a brighter neutral dome with broad, soft key/fill lights.
  These are appearance adjustments, not a photometric calibration.

The embodiment uses seven **absolute** joint targets: six UR arm joints followed
by `finger_joint` in radians. Gripper commands are continuous. Do not send a
zero-action policy to hold this embodiment; send its default joint positions.
The scene keeps the Arena UR7e gains and reset/recording infrastructure.

The new gripper clocking and spacer change tool kinematics. Old RR URDF FK,
cuMotion collision models, TCP conversions and trained policies must be adapted
and checked before collecting task trajectories here. Pine WM has its own
cuMotion description and first20 task driver. RR EEF dataset conversions and trained
policies are not automatically qualified on this hardware.

## Cameras

| Observation key | Resolution | Attachment |
| --- | --- | --- |
| `realsense_d435_rgb` | 640 x 480 | Fixed relative to the environment |
| `wrist_a_rgb` | 640 x 480 | Robot `wrist_3_link/flange/tool0` |
| `wrist_b_rgb` | 640 x 480 | Robot `wrist_3_link/flange/tool0` |
| `scene_cam_rgb` | 1280 x 720 | Diagnostic third-person overview |

These are in the normal Arena `camera_obs` observation group. Enable cameras
both in `PineWmEnvironmentCfg` and in `AppLauncher`. Arena camera recording can
consume the same camera rig.

The September 23 calibration JSON is preserved under
`isaaclab_arena/embodiments/ur7e/calibration/pine_wm/`. `T_tool0_cam` maps camera
optical coordinates into tool0; `T_base_cam` maps optical coordinates into the
robot root. This is active target-source notation, regardless of the shorthand
arrows in the source JSON comments. Camera offsets use ROS/OpenCV optical axes
(+X right, +Y down, +Z forward) and xyzw quaternions.

Simulation uses `T_tool0_cam`, not `T_tool0_cam_real_robot`. Projection follows
`scene.py`, including the negative horizontal principal-point offset. D405
renders are pinhole, without real lens distortion. The source wrist projection
uses `fx` for both effective focal lengths (the JSON `fy` differs by less than
one pixel); this behavior is retained, not silently recalibrated. Undistort real
D405 frames before comparing them. The JSON's original calibration accuracy
figures are source measurements, not a new calibration validation in Arena.

## External assets and reproduction

Download the pinned private HF release using [asset management](../tools/usdcraft_scene/README.md).
Set `ARENA_USDCRAFT_SCENE_ROOT` to the downloaded USDCraft-Scene root.
Robot payloads live in `embodiments/pine_wm/`, scenes in `scenes/pine_wm/`,
and shared objects in `assets/`. Stable manifest IDs resolve relative paths.
Explicit `ARENA_PINE_WM_ASSET_ROOT` legacy extracted roots remain supported for old recordings,
but there is no hardcoded source-host asset default and no UR5e fallback.
Models remain outside Git; embodiment code, calibration, mounts and planner descriptions belong in Git.

Run from the source worktree, using its existing shared native environment:

```bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH="$PWD:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab"
.venv/bin/python tools/pine_wm/render_scene.py \
  --output-dir /home/ubuntu/playground/experiments/pine_wm/20260925
```

The script builds `pine_wm` through `EnvironmentRegistry` and `ArenaEnvBuilder`,
holds the calibrated pose, rotates wrist joint 3 to check both wrist cameras
follow it while fixed cameras stay stationary, resets, and saves four PNGs,
`all_views.png`, and `validation.json`. It uses independent sensor render
products, not the unavailable headless viewport capture.

Validated on 2026-09-25, Isaac Sim 6.0.1 / L40S: all four RGB images were saved;
maximum seven-joint holding error was 0.0000813 rad. A 0.15 rad wrist command
moved the two wrist camera origins by 10.61 / 10.22 mm, while the D435 and
overview camera remained fixed. Camera position reset error was zero at the
reported tensor precision. The changed files passed host pre-commit hooks.
Artifacts are in `/home/ubuntu/playground/experiments/pine_wm/20260925/`.

For task composition, use `PineWmEnvironment().build(PineWmEnvironmentCfg(...))`
and extend the returned scene/task, or reuse `build_pine_wm_table`,
`build_pine_wm_lights` and `PineWmUr7eEmbodiment` in another registered factory.

The photo-based appearance update is rendered separately in
`/home/ubuntu/playground/experiments/pine_wm/20260925_visual_update/`, including
`gripper_detail.png` for inspecting the camera supports. The original import
preview remains in `20260925/`.

## Project boundary and planning description

Pine WM and RR real2sim are separate task families; see [task ownership](task_families.md).
The initial RR-drawer-on-Pine-WM experiment is retired. Its cross-family environment
and the Pine WM branch in the RR collection driver have been removed; external
historical recordings are retained but are not current runnable task definitions.
Pine WM drawer tasks T041–T045 use `P20_drawer` (`assets/drawer_first20/`),
not RR's `drawer_rr` or `usdcraft_drawer_arena.usda`.

`pine_wm_ur7e.urdf` and `.yaml` are Pine WM's separate planning descriptions:
the gripper base lies 13 mm beyond tool0, with camera and mount collision spheres.
The 162.8 mm grasp offset is relative to the gripper base. Shared UR7e control,
black gripper appearance and planner infrastructure are code reuse, not copied RR tasks.

## Towel fold

A USDCraft surface-deformable towel and its cuMotion diagonal-fold collection are described in
[pine_wm_towel_fold.md](pine_wm_towel_fold.md).
The USDCraft cable wrapped around a post is in [pine_wm_cable_wrap.md](pine_wm_cable_wrap.md).

## First 20 supplied PhysX tasks

The September 25 package is integrated as `pine_wm_first20`; see
[pine_wm_first20.md](pine_wm_first20.md) for layout sampling, collision audits,
cuMotion collection, recorded task conditions, camera replay and qualification criteria.
Development successes are tracked separately from same-version stability qualification.
