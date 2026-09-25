# Non-USDcraft toaster cuMotion qualification

Source-host run: 2026-09-11. These three assets execute the existing USDcraft
task: close gripper, approach the paddle with a tilted tool, press vertically,
stop with the lever down (no retreat in recorded demos). The shared robot spawn
uses the GPU-verified `black_fingertips_direct` appearance.

## Matched settings and asset-specific measurements

Same robot, table, calibrated D435, original closed-drawer pedestal (+78 mm),
x [-0.22, -0.10] m, y [0.10, 0.14] m, yaw 0 +/- 10 degrees, seed 0, 0.03 rad
initial robot-joint jitter, and 75% normalized lever-travel success threshold.
Gravity is disabled on toaster bodies as in USDcraft. Passive joint velocity
is cleared at every driver reset to prevent cross-episode drift. Press depth
defaults to 45/47 of each joint's travel, retaining the original 45 mm command
on the 47 mm USDcraft asset. An explicit `--press-depth` still overrides it.
No baseline-specific grasp/tilt tuning or relaxed success threshold.

| Method | Width (m) | Joint limits (m) | Success travel (m) | Lever body / paddle offset (m) |
| --- | --- | --- | --- | --- |
| miniworkflow GPTSOL | 0.1434 | [-0.057, 0.004] | >0.04575 from upper rest | `Lever`, (0, 0, 0) |
| miniworkflow Astra | 0.1434 | [-0.061, 0] | >0.04575 | `Lift`, (0, -0.014, -0.0005) |
| Articraft | 0.143194 | [0, 0.04773133] | >0.0357985 | `carriage_lever`, source paddle center times 0.795522222222 |

The GPTSOL rest position is +4 mm (upper limit), not zero. Paddle tracking uses
the moving body's rotation and translation; no world-position hardcoding.
The paddle top thickness and planning bounds are measured per asset.

## Minimum asset changes

Both raw miniworkflow assets failed fixed-base spawning: articulation root on
plain `/Toaster` lacks a rigid-body API. Their `_arena.usda` overlays only move
that root API to `/Toaster/Body`. Geometry, dimensions, materials, joint limits
and drive parameters are unchanged. With the overlays, both passive probes
hold 80% travel (48.8 mm) and return to rest. A probe ends with a false task
predicate by design; it is not a robot press demonstration.

Astra additionally failed actual pressing because PhysX could not cook the
`Lift/PaddleCaps` convex mesh (also `Body/Collision/Shell_24`, `Shell_25`,
`Shell_47`). The visible paddle had no effective collision surface. The final
Astra overlay triangulates convex hulls from the exact existing vertex clouds
for the three shell collision meshes, and adds an invisible triangulated
collision copy of PaddleCaps while disabling collision on the original visible
caps. No points are moved, no visual topology/material is replaced, and no joint
or drive is changed. The initial root-only run is a retained failure diagnostic,
not a usable training source.

Articraft is exported from the complete `articraft_toast/articraft_toast.urdf`.
Its source housing width is 0.18 m. Uniform factor 0.143194/0.18 = 0.795522222222
matches USDcraft's measured mesh width. Mesh dimensions, origins and prismatic
lengths scale together; mass/inertia scale at constant density (s^3/s^5).
Source URDF and OBJ files remain unchanged. Joint damping comes from the URDF;
no motor stiffness is added. Collision conversion uses convex decomposition,
and the importer creates a fixed world joint. Output is flattened standalone USD.
Do not apply the scale again. Reproducer (fresh output directory):

```bash
.venv/bin/python tools/rr_sim2real/convert_articraft_toast.py \
  --source /home/ubuntu/playground/rr_ur/articraft_toast/articraft_toast.urdf \
  --output-dir /absolute/fresh/articraft_toast --device cuda:0
```

Source hashes:

```text
b34189bb47824f086ce73d256ada62b2a3b629936af6ab67bc4f5db16447efc5 miniworkflow_gptsol_toast.usd
3e488db18af5ca24ae16dd907f56d9990461c30a4055e0a889c574a87b5308d3 miniworkflow_astra_toast.usd
c493063c2b25f2503e7548fa490f2828f265070001d2349be865e83207b3f671 articraft_toast/articraft_toast.usd
```

## Commands and outputs

Set the native runtime exports in [the handoff README](rr_sim2real/README.md).
Replace `METHOD` below with `miniworkflow_gptsol`, `miniworkflow_astra`, or
`articraft`; use fresh output paths for another run.

```bash
.venv/bin/python isaaclab_arena_cumotion/scripts/ur7e_press_toaster_cumotion.py \
  --env ur7e_METHOD_press_toaster --device cuda:0 --num-demos 3 --seed 0 --fps 15 \
  --video /home/ubuntu/playground/rr_ur/sim_renders/METHOD_toast_cumotion_black.mp4 \
  --record-dir /home/ubuntu/playground/datasets/rr_sim2real_aux/toast_qualification/METHOD \
  --dataset-name METHOD_press_toaster
```

There are D435 and `_scene.mp4` clips. Only successful state/action trajectories
are exported to HDF5; video/logs include failures. Preview appearance is the
default workcell, not randomized backgrounds/distractors (those are a later
state-replay rendering stage). Nothing is published to the HF dataset root and
no 200-demo collection or DP training is implied by qualification.

## Verified outcomes

| Method | Successful presses | Final travel (mm) | Recorded frames |
| --- | --- | --- | --- |
| miniworkflow GPTSOL | 3/3 | 57.500 / 57.500 / 57.500 | 264 + 279 + 254 = 797 |
| Articraft (uniformly scaled) | 3/3 | 47.732 / 47.732 / 47.732 | 262 + 278 + 261 = 801 |
| miniworkflow Astra (collision repaired) | 3/3 | 61.000 / 61.000 / 61.000 | 275 + 294 + 262 = 831 |

All nine HDF5 trajectories start at the correct rest position with negligible
slide velocity and have zero recorded housing translation drift. Logs show
progressive paddle descent under contact; the D435 previews confirm black
fingertips pressing the actual paddle. Terminal travel can exceed the joint
limit by sub-micrometre solver tolerance. This small sample establishes task
qualification, not a large-scale success-rate estimate. No USDcraft retest was
requested or performed here. Python/format hooks pass; original miniworkflow
payload hashes remain unchanged.

Preview PNGs under `rr_ur/sim_renders/`:
`miniworkflow_gptsol_toast_black_preview.png`, `articraft_toast_black_preview.png`,
`miniworkflow_astra_toast_black_preview.png`. Durable `cumotion.log` files are
saved beside each corresponding HDF5.

For Astra the first `miniworkflow_astra` directory/unsuffixed video is the
root-only failure diagnostic (0/3, invalid paddle collision). The corrected run
uses `toast_qualification/miniworkflow_astra_collision_fixed/` and
`miniworkflow_astra_toast_cumotion_black_collision_fixed.mp4` (plus `_scene`).
Do not substitute the failed root-only recording for the corrected run.
