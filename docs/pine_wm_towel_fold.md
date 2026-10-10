# Pine WM towel fold (USDCraft surface deformable)

`pine_wm_fold_towel` adds one free PhysX surface-deformable towel to the `pine_wm` workcell.
`data_engine/pine_wm/collection/fold_towel.py` collects a diagonal fold with cuMotion: pinch the
corner nearest the robot, lift, carry over the fold diagonal, lay it down short of the far corner,
release and return to the ready configuration. Nothing attaches the towel to the gripper; it is held
by Robotiq pad friction only.

## Asset

The towel is a USDCraft Generation record (2026-10-08):
`rec_generate-one-square-terry-cotton-hand-towel-appr_20261007_174054_3942d9f7_f4d33b99`.
It is 0.30 x 0.30 m, 2.5 mm, 550 g/m², 1927 nodes. All material values are unmeasured assumptions
from the generator. Its flattened PhysX export (`isaac/model.usdc`, SHA256 `dbeb05d1…`) is copied
to the gitignored `local_assets/usdcraft_towels/terry_hand_towel/` and selected with
`ARENA_PINE_WM_TOWEL_USD` or `PineWmFoldTowelEnvironmentCfg.towel_usd`. It is not yet in the
USDCraft-Scene HF release.

USDCraft's training-interaction sidecar ignores the towel's corner `pick` annotations, because
the part is not a rigid body. The driver picks corners from the settled towel nodes instead.

The USD already carries its deformable body, material and collision schemas. It is therefore
spawned with `DeformableObject(..., physics_backend=PhysicsBackend.PHYSX)` and no
`deformable_props`. Passing `deformable_props` would define a second, volume, deformable on the
asset root. PhysX's surface view reports `check()==false`; USDCraft records the same diagnostic
for its own surface fixtures. Nodal state reads, writes and resets work.

## Run

```bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export ARENA_PINE_WM_TOWEL_USD=$PWD/local_assets/usdcraft_towels/terry_hand_towel/isaac/model.usdc
.venv/bin/python -m data_engine.pine_wm.collection.fold_towel --output outputs/towel_fold/NEW_DIR \
    --enable_cameras --kit_args="--/rtx/verifyDriverVersion/enabled=false"
```

`--kit_args` is only needed on hosts whose NVIDIA driver is older than RTX accepts. Without it,
on driver 535 the camera reads crash with an illegal memory access. Outputs are `results.json`,
`annotations.json`, `demos.hdf5` (or `demos_failed.hdf5`) and four post-step preview videos under
`live_preview/`. The HDF5 file holds actions, robot states, per-step towel nodal positions and
velocities, skill/stage labels, and a `towel` group with the rest nodes, the corner node ids and
the USD hash.

## Success criteria

Success is judged on the settled towel nodes, 1.5 s after the arm has returned:
- **Mirror error.** Each node of the folded half is paired with its mirror image across the fold
  diagonal. The mean horizontal distance between the pairs must be at most 40 mm; an unfolded
  towel scores about 158 mm.
- **Footprint.** The convex-hull area must shrink to at most 0.7 of the rest area; an ideal
  diagonal fold gives 0.5.
- **Height and table.** The towel may stand at most 40 mm above the table and must stay on it.
- **Lift.** The pinched patch must rise at least 50 mm, or the run stops as `towel_not_lifted`.
- **Contacts.** Robot bodies other than the gripper must not touch anything (net force > 1 N).

The corner-node gap is reported but not used: a single corner node can curl back while the fold
itself is good.

## Measured (2026-10-08, RTX 3090, PhysX dt 1/120 s, one run each)

Defaults are `grasp_inset_m=0.045`, `pinch_clearance_m=-0.002` and `place_short_m=0.11`. The
Robotiq fingertips move 11.6 mm further along the approach axis when closing. Descending with
the fingers open therefore leaves the tips about 12 mm above the table, and they sweep the towel
up as they close.

| Run | Towel pose | Mirror error | Footprint | Result |
| --- | --- | --- | --- | --- |
| `demo_02` (cameras) | default | 9.1 mm | 0.504 | success |
| `robust_shift` | +80 mm x, +50 mm y | 8.3 mm | 0.508 | success |
| `robust_yaw-15` | yaw -15° | 14.1 mm | 0.504 | success |
| `robust_yaw+15` | yaw +15° | 47.1 mm | 0.586 | failed (flap overshoot) |

Without the 2 mm fingertip press, or with a 20–30 mm inset, the corner slipped out during the lift.
PhysX deformables are not deterministic here. Repeated runs with identical inputs differed by up
to 4 mm of mirror error. In one pair, the corner-node gap differed by 23 mm vs 92 mm. These are single development runs, not a qualified success rate;
the towel pose is not randomized by the environment.
