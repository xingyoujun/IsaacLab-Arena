# Asset selection and configuration for Agibot tasks

What an object has to look like for the Agibot to hold it, how to bring a new USD into Arena, and
the measurement that qualifies it. The short version: **size the object to the gripper, register
it with measured constants, run the pinch probe, and never tune its physics to make a task work.**

## The gripper envelope (measured on the spawned robot, 2026-09-04)

| quantity | value |
| --- | --- |
| jaw opening, fully open | 104.5 mm pad gap (commanded 0.994); 0.5 -> 57.8 mm, 0.25 -> 30.3 mm, fully closed -> 2.2 mm |
| pad collider | a 3 x 40 x 23 mm plate (thin along the jaw axis, 40 mm along the finger, 23 mm along the approach) |
| finger links behind the pads | 66-72 mm long, 23 mm wide, sweep **inward** as the gripper closes |
| tool frame (`gripper_center` / `right_gripper_center`) | 16.7 mm **past** the pads along the approach; closing is symmetric about it to 0.1 mm |
| approach axis / jaw axis (tool frame) | +z / y, both hands (measured with `probe_gripper_axes.py`) |
| pad region usable for a pinch | ~80 mm along the jaw-perpendicular axis, ~60 mm across the pinch |

Consequences, each measured with an ejection:

- **Longer than ~80 mm along the axis perpendicular to the jaws** and the finger linkage strikes
  the object before the pads arrive (a 120 mm billet was kicked out at a 95 -> 72 mm pad gap,
  5-7 m/s). The billet at 80 x 40 x 20 mm (scale 1.0) is held 3/3; at 1.2 it slips on a swing; at
  1.5 it is impossible.
- **Wider than the 104 mm opening** cannot be entered at all (bearing flange at 1.5 = 108 mm).
- **A wide base under a narrow neck** gets hit at the base (bearing flange 72 mm -> 2.4 m/s).
- **A grip band shorter than ~12 mm over a step** is only cradled: the pad's lower half lands on
  the step (bearing: a 9 mm land over a 48 mm hub, lifted 24 mm then slipped). Stepped or flanged
  parts need a tall grip land (>= 20 mm) or a different feature.
- **A bowl is grasped by its rim** (one finger inside, one outside), so the span never has to
  exceed the diameter; only the wall thickness must fit. Driving open fingers onto a bowl's centre
  jams the 110 mm bowl inside the 105 mm span and the convex shells interlock.
- **A flat cylinder pinched across its diameter aligns its axis with the approach**: a leaned
  grasp cannot hold it upright, and a peg has to square it up (sleeve_on_peg's near-horizontal
  grasp with the free rotation about the jaw axis is what makes seating work).
- **Thin slabs**: the pinch closes to 2.2 mm, so an 11.8 mm slice is fine; but the tool frame is
  behind the pads, and neighbours 25-33 mm apart are ploughed by a wide-open descent.

Good candidates: 40-80 mm across the pinch axis, 20-60 mm across the jaws, a flat or gently
curved grip face at least 20 mm tall, centre of mass near the grip, mass under ~0.5 kg. USDCraft
parts built to a prompt that says so come out right (the trays, the billet at 1.0, the sleeve).

## Conventions that have bitten us three times

- **Quaternions are (x, y, z, w)** everywhere in this Isaac Lab (`root_quat_w`, `body_quat_w`,
  `OffsetCfg.rot` for actions, frame transformers and cameras, `init_state.rot`). Identity is
  `(0, 0, 0, 1)`. A wxyz value in an xyzw field is a silent 180-degree flip (an upside-down object,
  a runaway left arm, a scrambled eef dataset). Build quaternions with `isaaclab.utils.math`, never
  by hand, and read the tilt back before trusting a write. cuMotion's own API is **wxyz**
  (`grasps.quat_wxyz_from_matrix`) -- convert at the boundary.
- **Arena's Agibot faces +x**; RoboDojo's x is lateral and y points at the robot. Swap when porting.
- **An origin is wherever the author put it.** Bread: bottom face. Bowl: geometry centre (30.1 mm
  up). USDCraft parts: centre of the underside. Factory gear: 50.75 mm off the part along x and
  5 mm up (the gear-base shaft frame). Measure with `probe_drop_settle.py` before placing by origin
  or aiming a grasp at it; aim grasps at the bounding-box centre.
- **The surface height is measured, not inferred.** The RoboDojo table's top is `TABLE_TOP_Z`
  = 0.6232 (a dropped block's bottom comes to rest there). Never derive a surface height from
  another environment's object placement (the mug's 0.75 in `tabletop_place_upright` is the mug's
  origin height, not the table's), and never trust a prim bbox for it.

## Bringing a USD into Arena

1. **Look at what you have.** `agibot_assets_v0/rec_*/isaac/model.usdc` is a flattened PhysX
   runtime entry per record (README beside it lists parts, joints, collider counts). Multi-part
   records (peg + sleeve) hold sibling rigid bodies under `/Asset/Links`; two-link records
   (bearing) are articulations.
2. **One body per `LibraryObject`.** Split multi-body assets with a `.usda` override layer that
   references the whole `/Asset` and sets `active = false` on the other links (a sub-prim reference
   drops the `/Asset/Looks` and `/Asset/PhysicsMaterials` bindings). Keep the layers beside the
   source under `/home/ubuntu/playground/objects/arena_local/<asset>/`, with the authoring script.
3. **Single-link articulations load as RIGID.** Factory pegs and gears carry
   `PhysicsArticulationRootAPI` with zero joints; on the CPU device (Kit teleop) that crashes
   `Articulation._create_buffers`. `delete apiSchemas = ["PhysicsArticulationRootAPI"]` in the layer.
   Real articulations (bearing with a spinning race) are `ObjectType.ARTICULATION` and are fine on
   both devices.
4. **Register** in `isaaclab_arena/assets/local_objects.py`: `name`, `tags`, `usd_path` under the
   env-overridable directory constant, `object_type`, and measured constants with attribute
   docstrings (half extents, rim height, origin offset, wall margin). Fixtures (platform, trays,
   bin) take `spawn_cfg_addon = {"rigid_props": RigidBodyPropertiesCfg(kinematic_enabled=True)}`.
5. **Mass**: `MassPropertiesCfg` in a `spawn_cfg_addon` only modifies an existing
   `UsdPhysics.MassAPI`; without it the change is a silent no-op (a warning in the log). Add the
   API in an override layer (`bread_massed.usda`, `t_block_fixed.usda`) and verify with
   `root_physx_view.get_masses()` at run time. USDCraft exports carry the API.
6. **Run the intake probes**: `probe_drop_settle.py` (origin, rest, noise), then `probe_pinch.py`
   at the intended scale, then `probe_reach.py` at the intended spot.

## Physics you may and may not change

Governing principle (from porting RoboDojo tasks): **only set what the source sets**. RoboDojo's
rigid loader sets mass and friction from `metadata.json` and never touches the collider; its
articulation loader sets neither. USDCraft's authored hinge friction, colliders and materials are
tuned. So:

| may change | must not change (each was tried and regressed) |
| --- | --- |
| mass / friction when the source metadata says so (bowl 0.32 kg, table friction 0.8) | collider type or parameters (`minThickness`, `maxConvexHulls`, SDF vs convex) to fix a grasp |
| kinematic flag on fixtures | hinge `jointFriction` / damping on generated articulations (raising it made a door ungrippable) |
| uniform **scale**, after re-running the pinch probe | mass to "make it stick" (lighter flung harder) |
| an override layer that re-centres geometry or adds `MassAPI` | anything on the robot |

Two documented exceptions where the collider *was* the problem and the decision is recorded on the
asset: the sleeve's convex decomposition had an effective bore of 9 mm against a 12 mm peg (no
insertion possible), so `sleeve.usda` is SDF (inserts; pinchable only with the ramped close), with
the original convex and a 24-wedge ring kept beside it; and the Factory gear is re-centred. Both are
geometry facts, not tuning.

## Scale rules

- Pick the scale from the gripper envelope, then check looks -- not the other way round.
- Containers scale independently of parts (`container_scale`), but predicates that use container
  footprints must scale their constants by `container.scale[0]`.
- After any scale change, re-run `probe_pinch.py` (parts), re-check that parts fit inside containers
  (a 254 mm wrench does not lie flat inside a 0.7-scale tray and rattles on the rim forever), and
  re-run `probe_reach.py` for the release points.

## Local asset inventory

| name | source | role | notes |
| --- | --- | --- | --- |
| `robodojo_simple_room`, `robodojo_table` | RoboDojo room; authored slab (1.1 x 1.4 x 0.05, Mahogany MDL, friction 0.8) | stage | table long axis on y |
| `bowl` | RoboDojo | stack_bowls | 110 mm, centre origin; convex colliders exact |
| `bread`, `bread_shelf`, `toaster` | RoboDojo | make_toast / handover | bread bottom-origin, 11.8 mm slab; toaster lever inverted vs RoboDojo threshold |
| `peg_platform`, `peg_sleeve` | USDCraft `agibot_assets_v0`, split layers | sleeve_on_peg | sleeve SDF; platform kinematic |
| `gray_tray`, `green_tray`, `blue_bin` | USDCraft `agibot_assets_v0` | tidy_workbench fixtures | kinematic; interiors real (drop test) |
| `wrench`, `metal_billet` | USDCraft `agibot_assets_v0` | tidy_workbench parts | steel friction only 0.25/0.18 (authored) |
| `bearing_assembly` | USDCraft `agibot_assets_v0` | tidy_workbench (not graspable) | kept selectable; replaced by the gear |
| `small_gear_centred` | Arena Factory gear, re-centred layer | tidy_workbench finished part | scale 2.0 = 44 x 50 mm |
| `t_block*`, `t_pad`, `headset*`, `laptop*`, `spring_button*`, `toaster_oven_*` | earlier ports (push_T, press_button, store_laptop, kitchen) | **not used on main** | on branch `chuanruiz/feature/robodojo-tasks`; archive candidates |

All `usd_path`s resolve under `ARENA_LOCAL_ASSET_DIR` (default
`/home/ubuntu/playground/objects/arena_local`) and `ARENA_AGIBOT_ASSETS_V0_DIR` -- host-specific;
another machine sets the two environment variables.
