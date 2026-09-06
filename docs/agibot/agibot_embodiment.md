# The Agibot embodiment: frozen default configuration

One default Agibot configuration serves every task. It lives in
`isaaclab_arena/embodiments/agibot/agibot.py` (`AGIBOT_ARENA_A2D_CFG` and the config classes) and
in the standard control stack installed by `install_agibot_control_stack`. Per-task overrides are
the rare exception and are limited to the knobs exposed on `AgibotTabletopEnvironmentCfg`. This
page records what the default is, why each part is what it is, and what was refuted on the way --
so that no future task re-derives it.

## What the default is

| part | value | origin |
| --- | --- | --- |
| rest pose | shipped `AGIBOT_A2D_CFG.init_state`, with the **left wrist mirrored onto the right's** (`left_arm_joint6` 0.7, `joint7` 0.0 instead of +1.4725 / -0.1599) | the shipped pose is asymmetric only at the wrists; the mirror puts both hands 160 mm into symmetry and inside the head view. Stated in three places that must agree: `init_state`, each arm yaml's `default_q`, the other arm yaml's `cspace_to_urdf_rules` -- hence Arena carries patched copies of both Lula yamls in `embodiments/agibot/rmpflow/` |
| grippers | left gripper = a copy of the right: effort 100/100 (main/support), velocity 10, passive effort 100 (shipped left: 10/1, 2, 10) | the shipped asset is asymmetric and only the right's numbers behave; velocity 2 cannot grasp at all (0/15), effort >= 30 is behaviourally identical to 100 |
| arm actuators | shipped (stiffness 2e4-1e7, damping 0, effort 1000-2000) | left and right are already identical |
| arm effort limit | **shipped 1000-2000 N m** -- the task-level 300 N m ceiling was removed 2026-09-06 (user decision: every path runs the embodiment defaults, `arm_effort_limit` defaults to None and exists for measurement only) | stops torque saturation when the stiff arm meets the table (2000 -> 300 with identical tracking); 100 makes the arm collapse; not blessed as an embodiment default by the user, so it stays a task field |
| left-arm frame offsets | `body_offset.rot` and the FrameTransformer offset are `(0, -0.7071, 0, 0.7071)` (xyzw) | the URDF gives the left tool frame an extra quarter turn; the value was shipped in wxyz and read as a 180-degree flip |
| reset event | `reset_joint_position_and_velocity_to_defaults` on the **robot only** | the shipped embodiment had no reset event, so every joint reset to 0; a scene-wide reset instead tramples object placement on the post-success auto-reset |
| dual-arm mode | `ArmMode.DUAL_ARM` with `AgibotDualArmSceneCfg` / `AgibotDualArmActionsCfg` (layout `[left pose 6, left gripper 1, right pose 6, right gripper 1]`) | Arena shipped single-arm only |
| recording actions | `AgibotDualArmJointActionsCfg`: arms first-order-hold (`SmoothJointPositionAction`), grippers plain zero-order hold | a zero-order arm hold at 15 Hz jolts a pinched slab loose; a smoothed gripper term pulses the grip at 15 Hz and walks a bowl out of the hand |
| cameras | head cam true-ego (base frame pos (0.56, 0, 1.30), focal 12, looking at (1.103, 0, 0.633)), two D405 wrist cameras world-anchored and re-posed every frame | a camera prim under a moving link does not track it (runtime prims read USD, not Fabric) |
| head viewer | `HEAD_VIEW_EYE` (0, 0, 0.42), `HEAD_VIEW_LOOKAT` (0.66, 0, -0.63), world-axis offsets from the head | viewport FOV is fixed at +/-30 degrees; the standoff must go straight up |

## The standard control stack (teleop path)

Installed, in this order, by `install_agibot_control_stack(env_cfg, cfg, surface_z)`:

1. `install_arm_target_hold` -- an RMPFlow arm commanded exactly zero holds its previous target.
   Without it a dual-arm idle arm walks 88 mm off its reset pose in 0.7 s (relative mode rebuilds
   the target as current pose + delta; a first-step jump is latched in).
2. `install_surface_guard(surface_z)` -- clamps the commanded descent so the lowest gripper body
   stays 18 mm above the surface. Pressing the stiff arm into the table and closing gives 1.0 m/s
   finger speeds; the clamp brings every descent condition back to the free-close baseline.
   Rotation and lateral motion keep full authority; it is not a containment volume.
3. `install_ramped_gripper(ramp_seconds)` -- the binary gripper target ramps open -> closed over
   1.67 s (the executor's ramp), then holds. The stock action steps the target and drives the
   fingers in at ~0.9 m/s (an SDF sleeve was spat out 5/5; ramped, held 5/5 at 0.35 m/s).
4. `apply_arm_gains` -- the effort limit above, on `env_cfg.scene.robot.actuators` (patching PhysX
   at run time leaves Isaac Lab's `ImplicitActuator` with stale gains and `applied_torque` lying).

Every installer skips joint-space terms, so the recording path (joint actions + executor) is
unaffected. `record_demos.py`, teleop, replay and closed-loop eval all build through the same
callback, so they all get the stack.

## Gripper anatomy versus Franka and RoboDojo's X5 (measured 2026-09-06)

| | Agibot A2D hand | Franka Panda hand | RoboDojo ARX X5 |
| --- | --- | --- | --- |
| mechanism | four-bar linkage per finger: 11 links, 1 driven joint (`hand_joint1`), 1 mimic joint (`Right_1_Joint`), 2 loop-closing joints (`*_2_Joint`, `excludeFromArticulation`, maximal-coordinate), passive `*_0_Joint` (+/-10 deg) and `*_RevoluteJoint` (free) | 2 prismatic fingers | 2 prismatic fingers |
| open span / pad | 105 mm; pad 3 x 40 x 23 mm flat plate | 80 mm; finger 21 x 26 x 54 | 88 mm; finger 86 x 37 x 61 |
| pad motion while closing | pads stay parallel (<= 0.1 deg) but **advance 18 mm along the finger axis** (hand-frame z 189 -> 207 mm): a top-down pinch plunges 18 mm deeper as it closes | pure lateral | pure lateral |
| loop closure under a pinch | anchor gap <= 0.3 mm: effectively rigid | n/a | n/a |
| open width of the finger structure | support arms at x = +/-75 mm, 26 mm thick -> ~175 mm overall; a hand centred over a 106 mm bowl mouth lands the arms on the rim | 80 mm | 88 mm |
| colliders | every link `convexHull`; pads exact (fill 1.00), finger links 74-88 % fill, housing 62 % | `convexHull`; fingers 60 % fill, hand 90 % | `convexHull`; fingers 85 %, link6 62 % |
| visual geometry without any collider | 14 meshes per hand (D405 wrist camera, adapter shell, flange, screws): x[-44, 41] y[-125, 43] z[-41, 100] mm in the hand frame, i.e. up to 85 mm outside the 80 x 80 x 131 mm housing collider, all >= 90 mm above the pads | none | none |
| left/right authoring | USD differs in `hand_joint1` maxForce (1 vs 2), a 1e-4 stiffness on `left_Left_Support_Joint`, missing drive attrs on `left_Right_RevoluteJoint`; all overridden by Arena's actuator groups, kinematics identical | -- | -- |

What this means for a bowl:

- A rim pinch is collider-accurate: closing on a bowl held in place blocks at a 6.0 mm pad-origin
  gap, i.e. ~3 mm between the pad faces on a 3.7 mm wall. (The 14 mm figure quoted earlier came
  from a bowl that was falling while pinned.)
- A flat 40 mm pad against a 50 mm-radius inner wall touches at its centre while its ends sit
  ~4 mm into the visual wall (chord vs arc); the bowl's 16-vertex wall hulls are near-flat facets,
  so nothing stops it. That is what an "inner pad face stuck in the inner wall" looks like; the
  Franka and X5 fingers are narrower tangentially and show ~2 mm.
- The 18 mm plunge during closing and the 175 mm open finger structure are the Agibot-specific
  kinematic traits: closing over a bowl drives the pads deeper, and descending centred lands the
  support arms on the rim (measured: the bowl tilts 18 deg before the pads touch anything).
- The collider-less camera and shell only matter when the wrist itself comes within ~90 mm of an
  object (tilted wrist, tall objects, the robot's own body), not in a tabletop pinch.

## Rate

Control runs at Arena's default **15 Hz** (sim dt 1/120, decimation 8). That is a training-side
decision (DROID is 15 Hz; pi0.5 and cosmos measured better matched). `set_control_rate_50hz` is a
debugging escape hatch only; the one genuine 15 Hz effect (a deterministic 1-in-5 close fling in a
scripted rim grasp at effort >= 30) is narrow and did not reproduce after the default-config merge.

## Known costs, accepted

- In `tabletop_place_upright` (single left arm, seed 42) the mirrored wrist gives 2/10 scripted
  descend stalls and raises idle drift 25 -> 58 mm (no target hold in that env). User decision:
  accept, disclose in the PR.
- Each arm's cuMotion collision model holds the *other* arm at its rest pose; a receiving arm cannot
  see where the giving arm actually is.

## Upstream

Three Arena defects (no reset event; wxyz offsets; the `env_cfg_callback` doc example returning
nothing) are fixed in the team baseline and written up in
[upstream_bug_report_2026-08.md](upstream_bug_report_2026-08.md); the bug-fix branch
`chuanruiz/fix/agibot_left_arm_and_env_cfg_callback` awaits the user's push. Two root causes are
Isaac Lab asset-layer (gripper asymmetry, wrist asymmetry + yaml `default_q`) and should be
reported there with the cross-matrix as evidence. The left tool-frame relabelling in the generated
cuMotion description is ours, not Arena's.

## What was refuted (do not re-derive)

The full table is in [triage.md](triage.md). In one line: weakening, slowing or damping the
gripper throws objects harder; softening the arm makes it collapse or oscillate;
`ignore_robot_state_updates=False` diverges; raising clearance or easing descents regresses; and
the robot's rest pose was never the cause of the left-arm runaway. Gripper and arm configuration
are frozen (user directive, 2026-08-26).
