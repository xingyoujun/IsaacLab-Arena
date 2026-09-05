# RMPFlow vs joint-space control: what RoboDojo does, and what our collisions are made of

Research notes (2026-09-05), no code changed. Question from the team: why does teleoperation go
through RMPFlow at all instead of absolute control like the cuMotion path, is RMPFlow the reason
objects and the table get hit so hard, and how does RoboDojo drive its robot?

## 1. What the two control representations are

| | teleop today (Arena Agibot) | cuMotion today | RoboDojo (ARX X5) |
| --- | --- | --- | --- |
| command the runner produces | end-effector **delta** per control step (dx dy dz, axis-angle; 6 per arm) + binary gripper | absolute **joint** targets (7 + 3 per arm) | absolute joint targets, or an end-effector **pose** that is turned into joint targets by one cuRobo IK solve |
| how it becomes joint targets | RMPFlow: a reactive motion policy evaluated 5 times per physics substep, integrating joint accelerations to joint position + velocity targets | none: the planner already produced joint targets; a first-order hold walks the arm there over the control step | none / a single IK solve; then **joint-space linear interpolation** over 80 % of the control interval and a hold for the last 20 % |
| gripper | binary -> ramped joint target (1.67 s) | joint target, executor ramps 1.67 s | scalar opening interpolated per step and additionally **rate-limited to 20 % of range per physics step** |
| control / physics rate | 15 Hz / 120 Hz (8 substeps) | 15 Hz / 120 Hz | **25 Hz / 250 Hz** (10 substeps), obs and actions recorded at 25 Hz |
| arm actuator | stiffness 2e4-1e7, **damping 0**, effort 300 (ours) / 1000-2000 (stock) | same | stiffness **4400**, damping **40**, effort 100, armature 0.01, **gravity disabled on the robot**, 8 solver position iterations |
| contact model in the controller | RMPFlow knows its own links (self-collision spheres, torso cylinder); it knows nothing about the table or objects | cuMotion plans around the table and the objects it is told about | cuRobo plans around the table (planner mode); IK mode has no contact model either |

RoboDojo's shipped training data was produced by an internal `SyncCollectEnv` / `SkillManager`
pipeline that this eval-only release no longer contains; what remains (`robot_manager.plan_ee`,
`plan_endeffector_joint`, `CuroboPlanner.plan_path`) shows it was scripted cuRobo joint
trajectories executed as joint targets -- the same shape as our cuMotion path -- plus policy
evaluation that accepts joint or end-effector actions (`_robot_info.json`, `validate_action_dict`).
There is no reactive controller anywhere in RoboDojo.

## 2. Why teleoperation uses RMPFlow, and why it cannot simply be "absolute control"

A human on a keyboard, spacemouse or headset produces **end-effector motion**, not 7 joint angles.
Something has to solve the inverse kinematics of a redundant 7-DoF arm every step. The options in
this stack are:

- **RMPFlow** (Isaac's reactive motion policy, what Isaac Lab ships for the Agibot -- and the only
  controller Lab ships an Agibot config for). Benefits: solves IK for a redundant arm without
  tuning, resolves the redundancy consistently, respects joint limits, avoids self-collision (the
  torso cylinder and link spheres in the yaml), caps joint velocity, smooths the motion. That is
  why Arena's `tabletop_place_upright` and our ports use it.
- **Differential IK** (Isaac Lab `DifferentialInverseKinematicsAction`, what Arena uses for the
  Franka): one damped-least-squares step per control step from the *measured* pose, joint target
  held for the substeps. No self-collision model, redundancy drifts unless a null-space term is
  added, but it is simple, closed-loop on the real state, and its output is a plain joint target.
- **A single planner IK solve per step** (RoboDojo's `ee_pose` mode with cuRobo; our cuMotion has
  the equivalent). Collision-aware IK, then joint-space interpolation.

"Absolute control" is only possible once something produced joint targets. cuMotion has them
because it planned the whole motion offline; a teleoperator does not. So the real design choice
is *which* IK, and *how* its joint targets are applied -- not whether to have one.

## 3. Is RMPFlow the culprit for the violent contacts?

Partly. The measurements we have (all in `triage.md`) separate several mechanisms:

1. **The dominant, measured driver is the arm's PD configuration under a commanded press.** With
   the end-effector commanded 110 mm below the tabletop, finger speed at gripper close went from
   0.57 m/s (free) to 1.04 m/s, arm torque saturated at its ceiling. GR1T2 shows the same with 10x
   the torque; Franka (stiffness 80, damping 4) does not (0.20 m/s either way). Stiffness
   2e4-1e7 with damping 0 means a blocked arm turns position error straight into force with nothing
   to dissipate it. RoboDojo's X5 runs 4400 / 40 with gravity off; Franka 80 / 4.
2. **RMPFlow makes the press worse in one specific way: it is open-loop on the robot state.**
   Arena's Agibot config sets `ignore_robot_state_updates=True`, so after each reset the
   controller integrates its *own* copy of the joint state, five evaluations per substep, and
   never reads the measured joints back. When the arm is stopped by the table the internal state
   keeps travelling toward the commanded pose at up to the yaml's 3.14 rad/s, and the joint
   targets it emits run away from the real arm. (Turning the flag off was measured *worse*: the
   controller then re-plans from the blocked state every substep and its attractors still push,
   plus the idle drift explodes to 890 mm. Neither setting has a contact model.)
3. **RMPFlow's relative mode has no memory**: "zero delta" means "stay where you are *now*", so
   the first-step jump after reset latched into an 88 mm idle drift until we added the target
   hold. With joint targets there is no such drift; a held target is a held target.
4. **The gripper close was instant** in the stock binary action (fingers at ~0.9 m/s). That is an
   action-term problem, not RMPFlow's; fixed by the ramp. RoboDojo rate-limits the gripper to 20 %
   of its range per physics step for the same reason.
5. **Objects larger than the pad region** are ejected by the finger linkage regardless of
   controller (measured mid-air, no table, no RMPFlow). Asset problem.

So: the gripper flings we chased were mostly (4) and (5); the table press is (1) amplified by (2);
the idle drift is (3). RMPFlow is the amplifier of the table press and the source of the drift,
not the origin of every violent contact. The surface guard and target hold were built to contain
exactly (2) and (3) without touching the actuator.

## 4. What a RoboDojo-style teleop path would look like for us

The structural alternative is to make teleoperation produce **joint targets** like every other
path, and keep the Agibot on **one** action-term family:

```
operator delta (14)  -->  IK from the MEASURED end-effector pose, once per control step
                     -->  7 joint targets per arm, applied through SmoothJointPositionAction
                          (first-order hold over the 8 substeps, exactly what cuMotion recording uses)
gripper binary       -->  ramped 3-joint target (already the case)
```

Consequences to expect (hypotheses, to be measured before any code moves):

- **Contact**: a blocked arm's target advances by at most one operator delta per control step
  and is recomputed from the measured pose, so the press-through is bounded by the step size, not
  by a controller's runaway internal state. Combined with the surface guard this should remove the
  RMPFlow term (2) entirely.
- **Actuator gains**: RoboDojo-like soft gains (4400 / 40) were rejected for us because under
  RMPFlow teleop the soft arm coupled with the gripper and dropped a bowl mid-lift. That result was
  measured with RMPFlow in the loop and should be re-measured under joint-space control; if soft
  gains hold up there, the table contact problem shrinks at the source.
- **Data**: teleop demos would carry joint targets natively; the `joint_pos_target` relabel
  becomes a cross-check rather than the bridge.
- **Risks**: differential IK on a 7-DoF arm drifts in the redundant joint and has no self-collision
  or torso avoidance (RMPFlow's real strength); the left arm's tool-frame relabelling and
  `body_offset` have to be carried into whichever IK is used; the two arms are not mirror images.
  Using cuMotion's IK (collision-aware) per step is the alternative if differential IK proves
  too loose, at a latency cost to measure at 15 Hz.

## 5. The arm's PD, sized from measurement

The Agibot USD authors every arm drive at stiffness 1e7, damping 0, max force 1000 -- placeholder
"rigid" drives -- and Isaac Lab's `AGIBOT_A2D_CFG` keeps them (1e7 on joint 1, 2e4 on joints 2-7,
damping 0). Nobody has sized this robot's PD; the only thing we changed so far is the effort limit
(300 at task level). Measured in simulation on 2026-09-05 (PhysX generalized mass matrix and
gravity-compensation torques, rest pose and a mid-carry pose from a recorded demo; the authored
links weigh 2.6 kg per arm including the hand):

| joint | M_ii (kg m^2) | gravity torque (N m) | critical damping 2*sqrt(k*M) at k = 1000 / 4400 / 20000 |
| --- | --- | --- | --- |
| arm_joint1 (shoulder) | 0.21 | 4.4 | 29 / 61 / 130 |
| arm_joint2 | 0.19 | 2.7 | 27 / 57 / 122 |
| arm_joint3 | 0.13 | 2.9 | 22 / 47 / 101 |
| arm_joint4 (elbow) | 0.12 | 1.6 | 22 / 46 / 98 |
| arm_joint5 | 0.02 | 1.0 | 8 / 17 / 37 |
| arm_joint6 | 0.04 | 0.2 | 13 / 26 / 56 |
| arm_joint7 (wrist roll) | 0.0007 | 0.0 | 1.7 / 3.6 / 7.6 |

What this says:

- **The stock drive is an undamped spring.** At k = 2e4 and damping 0 the shoulder's natural
  frequency is sqrt(2e4 / 0.21) = 310 rad/s (about 50 Hz) with zero damping ratio: every contact
  stores and returns its impact energy. That is the physics behind the violent table and object
  contacts, independent of the controller in front of it.
- **Gravity is not the reason soft settings "collapsed".** The largest gravity torque anywhere in
  the arm is 4.4 N m. The earlier "effort 100 -> arm droops 332 mm" result cannot be gravity
  load (100 N m is 23x the load); it is what a 2e4-1e7 stiffness drive does when its force is
  clamped far below what that stiffness demands. With a sane stiffness, 50-100 N m is ample, and
  gravity sag under PD is g/k = 4.4/4400 = 1 mrad -- disabling gravity (RoboDojo, Galbot, GR1T2
  do) is optional here, not necessary.
- **RoboDojo's 4400 / 40 is a reasonable order of magnitude for this arm too**: 40 is close to
  critical at the shoulder and over-damped at the wrists. A per-joint damping at zeta ~ 1 (about
  60 / 55 / 47 / 46 / 17 / 26 / 4 for k = 4400) is the principled version; k = 1000 with about
  half those dampings is a softer alternative.
- **The PD cannot be chosen independently of the controller.** The soft-gain test that "dropped
  a bowl mid-lift" was run under RMPFlow, whose joint targets move every substep and carry
  velocity feed-forward; a soft arm lags them and oscillates. Under joint-space targets with a
  first-order hold (the cuMotion recording path, RoboDojo's interpolation) lag is benign. Gains
  and controller have to be tested as a matrix.

Where a change would land: `AGIBOT_ARENA_A2D_CFG.actuators["left_arm" / "right_arm"]` in
`agibot.py` (the embodiment default), replacing the current per-task `arm_effort_limit` knob. The
user has asked for this to be configured properly; it supersedes the earlier "frozen" decision
once the matrix below has been measured.

## 6. "IK every step" -- is that RoboDojo's way, and which IK?

Yes. RoboDojo's `ee_pose` action mode solves cuRobo IK once per 25 Hz action from the measured
joints, then linearly interpolates the joint targets over 8 of the 10 physics substeps and holds
for 2; its scripted data used cuRobo *planning* with the same joint-target execution. The
proposal is the same shape at our 15 Hz / 8 substeps. Three IK implementations exist in this stack:

| option | what it is | redundancy / limits | contacts | cost per step | notes |
| --- | --- | --- | --- | --- | --- |
| Isaac Lab `DifferentialInverseKinematicsAction` (dls / adaptive_dls) | one Jacobian step from the measured pose | joint-limit avoidance term only; the redundant joint drifts | none | negligible | Arena's Franka path; simplest; `body_offset` and the left tool-frame convention carry over |
| Isaac Lab **Pink IK** (`PinkInverseKinematicsActionCfg`) | QP over both arms: frame tasks + damping task + null-space posture task, joint limits as constraints, `lm_damping` for step jumps | posture task holds the elbows; limits enforced | none (no self-collision) | ~ms | what Isaac Lab's own teleop stack (isaacteleop / XR retargeters) targets, reference config in `pickplace_gr1t2_env_cfg.py`; needs a URDF consistent with the scene USD (Lab generates it from the USD) |
| cuMotion IK per step (our planner's `ik_reachable`) | collision-aware IK against the planner's world model | full | table and registered objects | ~25 ms per solve measured (576 solves in 15 s in the reach probe), two arms ~50 ms of a 66 ms budget | tight at 15 Hz, but it is the only one that knows about the table |

RMPFlow's one real advantage over all three is self-collision and torso avoidance from its sphere
model. For table-top bimanual work that is rarely exercised by an operator; Pink's posture task
covers most of it, and the surface guard stays as the table's representative in the control path.

Recommended candidate order: Pink IK first (it is the direction Isaac Lab itself is going for
teleop, handles both arms in one solve and gives joint targets the recording pipeline already
understands), differential IK as the cheap fallback, cuMotion IK if collision awareness in the
loop turns out to matter.

## 7. Proposed experiment plan (no changes yet)

A matrix, each cell measured with the existing probes on `agibot_stack_bowls` (table-press finger
speed and arm torque, idle drift, rim-grasp close peak and lift, cuMotion path tracking error):

| | stock PD (2e4 / 0) | k = 4400, zeta ~ 1 per joint | k = 1000, zeta ~ 1 per joint |
| --- | --- | --- | --- |
| RMPFlow + current stack (today) | baseline | expected: lag / oscillation (the old soft-gain result) | -- |
| joint-space targets (cuMotion recording path, no teleop) | tracking baseline | contact should soften; check tracking | check tracking |
| Pink IK teleop prototype (scratch env) | -- | primary candidate | fallback |
| differential IK teleop prototype | -- | cheap comparison | -- |

Order: (1) baseline row today; (2) joint-space row -- it needs no new code, only the gains, and
tells whether soft gains hold a pinch and track a plan; (3) Pink IK prototype with the gains that
won in (2), 5 minutes of noVNC teleop for feel (redundancy, elbows, self-collision), the same
measurements; (4) differential IK only if Pink is awkward to set up; (5) decide. If a joint-space
teleop path wins, the change is one action config class on the embodiment plus a device
retargeter and the PD in `agibot.py`; the recorder, LeRobot configs and cuMotion path do not move,
and `joint_pos_target` becomes a cross-check.

Until then the standard stays: RMPFlow + target hold + surface guard + ramped gripper, all on
`cuda:0`, with the frozen actuator config.
