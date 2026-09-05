# Triage: which layer is at fault?

Five layers can make an Agibot task misbehave, and their symptoms overlap almost completely. The
tree below takes the symptom to a **discriminating measurement** whose result names the layer.
Run the measurement; do not start from a hypothesis about a parameter.

The layers, from cheapest to most expensive to change:

1. **Harness / scaffolding** -- the probe or script itself (auto-reset, stale term, wrong frame).
2. **Task code** -- predicate polarity, thresholds, reset events.
3. **Layout** -- where objects are relative to the arms' real workspace.
4. **Asset** -- origin, collider, size relative to the gripper.
5. **Robot** -- embodiment config, controller stack. **Frozen.** Anything that ends here is a bug
   report, not a tuning job.

## "The object flies out of the gripper"

Four measured causes, one discriminating run each. Run them in this order.

| step | run | result -> cause | fix (already standard) |
| --- | --- | --- | --- |
| 1 | `probe_pinch.py` on the object (mid-air, no table) | ejected at 2-7 m/s, fingers close to ~2 mm -> **asset too big for the pads**: linkage inboard of the pads strikes it (parts > ~80 mm along the jaw-perpendicular axis, > 104 mm across, flange under a neck, grip band < 12 mm) | resize / regrasp / regenerate the asset. Nothing on the robot fixes this |
| 2 | same probe with `--ramp_seconds 0.01` (instant close) vs the default ramp | held only with the ramp -> **instant binary close** (teleop's stock action steps the target, fingers hit at ~0.9 m/s) | `install_ramped_gripper` -- part of the standard stack |
| 3 | teleop-path table pinch pressed vs not pressed (or read `asked_dz`/`allowed_dz` in the stack_bowls trace) | flies only when pressed into the table -> **commanded press-through** (stiff arm, damping 0, drives through the surface; closing while pressed gives 1.0 m/s finger speeds) | `install_surface_guard` -- part of the standard stack |
| 4 | watch the reset: object moves before anyone touches it | -> **spawned overlapping a fixture collider** (a 254 mm wrench at a random yaw reached into the bin) | rejection-sample the scatter against fixture keep-out rectangles using the yawed footprint |

If all four are clean and the object still leaves the hand mid-carry, check the **controller path**:

- Scripted driver: was `env.step()` called while `ArmExecutor` was driving? The arm goes limp and
  drops the object; the object lands directly below the staging pose (see cumotion_guide).
- Recording path (`EnvActionExecutor`): a gripper term with a smoothed/interpolated target pulses
  the grip at 15 Hz (97-138 mm of in-hand drift vs 13 mm). Gripper terms must be plain
  `JointPositionActionCfg` (they are, in `AgibotDualArmJointActionsCfg`).
- Carry speed: handover's slow speeds (climb 0.10, cross 0.06, lower 0.06) are the settled values;
  a 3x speed-up shed the slice repeatedly.
- Settle time is in **seconds**: a decimation rescale once cut 4 s to 0.3 s and every descent
  stalled on still-sliding objects.

## "The arm cannot reach / the grasp stops short"

| run | result -> cause |
| --- | --- |
| `probe_reach.py` at the object with tilts 0,30,50,75 | zero at every lean -> **layout**: move the object into `REACH_X_BAND_M` (0.35-0.45) / raise the surface / assign the other arm. Both arms differ: scan the one you use |
| `probe_drop_settle.py` | resting origin height not the documented convention -> **asset origin**: grasps aimed at the origin straddle the part off-centre (the bread's origin is its bottom face; one pad jammed after 3.7 mm) |
| `probe_gripper_span.py` (tool frame vs pads) | commanded pose reached to 0.1 mm but pads elsewhere -> **tool frame is 16.7 mm past the pads**; use `slab_grasps(approach_offset_m=0.017)` |
| `probe_tool_orientation.py` | left arm reaches the position but jaws point 90 deg off -> **generated robot description relabels the left tool axes**; the planner's `_measure_tool_correction` handles it; if you bypass the planner, you must too |
| descent stalls > 20 mm short in a scripted run | neighbours pushed: `pick()` descends **wide open** (104.5 mm gap); use `pregrasp_gripper_pos` (0.25 = 30 mm) and `max_attempts=1` when a failure disturbs the scene |
| RMPFlow arm stalls at z ~0.89 and springs back | c-space attractor floor (`cspace_target_rmp`); fixed in the default config by the reset event -- if you see it, the embodiment's reset event is missing |

## "Success never fires" / "success fires immediately"

| run | result -> cause |
| --- | --- |
| `probe_staged_success.py` with the goal poses | False on a correctly staged goal -> **task predicate**: threshold (rest velocity below the stacked-body noise floor 0.03-0.05 m/s), inverted joint polarity (`Pressable.is_pressed` inverts negative-lower-limit joints), wrong frame box |
| same, `--physics_seconds 2` | True immediately, False after physics -> the **goal pose is not a resting state** (origin convention, `place_upright()` centre trap) |
| True at reset | predicate too loose, or the fixture intersects the object at spawn (a table raised into the gripper reported pressed=False and success=True) |
| object "jumps" one step after being staged in a validation script | **harness**: success -> termination -> automatic reset -> the placement event re-placed it. Disable the success term (probes do this) |
| the stack looks right but never settles | a bowl **jammed inside the open fingers** (110 mm bowl vs 105 mm span): convex shells interlock; correct technique is a rim pinch, and a stuck-timeout guard on the predicate catches it |

## "An idle arm drifts / the left arm runs away"

Both are fixed in the default embodiment; if they reappear, the environment is not using the
standard stack or the embodiment file was edited.

- Zero command, arm walks 88 mm in 0.7 s in dual-arm mode: relative-mode RMPFlow rebuilds the
  target as current pose + delta, so a first-step jump is latched in. `install_arm_target_hold`
  holds the previous target. (`ignore_robot_state_updates=False` is **worse**: 890 mm.)
- Left arm runs 270-450 mm after reset: a wxyz quaternion in an xyzw `OffsetCfg.rot`
  (`(0.7071,0,-0.7071,0)` reads as a 180-degree flip). Correct is `(0, -0.7071, 0, 0.7071)`.
  This project has hit the quaternion order three times -- see the asset guide's conventions box.
- Hands at y = +/-1.12 m after reset, no tracking: the embodiment has no robot-only reset event
  (`reset_joint_position_and_velocity_to_defaults`), so every joint reset to 0 and Lula's fixed
  `joint_lift_body` / `joint_body_pitch` no longer describe the robot.

## Refuted fixes -- do not retry without new evidence

Each of these was measured on the Agibot and made the problem the same or worse. If you find
yourself reaching for one, the measurement above has not been run.

| "fix" | measured result |
| --- | --- |
| gripper `effort_limit_sim` 100 -> 10, `velocity_limit_sim` 10 -> 2 | 100/10 lifts cleanly at 0.586 m/s; 10/10 flings at 5.03 m/s; velocity 2 cannot grasp at all (0/15). A weaker or slower gripper throws harder: the joint lags its target longer and PhysX discharges the error as an impulse |
| gripper damping 0 -> 1.0 (passive) / 0.1 -> 2.0 | pressed-into-table finger speed 1.04 -> 4.69 / 1.22 m/s |
| arm damping 0 -> 40, arm effort -> 100, RoboDojo's ARX X5 triple (4400/40/100) | 1.6 m/s and 223 mm drift; arm collapses under its own weight; softer arm couples with the gripper and drops a bowl mid-lift |
| `ignore_robot_state_updates` True -> False | gripper penetrates the table 46 mm at 2.6 m/s; idle drift diverges to 890 mm |
| raising the surface-guard clearance (18 -> 27 mm) | rim-grasp fling 0.53 -> 1.05 m/s. Clearance is not the anti-fling knob |
| easing the descent near the surface | throttles legitimate descents onto objects: 1.69 m/s vs 0.50 unguarded |
| object collider `minThickness` 20 -> 3 mm, `maxConvexHulls` 128 | regression; stock is correct |
| SDF instead of convex decomposition on the bowl | fixes the pinch, jitters resting bowls (89 mm/s, 8 mm creep) |
| mass override (0.075 -> 0.32 kg) | flings harder at the lighter mass; mass is not the driver |
| `maxDepenetrationVelocity` at run time | USD write is a no-op |
| Factory physics (192 iterations, 5 mm contact offset) for a pinch problem | changes the hold, does not fix causes 1-3 above |
| switching to 50 Hz to fix a 15 Hz failure | 15 Hz is a training-alignment decision; the "15 Hz drops it" case was a settle-time bug, and 50 Hz silently changed how the rack settled |
| mirroring the left rest pose to fix the left arm runaway | not the cause (stock pose was worse); the mirror is now in the default for framing/reach reasons, not for this |

## Method

- One hypothesis at a time, with a measurement that can refute it, and say what result would kill
  it before running. No parameter sweeps chosen by plausibility.
- Change one thing per run. Effort and velocity changed together could not be attributed.
- A failure that reproduces identically under every change to the thing you suspect is evidence
  **against** the suspicion. Check the scaffolding (executor vs `env.step`, auto-reset, the shell
  not word-splitting an unquoted `$var` so two "different" configs ran the same baseline).
- Read the source first: RoboDojo's loaders (`env/scene_manager/objects/`), Isaac Lab's actual
  handling of the parameter, Arena's path to the sim. RoboDojo sets only mass and friction on rigid
  objects and never touches the collider.
- After one failed validation run, report the symptom and the candidate causes to the user before
  the next attempt. The user often has the context that collapses the search.
