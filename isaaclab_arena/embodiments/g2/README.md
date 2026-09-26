# G2

G2 uses the shared data_engine native cuMotion planner and environment action executor.
Follow the [G2 development and collection guide](../../../docs/g2_development.md).

Supported collection tasks are `stack_bowls`, `peg_into_sleeve`, and `clean_workcell_table`.
Use `tools/data_collection/run_g2.sh` for collection, raw validation, three-camera rendering and export.
The implementation lives in `isaaclab_arena_cumotion/g2_collection/`.
Retired cuRobo collectors, independent replay tools and old launchers were removed from this branch;
Git history retains them.

Robot USD, planning descriptions, room and task objects resolve through the pinned HF
`xingyoujun/USDCraft-Scene` release. No historical `/datasets` mounts are required.
Git owns the robot configuration, authored task semantics and calibration; HF contains matching snapshots.

Collection uses absolute joint-position actions: right arm (7), right gripper (1), left arm (7),
left gripper (1). Gripper commands are +1 open and -1 close. Observations and all three camera
images precede the matching action. Cameras are head 640×400 and two wrists 640×528, at 15 FPS.

The keyboard controller remains an embodiment utility with its separate relative-pose action
interface. It is not a dataset collection or validation entrypoint.
