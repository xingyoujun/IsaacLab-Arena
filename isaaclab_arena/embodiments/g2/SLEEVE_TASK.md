# G2 peg-into-sleeve task

`g2_sleeve` uses two Arena Factory assets: a movable cylindrical `peg` and
an upright fixed `hole` instantiated as `sleeve`. G2 picks up the rod from
above and inserts its bottom into the sleeve until it reaches the bore floor.

## Assets and layout

Both assets retain the library's 3x scale and SDF collision geometry.
The rod is approximately 24 mm in diameter and 150 mm tall. The sleeve has
a 27 mm bore, 75 mm bore depth, and 120 mm square bottom flange.
The tabletop is at world z=0. The rod starts at (0.05, -0.22, 0.003) m and
settles onto the table. The fixed sleeve's bore-floor origin is at
(-0.10, 0.0, 0.009) m; its opening faces up at z=0.084 m.
`--peg_xy x y` and `--sleeve_xy x y` configure separated tabletop positions.

The sleeve's authored world joint is disabled in the spawned stage and
replaced by a kinematic rigid-body fixture. The source USD is unchanged.

## Success

All conditions are evaluated in the sleeve frame:

- Peg bottom centre is within 1 mm of the bore axis.
- Peg bottom is within 3 mm of the bore floor.
- Peg and sleeve local Z axes align within 1 degree; axial rotation is unrestricted.

Insertion completes the environment episode. Gripper release is not required
by the task predicate. Hovering, partial insertion and reversed axes fail.
The default timeout is 120 s; dropping the peg below z=-0.10 m fails.
Reset restores both initial poses. Existing preview environments are unchanged.

Configuration: `isaaclab_arena_environments/g2_sleeve_environment.py`.
Success predicate: `isaaclab_arena/tasks/sleeve_task.py`.
Targeted tests: `isaaclab_arena/tests/test_g2_sleeve.py`; these cover rotated
coordinate frames, incorrect poses, physical insertion, termination and reset.

## cuRobo collection

Run as the host user in the clone's discovered cuRobo container:

```bash
cd /workspaces/isaaclab_arena
PYTHONPATH=/workspaces/isaaclab_arena/submodules/IsaacLab/source/isaaclab \
/isaac-sim/python.sh isaaclab_arena_examples/g2_sleeve_curobo.py \
  --robot_yaml /datasets/agibot_dataset_v1_raw/assets/g2/robot.yaml \
  --robot_urdf /datasets/agibot_dataset_v1_raw/assets/g2/robot.urdf \
  --output_dir /datasets/agibot_dataset_v1_raw/pilot/peg_into_sleeve/attempt_new \
  --headless --device cpu --enable_cameras --record_video
```

A fresh output directory is required. One attempt is the default. The collector
uses joint trajectories and real gripper contact, measures lift, aligns the rod,
inserts it, opens the gripper and retreats upward. Planning attachment spheres
represent the rod without physically attaching it to the robot. Assembly
objects are excluded from planning during contact; their physics stays active.
The collector additionally requires 15 stable successful steps after release.

`episodes.hdf5` contains successes; `episodes_failed.hdf5` retains failures.
Raw data contains actions, measured joints and TCP poses, and object states.
Four aligned MP4s contain overview, head and both wrist cameras at 15 Hz.
`report.json` records phases, measured execution errors and task variant
`movable_peg_fixed_sleeve`. Source snapshots preserve provenance.

Validate a successful single-attempt recording:

```bash
/isaac-sim/python.sh isaaclab_arena_examples/check_g2_sleeve_dataset.py \
  /datasets/agibot_dataset_v1_raw/pilot/peg_into_sleeve/attempt_new
```

Validation independently checks rod lift, fixed sleeve, final insertion,
gripper release, finite data and video alignment, and writes artifact hashes.
The earlier `pilot/sleeve/attempt_006` records the opposite task direction;
it is retained as historical data and must not be used as this task's demo.

## Validated pilot

On 2026-09-16, `pilot/peg_into_sleeve/attempt_001` completed one physical
right-arm demonstration: 160.1 mm lift, full insertion, release and 100 mm
upward retreat. The raw and all four videos contain 1133 frames at 15 Hz
(75.5 s). Independent validation measured 0.252 mm final radial error and
confirmed the sleeve stayed fixed. Overview frames were visually inspected.
The collector uses a 600 s diagnostic timeout with explicit final success
checks; the normal task's 120 s timeout accommodates this pilot trajectory.

Host location:
`/home/ubuntu/datasets/agibot_dataset_v1_raw/pilot/peg_into_sleeve/attempt_001`.
The two targeted simulation tests and scoped pre-commit checks pass.
The full three-phase test suite was not run. One fixed-layout pilot does not
establish a collection success rate.

## Background batch collection

`bash isaaclab_arena_examples/launch_g2_peg_collection.sh --demos 200` runs
one worker sequentially, collecting successful raw episodes before offline
three-camera LeRobot v2.1 export. Defaults match stack bowls: independent seeds
starting at 10000, independent ±2 cm XY offsets for both objects, at most 600
attempts, 900 s per attempt, and stop after three consecutive initialization
crashes. The fixed sleeve stays stationary within each episode.

Final host output: `/home/ubuntu/datasets/agibot_dataset_v1/peg_into_sleeve/`
(`data/`, `meta/`, `videos/` only). Raw, logs, provenance and checkpoints live at
`/home/ubuntu/datasets/agibot_dataset_v1_raw/peg_into_sleeve/`.
Only independently validated successes enter the dataset. A filesystem lock
rejects another collector for this task. Resume with the same command and
unchanged sources/configuration; `--stage core` and `--stage render` are supported.
The task annotation is peg insertion, with the same G2 fields and pre-action
15 Hz alignment as stack bowls. No upload is performed.
