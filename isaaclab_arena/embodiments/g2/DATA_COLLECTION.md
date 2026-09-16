# G2 randomized stack-bowls collection

Run from the host repository root inside tmux. The launcher discovers this
workspace's running cuRobo container and runs as the host user, not root.

```bash
tmux new -s g2-stack-bowls
bash isaaclab_arena_examples/launch_g2_collection.sh --demos 200
```

This first collects **200 successful** physical demos without cameras, then
renders all three native cameras and exports LeRobot **v2.1**. It does not upload
anything. There is a default budget of 600 attempts and 900 seconds per attempt;
three consecutive worker crashes stop the run. Planning/grasp/task failures do
not count as successes. Inspect the logs if the attempt budget is exhausted.

Detach tmux with `Ctrl-b d`. Reattach with `tmux attach -t g2-stack-bowls`.
After interruption, rerun the same command: completed raw attempts and committed
rendered episodes are reused. A per-run lock rejects concurrent writers.
Keep the source and collection parameters unchanged during a run: source hashes
are pinned and a changed configuration is rejected on resume.

To explicitly separate collection and rendering:

```bash
bash isaaclab_arena_examples/launch_g2_collection.sh --demos 200 --stage core
bash isaaclab_arena_examples/launch_g2_collection.sh --demos 200 --stage render
```

## Output

The default host destination is
`/home/ubuntu/datasets/agibot_dataset_v1/stack_bowls/`
(container path `/datasets/agibot_dataset_v1/stack_bowls/`). It contains only
`data/`, `videos/`, `meta/`, matching the task-root layout of
[agibot_arena_v0](https://huggingface.co/datasets/xingyoujun/agibot_arena_v0/tree/main/stack_bowls).
There is no physical `train/` directory. LeRobot's `meta/info.json` still declares
the logical training split; that is metadata, not a filesystem folder.

All private working files live outside this sync root, in the sibling directory
`/home/ubuntu/datasets/agibot_dataset_v1_raw/`. `--raw_root` overrides this path;
by default it is `--root` with `_raw` appended. Raw must remain outside the sync root.
Formal work files are under `agibot_dataset_v1_raw/stack_bowls/`:

- `attempts/attempt_*/`: permanent native HDF5 core, full physical scene states,
  actions, joint/EEF observations, success/failure reports and planner snapshots.
- `provenance/`: collector, renderer, robot/environment source snapshots and
  matching vendor URDF/YAML hashes.
- `episodes/`: per-episode transactional completion markers and artifact hashes.
- `collection_manifest.json`: all attempts and validated successful raw episodes.
- `logs/`: per-attempt output and `render.log` for the offline pass.
- `render_staging/`: interrupted temporary encodes, if any; never treated as
  completed episodes. Source raw is never deleted by the pipeline.

`--split pilot` keeps its exported dataset in
`agibot_dataset_v1_raw/pilot/stack_bowls/` and work files in
`agibot_dataset_v1_raw/pilot/work/`; neither enters the sync root.
Sync `agibot_dataset_v1/` only. No raw exclusion rule is needed with this layout.

The first successful perturbed raw was about 1.38 MB without RGB. Actual sizes
depend on trajectory length; videos are only encoded once into the final dataset.
Preserve the external `/home/ubuntu/datasets/GenieSimAssets` library as well as
the raw/source snapshots to enable future re-rendering. Model/texture assets are
not duplicated into every episode. Permanent planner inputs for the launcher are
under `agibot_dataset_v1_raw/assets/g2/`, not `/tmp`.

## Randomization and acceptance

Each independent attempt gets seed `10000 + attempt_index`. Each bowl receives
independent uniform XY perturbations of up to 2 cm on each axis around its
nominal position. Configurable with `--seed` and `--bowl_xy_noise_m`; use a new
split/root if changing these after collection starts. The table stays at 0.75 m.
Layouts outside table margins or with bowl-centre separation below 18 cm are
rejected. Actual settled states and requested coordinates are retained.

The G2-specific planner/physical execution checks remain enabled. Both lifts must
exceed 10 cm. The task success predicate and correct bowl ordering must hold for
ten final steps after release. Export additionally checks final stability,
finite actions/observations, normalized quaternions, raw hashes and video lengths.
Failed attempts remain in raw and are excluded from LeRobot.

## Training fields and timing

- `observation.joint_position`: **46** measured absolute simulator joints in
  radians, including arms, gripper linkage, head, torso and fixed-base wheel joints.
- `observation.joint_velocity`: the same 46 joints in radians/second.
- `observation.eef_pose`: **14** values, right `[x,y,z,qx,qy,qz,qw]` then left,
  expressed relative to the robot root/base; position units are metres.
- `observation.eef_pose_world`: the same TCPs in simulation world coordinates;
  the tabletop is world z=0.
- `observation.state`: **60** values, absolute joints followed by base-frame EEF poses.
- `action`: **16** values, right arm absolute joint targets (7), right gripper (1),
  left arm absolute joint targets (7), left gripper (1); +1 open, -1 close.
- `observation.images.head`: 640x400 RGB.
- `observation.images.left_wrist`, `observation.images.right_wrist`: 640x528 RGB.

All fields are aligned at **15 Hz**: `observation[t]` and `images[t]` describe the
state before `action[t]`. Offline rendering restores these exact states without
stepping physics, including the actual gripper/object motion. It checks restored
TCP positions against stored EEF poses to within 1 mm. Videos use H.264/yuv420p.
Joint names, units, pose convention and modality slices are saved under `meta/`.
These are G2 fields, not Agibot's 159-state/40-action padded schema.

## Physical replay checks

EEF observations are measured poses, not the original controller targets. The
following diagnostic restores the initial scene once, then steps physics without
restoring intermediate states. Run inside the workspace's cuRobo container:

```bash
/isaac-sim/python.sh isaaclab_arena_examples/g2_check_pose_replay.py \
  --raw /datasets/agibot_dataset_v1_raw/stack_bowls/attempts/attempt_000000/episodes.hdf5 \
  --mode eef --repeat 2 --output /tmp/g2_eef_replay.json --headless --device cpu
```

EEF mode feeds the next recorded base-frame pose into the native absolute-pose
differential IK controller, with the original current-step gripper command.
`--repeat 2` holds each target for two simulation steps (half-speed playback).
Use `--mode joint --repeat 1` for the original joint-action baseline.

Validation on episode 0: joint-action replay reproduced the recorded trajectory
and completed the task; EEF replay at original speed did not complete it, whereas
half-speed EEF replay completed it. Half-speed mean right/left position errors
were 2.9/3.3 mm, but peak errors reached 57/64 mm. This is a single-episode check,
not a guarantee across the dataset. EEF poses alone do not specify arm nullspace
or preserve the original joint-controller dynamics. All 200 exported episodes
(241647 frames) were separately checked for exact EEF agreement with raw data,
normalized XYZW quaternions and consistent base/world transforms.

The renderer needs Isaac Sim, imageio/ffmpeg, h5py and pyarrow already present in
the current cuRobo container. The official LeRobot package is not required to
collect/export; consumers should use a v2.1-compatible loader or explicitly
convert to a newer schema. Do not label this dataset v3 without conversion.
