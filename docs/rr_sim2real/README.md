# RR sim2real: UR7e agent handoff

**2026-09-25 status audit:** see [local changes and upstream migration assessment](upstream_audit_20260925.md).
All twelve method/task datasets now report 200 black-gripper randomized demos,
and the 240-episode DP evaluation is complete. The dated 2026-09-11 pending-data,
pending-render and training-status statements below are historical, not current
status. Preserve their provenance; use the audit and task-specific handoffs for
current results. No upstream merge or runtime upgrade has been performed.

Last verified on the source host: 2026-09-11. This branch shares **code and
documentation**, not a complete asset/data/runtime distribution. Read this page
and the root `AGENTS.md` before changing tasks or comparing results.

**Current naming:** read [the four-method/eight-asset mapping](asset_naming.md)
and [machine-readable inventory](assets.json) first. They supersede historical
names below. Training has finished and the compatibility Zarr symlink was
removed. Existing exports now carry method prefixes; no dataset content changed.
Only Articraft may receive uniform size normalization. For the newly tested
miniworkflow Astra drawer, see [qualification](../ur7e_open_drawer_astra.md);
the three non-USDcraft toasts are covered in
[toast qualification](../ur7e_press_toaster_baselines.md).

The new [toaster-knob task](turn_toaster_knob.md) reuses all four toast assets on
two boxx supports. It has separate environment names and single-demo cuMotion
recording qualification; it does not replace the press-toaster datasets.
The [four-method bulk pipeline](collect_toaster_knob.md) prepares 200 randomized
knob demos per method with three collection workers and two rendering workers.

## Scope and branch workflow

- Shared fork: `git@github.com:xingyoujun/IsaacLab-Arena.git`, branch
  `chuanruiz/rr_sim2real`. The source worktree is
  `/home/ubuntu/code/IsaacLab-Arena-tasks`. Do not edit its sibling main checkout.
- This work concerns the UR7e real workcell, not the Agibot benchmark. The branch
  inherits existing Arena/cuMotion infrastructure; inherited Agibot files are
  not an instruction to apply Agibot asset qualification gates to UR7e.
- On another machine, clone the fork and select this branch. For concurrent
  work, branch from it into `<user>/feature/<topic>`, coordinate ownership of
  shared files, and merge normally. No force-push or automatic PR to upstream.
- Never commit datasets, checkpoints, robot/object meshes, HDRIs, videos, caches,
  credentials or runtime environments. Transfer required payloads separately.
  `.gitignore` is only a guard; review the staged file list and sizes before push.

## What is implemented

| Component | Entry point |
| --- | --- |
| Robot, cameras, joint controls, recording | `isaaclab_arena/embodiments/ur7e/` |
| Robot FK/IK description | `isaaclab_arena/embodiments/ur7e/rmpflow/` |
| Table, lights, calibrated D435 | `isaaclab_arena_environments/ur7e_workcell_environment.py` |
| Original/image-mesh drawers | `isaaclab_arena_environments/ur7e_open_drawer_environment.py` |
| Width-normalized Articraft drawer | `isaaclab_arena_environments/ur7e_open_drawer_articraft_environment.py` |
| Unscaled Astra drawer | `isaaclab_arena_environments/ur7e_open_drawer_astra_environment.py` |
| Original toast on drawer pedestal | `isaaclab_arena_environments/ur7e_press_toaster_environment.py` |
| Three comparison toasters | `isaaclab_arena_environments/ur7e_press_toaster_baselines_environment.py` |
| Visual randomization | `isaaclab_arena_environments/ur7e_workcell_randomization.py` |
| cuMotion drivers and collection | `isaaclab_arena_cumotion/scripts/ur7e_*`, `collect_ur7e_*` |
| State replay to camera video | `isaaclab_arena_cumotion/scripts/rerender_embodiment_cameras.py` |
| LeRobot, EEF labels, DP replay buffer | `isaaclab_arena_gr00t/lerobot/` |
| DP closed-loop client | `isaaclab_arena/policy/ur7e_diffusion_policy_remote.py` |
| Small external-code migration snapshots | `tools/rr_sim2real/` |

The external `scene.py` layout/calibration source is archived as
`tools/rr_sim2real/reference/scene.py.txt`. It is a reference, not Arena's runtime
entry point. Current Arena layout/camera constants live in the files above.

## Runtime on another machine

Provision Isaac Sim/Arena through the repository's setup instructions/skills in
a **separate installation**. Do not copy the source host's `.venv` symlink.
Simulation requires compatible NVIDIA drivers, Isaac Sim, Isaac Lab and the
`isaacsim.robot_motion.cumotion` extension. A normal Python-only installation
cannot run the task. A clean-machine end-to-end installation has not been tested
as part of this publication.

Recorded submodule revisions (do not silently upgrade for a comparison):

- Isaac Lab: `bb0c8e1b9af381bf13064ec3303e17db79e4b6ef`
- Isaac-GR00T: `e29d8fc50b0e4745120ae3fb72447986fe638aa6`

On the source host only, `.venv` is shared with the main checkout. Every Arena
run must use this worktree first on `PYTHONPATH`, plus the Isaac Lab source:

```bash
cd /home/ubuntu/code/IsaacLab-Arena-tasks
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export ARENA_ISAACLAB_SOURCE=/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab
export PYTHONPATH="$PWD:$ARENA_ISAACLAB_SOURCE"
```

On another host, change these paths to the actual clone and Isaac Lab source.
`ARENA_ISAACLAB_SOURCE` is also read by the integrated drawer collection pipeline.
Do not run `uv sync` or install packages into the shared source-host environment.
The Docker section in the general AGENTS guide does not apply to this source
host's native worktree; a separately provisioned machine may use supported Docker.

Use the Experiment Runner, headless (`--viz none`), and sensor-camera recording.
Do not use headless viewport recording: this host's `render()` returns None.
The direct Policy Runner has a source-host pxr version collision; it is not the
supported entry point here. Camera videos flush when an episode ends.

Reserve the GPU for collection/rendering. Two simulation workers were used on
an L40S. Concurrent DP **training** has caused RTX crashes; one DP inference
server plus one simulator was successfully tested. Stop only processes you own.

## External assets: obtain separately, preserve dependencies

Source root: `/home/ubuntu/playground/rr_ur`. Copy the collected robot's complete
directory and all USD references/textures, not just its top-level file. Object
overlays are tiny editable configuration layers included under
`tools/rr_sim2real/asset_overlays/`; place them beside the referenced payloads.
They are not substitutes for the missing meshes/USD payloads.

| Environment variable | Source-host value / required asset |
| --- | --- |
| `ARENA_UR7E_USD` | `rr_ur/ur7e_usd/ur7e_gripper/Collected_ur7e_gripper/ur7e_gripper.usd` and its complete dependency directory |
| `ARENA_DRAWER_USD` | `rr_ur/usdcraft_drawer_arena.usda`, referencing `usdcraft_drawer.usdc` |
| `ARENA_DRAWER_GPT56_USD` | `rr_ur/miniworkflow_gptsol_drawer_arena.usda`, referencing `miniworkflow_gptsol_drawer.usd` |
| `ARENA_DRAWER_ARTICRAFT_USD` | `rr_ur/articraft_drawer/articraft_drawer.usd` (already scaled, standalone) |
| `ARENA_DRAWER_ASTRA_USD` | `rr_ur/miniworkflow_astra_drawer_arena.usda`, referencing `miniworkflow_astra_drawer.usd` |
| `ARENA_TOASTER_USD` | `rr_ur/usdcraft_toast_arena.usda`, referencing `usdcraft_toast.usdc` |
| `ARENA_TOASTER_GPTSOL_USD` | `rr_ur/miniworkflow_gptsol_toast_arena.usda` and its original USD |
| `ARENA_TOASTER_ASTRA_USD` | `rr_ur/miniworkflow_astra_toast_arena.usda` and its original USD |
| `ARENA_TOASTER_ARTICRAFT_USD` | `rr_ur/articraft_toast/articraft_toast.usd` (already uniformly scaled) |

Set variables **before** importing Arena. Defaults preserve the source-host
paths. The toast environment also requires the original drawer for its pedestal.
If the UR7e robot is incomplete, the code warns and falls back to UR5e: that is
only useful for plumbing checks, **not a valid UR7e comparison**. Verify the
selected robot and all USD dependencies before collecting/evaluating.

SHA-256 of the current object payloads, for transfer verification:

```text
4bea86ea1ef2441e80a6c05ea5061afd1fd424a2c91b3b7dd218789d0c30054e  usdcraft_drawer.usdc
45f589472ca3b8f7b15902997c703a6bdc12105c7cea4d3bb509c8b723ac2483  miniworkflow_gptsol_drawer.usd
4335f193549717ccfe27fb3c80f52609ac4e291439b104539229840884b416a1  articraft_drawer/articraft_drawer.usd
0fda19498b23c12adac55836163d8dec42091f7c7d53c09ddccdeb938a4b220d  usdcraft_toast.usdc
```

Source HDRIs are `/home/ubuntu/playground/assets/skies` (11 `.hdr` files); move
them separately and supply `--skies`. Visual randomization also uses Isaac
material libraries, so assets/cache access matters on a fresh machine.

## Experimental invariants

### Black gripper default (2026-09-11 onward)

All UR7e environments now use `embodiments/ur7e/appearance.py` through the shared
robot spawn function. Only the Robotiq subtree gets a strong inherited black
visual material (diffuse 0.015, roughness 0.5). Arm appearance, USD source files,
geometry, joints, collisions and friction are unchanged. This includes both
joint-control and IK embodiments, collection, replay and DP evaluation.

RTX did not honor the inherited override on the instanced inner fingers: their
tips remained white. The corrected implementation makes only left/right
`inner_finger/visuals` non-instanceable and directly binds black `full`/`preview`
materials to their 4 meshes and 14 face subsets. All-purpose/physics material
bindings and collision branches remain untouched. Cache identity is now
`black_fingertips_direct`; earlier `black_v1` and diagnostic `black_fingertips_v2`
sidecars must not be reused as corrected renders. This is an internal appearance
marker, not a dataset filename suffix.

Existing datasets/checkpoints and the results below predate this change. They
are **not** black-gripper data. Do not claim they have been regenerated. The
corrected fingertip bindings and unchanged physics properties were checked on
CPU; one Astra drawer trajectory (248 frames, D435 + scene) was replayed on GPU
and the formerly white fingertips verified black on 2026-09-11. Preview:
`rr_ur/sim_renders/astra_black_fingertips_direct.png`. Video sidecars:
`datasets/rr_sim2real_aux/black_fingertips_validation/astra_drawer.hdf5.cameras/`.
No full training dataset was regenerated by this visual fix.

When the GPU is available, first render a one-demo preview in an isolated raw
directory, then render the full toaster dataset. These commands are prepared,
not yet executed:

```bash
.venv/bin/python isaaclab_arena_cumotion/scripts/rerender_ur7e_press_toaster_black.py \
  --target 1 --raw /home/ubuntu/playground/datasets/rr_sim2real_raw/press_toaster_black_preview
# After checking that preview:
.venv/bin/python isaaclab_arena_cumotion/scripts/rerender_ur7e_press_toaster_black.py
```

The second command reuses the 200 original successful v2 trajectories and seed-0
visual randomization, with two staggered replay workers. It writes a separately
named HDF5 and camera sidecars under `rr_sim2real_raw/usdcraft_press_toaster_black_gripper`.
It does not overwrite v2 or publish training data. Subsequently convert to a
new LeRobot/Zarr version and retrain; do not replace data used by active training.
The camera re-renderer rejects legacy sidecar caches without the black-material
marker; copying the HDF5 to a fresh version is preferred over deleting old videos.

Preview PNGs and smoke outputs were moved out of the HF upload root to
`/home/ubuntu/playground/datasets/rr_sim2real_aux/2026-09-11_cleanup/` with a
`move_manifest.json`. Official dataset paths are unchanged. Historical preview
links in older completion files refer to their pre-move locations.

### Geometry, motion and data comparison rules

- Robot, table and D435 calibration are shared across drawer ablations. Drawer
  resets use x `[0.05, 0.18]` m, y `[0, 0.15]` m and yaw `-90 +/- 10` degrees.
  cuMotion collection adds 0.03 rad start-joint jitter, default pull 0.10 m.
- Original drawer width is **0.198 m**. Articraft was uniformly scaled **0.495**
  in place to this width. Its enclosure is 0.198 x 0.2475 x 0.07425 m, slide
  travel 0.1683 m, success above 0.08415 m. Do not scale twice, change individual
  axes, reshape the knob or fix the wrong aspect ratio: that mismatch is the
  intended ablation. Original Articraft URDF/OBJ remain unscaled references.
- GPT56 slide opens in the negative direction and succeeds past 77.5 mm;
  Articraft opens in the positive direction. The driver uses per-asset knob
  geometry/joint sign. Do not validate all assets using one signed joint value.
- Open-drawer comparison demos end **after pulling, before release/retreat/home**.
  Use `--stop-after-pull`. The driver's flag-free default still runs the longer
  sequence; do not mix those datasets.
- Toast uses original `toast_rr`, on a +78 mm original-drawer pedestal. Random
  x `[-0.22,-0.10]`, y `[0.10,0.14]` m, yaw `0 +/- 10` degrees. Success is
  `carriage_slide > 0.75 * 0.047 m` (35.25 mm). No return spring is simulated.
- Visual randomization changes background/HDRI, light, table/floor material and
  0–6 non-colliding distractors per demo. It replays fixed recorded states and
  does not perturb robot or task physics. Default seed 0; retain per-demo JSON.
- Matched pose distributions/seeds do not guarantee paired successful episodes:
  rejection and success-only export can select different samples across assets.

For detailed asset history, see [Articraft](../ur7e_open_drawer_articraft.md) and
[image-mesh drawer](../ur7e_open_drawer_gpt56.md). Old **unscaled** Articraft
recordings must not be replayed against the scaled USD.

## Collect and convert

For all eight cases in one resumable tmux job, use the
[black-gripper master pipeline](prepare_all_datasets.md). It replays the four
existing datasets, collects the four missing ones, validates both formats, and
archives old exports outside the HF root before replacing canonical pairs.
The prepared full batch has not yet been launched.

The maintained GPT56/Articraft workflow is:

states-only cuMotion → exactly 200 successful demos → randomized camera replay
→ LeRobot v2.1 → EEF labels → DP Zarr → validation and completion marker.

There is **no need to render v1 first**. Each integrated pipeline uses two
workers, batches of 8 attempts, at most 40 rounds, and resumes using its
`pipeline.json` and file lock. Use a new raw/final directory when changing
settings or assets. Never run two jobs into one dataset directory.

Source-host Articraft command (in tmux, after setting runtime variables):

```bash
bash isaaclab_arena_cumotion/scripts/collect_ur7e_open_drawer_articraft.sh
```

For GPT56 use `collect_ur7e_open_drawer_gpt56.sh`. On another machine override:

```bash
export ARENA_DP_REPO=/absolute/path/to/diffusion_policy
bash isaaclab_arena_cumotion/scripts/collect_ur7e_open_drawer_articraft.sh \
  --raw /absolute/data/rr_sim2real_raw/open_drawer_articraft_v2 \
  --final /absolute/data/rr_sim2real/articraft_open_drawer \
  --skies /absolute/assets/skies \
  --dp-python /absolute/robodiff/bin/python
```

Run a separate `--target 1` smoke pipeline before starting 200 on a fresh host.
Do not resume the one-demo directory with a new target. `complete.json` is
written only after video, modalities, episode counts and Zarr validation pass.
The shared implementation retains its historical `collect_*_gpt56.py` filename;
the Articraft entry point passes its own environment, joint sign and threshold.

Original drawer/toast v1/v2 orchestration is archived as **non-executable text**
in `tools/rr_sim2real/reference/*.sh.txt`. These are provenance, not maintained
launchers: they contain source-host absolute paths, weaker error handling and
`rm -rf` output replacement. Do not run blindly. Port them to isolated outputs,
update PYTHONPATH, and remove destructive replacement before reuse. The drivers,
re-renderer, truncation and conversion utilities they reference are in this repo.

## Dataset contract (do not silently change)

### Canonical dataset names after cleanup on 2026-09-11

The deprecated original `open_drawer`, `open_drawer_dp.zarr` and `press_toaster`
exports were deleted at the user's request. Their former randomized v2
replacements now occupy those canonical names. Likewise GPT56 and Articraft
exports have no `_v2` suffix. All four canonical datasets mean **randomized
backgrounds/materials/distractors by default**; they have not yet been regenerated
with black grippers. New collection pipelines also omit `_v2` in default names.
Existing raw HDF5 directories and historical logs retain their original names.

The user subsequently confirmed Articraft training finished. Its temporary
`open_drawer_articraft_v2_dp.zarr` symlink was removed during method-prefix
renaming. Future runs use `articraft_open_drawer_dp.zarr`; old checkpoint
resumes must override the historical dataset path. Stored training configs
and checkpoints were not rewritten.

`rr_sim2real/README.md` lists the eight canonical export directories and counts.
The migration manifest is in `rr_sim2real_aux/2026-09-11_cleanup/`.
Archived v1/v2 shell scripts are historical references, not valid publication
commands: they can overwrite canonical data or recreate deprecated output names.
Do not use them unchanged. For new material experiments, write isolated raw and
staging directories, validate, then explicitly replace the canonical export only
when no active training reads it. This avoids publishing numbered duplicates.

| Representation | Contract |
| --- | --- |
| Raw HDF5 | Recorded states + 7 absolute joint targets; preserve root/data metadata and quaternion format version |
| LeRobot state/action | `[6 UR arm joints, finger_joint]`, radians, 7D |
| LeRobot EEF 9D | UR `base` frame: TCP xyz + first two rotation-matrix **columns** |
| D435 LeRobot video | H.264, 640x480, 15 fps, one `observation.images.realsense_d435` stream |
| DP image | Separate 320x240 resize, HWC uint8 in Zarr |
| DP EEF/action | Rotation 6D **rows**, TCP in UR base; action 10D = xyz + rot6d + gripper |
| DP gripper | `finger_joint / (pi/4)`, not a jaw width in metres |

The TCP is `tool0 + 0.1628 m` along tool z. UR `base` differs from the URDF root
by 180-degree yaw. Reuse the checked-in FK/IK and rotation converters. Converting
next-step targets drops one training row per episode; extra final video frames
are expected. Register DP's vendored JPEG2000 codec when reading compressed Zarr.

## DP training and inference

DP runs in its own `robodiff` environment, **not** Arena's Python environment.
The source DP checkout is `/home/ubuntu/code/diffusion_policy`, baseline commit
`5ba07ac6661db573af695b419a7947ecb704690f` plus local modifications. The small
UR7e-specific files and workspace config are included in
`tools/rr_sim2real/diffusion_policy_overlay/`. Review/diff and copy them into a
compatible DP checkout, preserving relative paths; do not overwrite someone
else's uncommitted changes. This is not a full fork or installer.

Known source DP runtime: Python 3.9, torch 1.12.1, torchvision 0.13.1,
diffusers 0.11.1, hydra-core 1.2.0, omegaconf 2.2.3, numpy 1.23.3,
zarr 2.12.0, numcodecs 0.10.2, imagecodecs 2022.9.26, robomimic 0.2.0,
av 10.0.0, wandb 0.13.3. These describe the existing environment, not a tested
cross-hardware lockfile. Check old torch/CUDA compatibility on the target GPU.

Articraft training command (fresh timestamped run, 200 epochs, batch 64):

```bash
cd /home/ubuntu/code/diffusion_policy
/home/ubuntu/miniconda3/envs/robodiff/bin/python train.py \
  --config-name=train_diffusion_unet_real_image_workspace \
  task=ur7e_open_drawer_image \
  task.name=ur7e_open_drawer_articraft_image \
  task.dataset_path=/home/ubuntu/playground/datasets/rr_sim2real/articraft_open_drawer_dp.zarr \
  logging.project=rr_sim2real logging.name=ur7e_open_drawer_articraft \
  training.num_epochs=200 training.checkpoint_every=20 \
  training.rollout_every=1000 dataloader.batch_size=64 dataloader.num_workers=4
```

Use **real_image_workspace**, not `train_diffusion_unet_image_workspace`:
the latter has different crop dimensions and checkpoint metrics. The retained
config uses ResNet18, 240x320 input, 216x288 crop, horizon 16, 2 observations,
8 action steps, EMA and seed 42. WandB is online by default; configure credentials
privately, or explicitly choose offline logging on a new host. Output goes to
`data/outputs/<date>/<time>_train_diffusion_unet_image_<task>/checkpoints`.

To evaluate a user-selected checkpoint, start the snapshot's server in DP's
environment with `--ckpt <path> --port 5758 --num-inference-steps 100`, then run
from the Arena checkout with the runtime variables set:

```bash
.venv/bin/python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/experiment_configs/ur7e_press_toaster_dp_eval.yaml \
  --viz none --enable_cameras --record_camera_video \
  --output_base_dir /absolute/output/toast_dp_eval
```

This YAML uses original toast, random placement, 10 episodes, 20 s maximum,
one environment, default scene appearance. It does **not** enable visual domain
randomization. Do not use this toast YAML for an Articraft drawer evaluation;
create a typed YAML selecting `ur7e_open_drawer_articraft` and the matching server.
The server's filename says drawer but its protocol is also used by toast.

The current ZMQ server uses pickle and binds all interfaces: trusted hosts only,
firewall the port, and never expose it to untrusted networks. It is a research
adapter, not a production endpoint. It does not explicitly seed diffusion
sampling; environment seed alone does not make policy actions reproducible.
The client uses cuMotion **IK**, not collision-aware motion planning, to execute
predicted TCP targets. This workflow does not authorize deployment on a real robot.

## Verified results and current next step

Source data lives under `/home/ubuntu/playground/datasets/`, outside Git:

| Dataset / evaluation | Verified outcome |
| --- | --- |
| Non-USDcraft toast qualification, black fingertips | GPTSOL 3/3 (57.5 mm), Articraft 3/3 (47.732 mm), Astra 3/3 (61.0 mm) after collision repair; no training datasets ([report](../ur7e_press_toaster_baselines.md)) |
| miniworkflow Astra drawer qualification | 3/3 genuine pulls, 98.878/98.955/98.874 mm; unscaled, minimal root overlay; no 200-demo dataset yet ([report](../ur7e_open_drawer_astra.md)) |
| `rr_sim2real/miniworkflow_gptsol_open_drawer` | 200 successful demos, 50,803 training rows, randomized LeRobot + DP Zarr |
| `rr_sim2real/articraft_open_drawer` | 200/200 collection attempts succeeded; 51,325 recorded frames → 51,125 training rows; all stop after pull; final opening 99.7–101.9 mm |
| Articraft final files | `rr_sim2real/articraft_open_drawer_dp.zarr`, raw `rr_sim2real_raw/open_drawer_articraft_v2/complete.json` |
| Toast epoch 60 DP | 5/10 success, all failures timeout at 20 s, 0 IK holds; successful episodes 11.1–14.8 s; default background |

Toast checkpoint: source DP output
`2026.09.10/09.18.22_train_diffusion_unet_image_ur7e_press_toaster_image/checkpoints/epoch=0060-train_loss=0.002.ckpt`.
Evaluation: `/home/ubuntu/playground/experiments/ur7e_press_toaster_dp_epoch60/2026-09-11_07-36-37/`,
with canonical JSON, HTML and 20 camera videos (not committed). The inference
service was stopped after evaluation. This is a small-sample result, not a
stable estimate of real-world performance.

The user reports the Articraft 200-epoch/batch-64 training has finished. The
training process exited; final logs/checkpoint selection and policy performance
have not yet been evaluated. Inspect them before claiming a training result.

Recommended continuation: provision/check assets → isolated one-demo smoke →
verify selected training checkpoint and dataset identity → run matched original,
GPT56 and Articraft evaluations → report success counts and both success/failure
videos. Preserve intended geometry mismatches. Ask before changing calibration,
task thresholds, control conventions or the definition of the ablation.

## Validation boundary

The source-host collection and toast evaluation results above were measured
before publication. Publication adds docs, migration snapshots, path overrides
and lint cleanup. Static checks do not qualify a new GPU, clean installation,
relocated assets, or newly copied DP environment. Run the smoke workflow on each
new machine; do not interpret a missing asset/server as a learned-policy failure.
