# RR sim2real: UR7e agent handoff

Last verified on the source host: 2026-09-11. This branch shares **code and
documentation**, not a complete asset/data/runtime distribution. Read this page
and the root `AGENTS.md` before changing tasks or comparing results.

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
| Original toast on drawer pedestal | `isaaclab_arena_environments/ur7e_press_toaster_environment.py` |
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
| `ARENA_DRAWER_USD` | `rr_ur/drawer_rr_arena.usda`, referencing `drawer_rr.usdc` |
| `ARENA_DRAWER_GPT56_USD` | `rr_ur/drawer_rr_gpt56_arena.usda`, referencing `drawer_rr_gpt56_sol_high_image_mesh.usd` |
| `ARENA_DRAWER_ARTICRAFT_USD` | `rr_ur/drawer_articraft/drawer_articraft.usd` (already scaled, standalone) |
| `ARENA_TOASTER_USD` | `rr_ur/toast_rr_arena.usda`, referencing `toast_rr.usdc` |

Set variables **before** importing Arena. Defaults preserve the source-host
paths. The toast environment also requires the original drawer for its pedestal.
If the UR7e robot is incomplete, the code warns and falls back to UR5e: that is
only useful for plumbing checks, **not a valid UR7e comparison**. Verify the
selected robot and all USD dependencies before collecting/evaluating.

SHA-256 of the current object payloads, for transfer verification:

```text
4bea86ea1ef2441e80a6c05ea5061afd1fd424a2c91b3b7dd218789d0c30054e  drawer_rr.usdc
45f589472ca3b8f7b15902997c703a6bdc12105c7cea4d3bb509c8b723ac2483  drawer_rr_gpt56_sol_high_image_mesh.usd
4335f193549717ccfe27fb3c80f52609ac4e291439b104539229840884b416a1  drawer_articraft/drawer_articraft.usd
0fda19498b23c12adac55836163d8dec42091f7c7d53c09ddccdeb938a4b220d  toast_rr.usdc
```

Source HDRIs are `/home/ubuntu/playground/assets/skies` (11 `.hdr` files); move
them separately and supply `--skies`. Visual randomization also uses Isaac
material libraries, so assets/cache access matters on a fresh machine.

## Experimental invariants

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
  --final /absolute/data/rr_sim2real/open_drawer_articraft_v2 \
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
  task.dataset_path=/home/ubuntu/playground/datasets/rr_sim2real/open_drawer_articraft_v2_dp.zarr \
  logging.project=rr_sim2real logging.name=ur7e_open_drawer_articraft_v2 \
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
| `rr_sim2real/open_drawer_gpt56_v2` | 200 successful demos, 50,803 training rows, randomized LeRobot + DP Zarr |
| `rr_sim2real/open_drawer_articraft_v2` | 200/200 collection attempts succeeded; 51,325 recorded frames → 51,125 training rows; all stop after pull; final opening 99.7–101.9 mm |
| Articraft final files | `rr_sim2real/open_drawer_articraft_v2_dp.zarr`, raw `rr_sim2real_raw/open_drawer_articraft_v2/complete.json` |
| Toast epoch 60 DP | 5/10 success, all failures timeout at 20 s, 0 IK holds; successful episodes 11.1–14.8 s; default background |

Toast checkpoint: source DP output
`2026.09.10/09.18.22_train_diffusion_unet_image_ur7e_press_toaster_image/checkpoints/epoch=0060-train_loss=0.002.ckpt`.
Evaluation: `/home/ubuntu/playground/experiments/ur7e_press_toaster_dp_epoch60/2026-09-11_07-36-37/`,
with canonical JSON, HTML and 20 camera videos (not committed). The inference
service was stopped after evaluation. This is a small-sample result, not a
stable estimate of real-world performance.

The Articraft 200-epoch/batch-64 training command was handed to the user for
tmux execution. Training completion/checkpoint selection is **not yet verified**;
inspect the actual training logs on the owning machine before claiming a result.

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
