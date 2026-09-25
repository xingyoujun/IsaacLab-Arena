# Sequential DP training for the four knob tasks

Run on the source host, in tmux:

```bash
bash /home/ubuntu/code/IsaacLab-Arena-tasks/tools/rr_sim2real/train_toaster_knob_dp.sh
```

The script activates `robodiff` itself (with the Conda nounset-hook workaround).
WandB must already be logged in; otherwise run `wandb login` in that environment.
It does not launch Isaac Sim, change dependencies, upload datasets, or train the
previous drawer/press datasets. Do not run simulation alongside training.

Order: USDcraft, Articraft, miniworkflow GPTSOL, miniworkflow Astra. Each uses its
own `<method>_turn_toaster_knob_dp.zarr` under `datasets/rr_sim2real`.
The existing `ur7e_press_toaster_image` task config is reused only as an identical
observation/action schema; `task.name` and `task.dataset_path` are overridden per
knob case. It does not load press-toaster data or a pretrained checkpoint.

Settings match the latest remote training handoff, not the earlier 200-epoch /
BS64 experiments: 80 epochs, train/val BS128, seed42, EMA, AdamW lr1e-4, cosine
schedule, ResNet18 + GroupNorm, 240x320 input, 216x288 random crop, horizon16,
2 observation steps, 8 action steps, 100 diffusion steps. Validation every10
epochs, sampling every20, checkpoints after epochs39 and79, top-k2, online
WandB project `rr_sim2real`. Dataloader workers remain4; these are unrelated to
the three simulation collection workers. Only one training process runs at a time.

Before training, every dataset is checked for a completed publication, matching
LeRobot/Zarr build IDs, 200 episodes, black fingertips, randomization, expected
array dimensions, finite numeric data and sample image decoding. Source/config
SHA-256 values must match the remote handoff. Local published Zarr is authoritative;
HF upload completion and tar availability are not prerequisites for local training.

Read-only validation (no GPU allocation or training):

```bash
bash /home/ubuntu/code/IsaacLab-Arena-tasks/tools/rr_sim2real/train_toaster_knob_dp.sh --check-only
```

Output: `/home/ubuntu/code/diffusion_policy/data/outputs/rr_toaster_knob_bs128_e80_<UTC timestamp>/`.
Each case has its own checkpoints, Hydra configuration, WandB logs, and console
log. Environment versions, script snapshot, code revision/patch and dataset build
IDs are retained. Failure stops the chain; the script verifies an epoch79
checkpoint exists before advancing to the next case. At least65 GiB free disk
and an idle GPU are required at launch. Each case writes approximately13 GiB of
checkpoints under the existing saving policy.

Every invocation starts a fresh training run. This script does **not** automatically
resume interrupted training; do not rerun blindly if earlier cases have finished.
Existing output directories are never overwritten.

2026-09-14: Bash syntax, dependency consistency, four pinned source hashes and
all four published datasets passed `--check-only`. No training was started during
script preparation.
