# Twelve-checkpoint DP evaluation

`tools/rr_sim2real/eval_all_dp.py` runs the twelve method/task checkpoints in
`/home/ubuntu/code/diffusion_policy/ckpts`. Methods: USDcraft, Articraft,
miniworkflow GPTSOL, miniworkflow Astra. Task order: open drawer, press toaster,
turn toaster knob. Cases run serially; each case runs two independent native
simulators and two independent DP servers, ten episodes each (240 total).

Use robodiff Python to launch the orchestrator in tmux. It starts the simulator
with this worktree's `.venv` and explicit `PYTHONPATH`. No conda activation is
necessary. Reserve the GPU for this batch: do not start training concurrently.

```bash
cd /home/ubuntu/code/IsaacLab-Arena-tasks
/home/ubuntu/miniconda3/envs/robodiff/bin/python tools/rr_sim2real/eval_all_dp.py \
  --output /tmp/rr_dp_eval_plan --plan-only
# In tmux, choose a NEW output directory:
/home/ubuntu/miniconda3/envs/robodiff/bin/python -u tools/rr_sim2real/eval_all_dp.py \
  --output "$PWD/outputs/dp_eval12_NEW_RUN"
```

Settings: environment seeds 42/1042, one env per worker, CUDA 0, existing policy loader
(EMA when configured), 100 inference steps, checkpoint-defined two observation
frames/eight action steps. Simulator assets retain the shared black gripper,
randomized task-object poses and standard evaluation background. This is not a
new background/distractor-randomized evaluation protocol. The policy observation
and action conversion are unchanged from previous DP evaluations. Each policy
process has independent state; its workspace applies the checkpoint's training
seed on load (not the environment seed).

Drawer timeout is 30 simulated seconds. Press and knob timeouts are each case's
longest dataset `meta/episodes.jsonl` length divided by `meta/info.json` FPS,
multiplied by 1.5 and rounded UP to a whole second. The run manifest records the
source lengths and resulting limits. Timeout does not force successful episodes
to continue: the existing task success termination is unchanged.

Each worker uses ports 5760/5761 respectively and stores its YAML, logs, PIDs,
policy metadata, native `arena_experiment_result.json`, episode JSONL, report
`index.html`, and both observation-camera MP4 streams. Completed cases produce
`CASE/results.json`; root `summary.csv` is updated after each case and root
`manifest.json` tracks pending/running/completed/execution_failed cases.
Each worker must produce exactly ten result episodes and at least twenty
nonempty, ffprobe-readable camera clips before a case is marked completed.
Run names include the batch, case and worker, which also isolates the native
metric recorder's temporary HDF5 filename; sharing a run name across concurrent
simulators causes a file-lock error even when report directories are separate.
Infrastructure errors stop the batch and preserve partial artifacts, rather
than counting missing episodes as policy failures. A three-hour wall-time
safety limit per case detects hangs independently of simulated episode time.

The orchestrator locks `outputs/.rr_dp_eval_all.lock` against duplicate batches,
checks that ports are free and disk free space exceeds 20 GiB, verifies input
checkpoint size/mtime has not changed, and terminates only its own subprocess
groups on exit/SIGTERM. Never overwrite a prior run directory to retry.

Initial launch on 2026-09-16:

- tmux session: `rr_dp_eval12_20260916`
- output: `outputs/dp_eval12_20260916_retry1`
- master log: `outputs/dp_eval12_20260916_retry1/master.log`

The first startup (`outputs/dp_eval12_20260916_043215`) stopped on a shared metric
HDF5 file-lock conflict before completing any episodes. It is diagnostic only,
not part of the evaluation denominator. The retry isolates native run names.

Inspect the manifest/logs before assuming completion. Disconnecting SSH does not
terminate the tmux session; stopping the orchestrator intentionally does stop its
workers. Existing unrelated tmux sessions are not modified.
