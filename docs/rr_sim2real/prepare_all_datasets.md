# Complete all eight black-fingertip datasets

The source-host inventory has four complete 200-demo datasets with old gripper
appearance and four qualified cases without a full dataset. Training is stopped.
The new master pipeline has been prepared, **not launched as a full GPU batch**.

## One command in tmux

```bash
bash /home/ubuntu/code/IsaacLab-Arena-tasks/isaaclab_arena_cumotion/scripts/prepare_rr_sim2real_all.sh --run
```

Run without `--run` for a read-only plan. The shell launcher works from any
directory and sets up the native source-host environment in its Python child.
No installation, Docker, Hugging Face upload or training is started. Detach
tmux with Ctrl-b then d. The pipeline remains in that tmux session; this is not
a service or cron job.

| Dataset | Action |
| --- | --- |
| usdcraft_open_drawer | Reuse 200 truncated states, black/randomized replay |
| usdcraft_press_toaster | Reuse 200 states, black/randomized replay |
| miniworkflow_gptsol_open_drawer | Reuse 200 states, black/randomized replay |
| articraft_open_drawer | Reuse 200 scaled-asset states, black/randomized replay |
| miniworkflow_astra_open_drawer | Collect 200 successes, black/randomized replay |
| articraft_press_toaster | Collect 200 successes, black/randomized replay |
| miniworkflow_gptsol_press_toaster | Collect 200 successes, black/randomized replay |
| miniworkflow_astra_press_toaster | Collect 200 successes, black/randomized replay |

Every row then runs LeRobot conversion, EEF labels, DP Zarr conversion and
validation. The final result is 8 x 200 = 1600 demos across eight LeRobot
datasets and eight Zarr siblings. Existing 800 trajectories are reused; 800
new demonstrations are collected. Qualification/smoke demos are not mixed into
the collection. At most two simulation workers run concurrently; cases execute
sequentially. Missing-case batches use eight attempts per worker, seed 0 onward,
closed-gripper ending, no retreat, and the already-qualified scaled/wrapped assets.

Camera replay uses seed-0 background/HDRI/material randomization and 0–6
non-colliding distractors, D435 640x480 at 15 fps, and the verified black finger
surface bindings. DP images stay 320x240. Existing source trajectories are not
replanned. Numeric filename versions are not added to canonical exports.

## Resume, logs, publication and disk

All work lives in `datasets/rr_sim2real_aux/black_gripper_all/<dataset>/`:

- `plan.json`: pinned parameters, code/assets/HDRI/source hashes.
- `run_*/worker_*/collect.log`: collection progress/failures.
- `logs/render_*.log`, `convert.log`, `eef.log`, `zarr.log`: stage logs.
- `<dataset>.hdf5` and camera sidecars: exact merged states and replay outputs.
- `converted.json`, `zarr.json`, `publication.json`, `complete.json`: stage checkpoints.

Re-run the **same command** after interruption. Complete workers/demos/stages
are reused; unreadable interrupted worker files are retained and skipped. A
collection round with zero valid progress stops for diagnosis instead of
silently looping. Failed renders stop and can resume; `.part.mp4` files are not
treated as completed videos. Do not edit task code/assets/HDRIs while this runs:
resume refuses changed pinned inputs. Increase `--max-rounds` only if the
configured attempt limit is exhausted; defaults allow 100 rounds per new case.

Only validated pairs replace `datasets/rr_sim2real/<dataset>` and
`<dataset>_dp.zarr`. Old exports are moved, not deleted, into
`black_gripper_all/previous_datasets/<transaction>/`; these remain recoverable.
Publication is journaled and resumes after interruption between the two moves.
There is **not** an atomic filesystem transaction across both directories:
do not train or upload from the final root until the script finishes. Markers
must not be edited by hand. No preview/log/raw/smoke files enter the HF root.

The script rejects competing GPU compute jobs at startup, takes orchestration
and publication locks, and requires 40 GiB free (source host currently has about
180 GiB). Expected extra persistent space is roughly 20–40 GiB, including
staging, new Zarrs and backups; interrupted attempts can increase this. It does
not stop other jobs, delete old datasets or automatically reclaim disk space.

Optional one-case smoke (separate workspace, no publication when target <200):

```bash
bash /home/ubuntu/code/IsaacLab-Arena-tasks/isaaclab_arena_cumotion/scripts/prepare_rr_sim2real_all.sh \
  --run --target 1 --only articraft_press_toaster \
  --work-root /home/ubuntu/playground/datasets/rr_sim2real_aux/black_gripper_one_demo
```

This is optional, not an extra command required for the full run. `--only` also
supports resuming selected cases at target 200. Keep the original settings.

## Timing and verification boundary

Historical source-host measurements: 200 Articraft drawers took about 1 h 46 m
to collect using two workers; their randomized replay took about 37 m and Zarr
conversion about 22 m. The original 200-toaster collection took about 3 h, with
about 43 m replay. Four new cases therefore need roughly 10–13 h collection;
eight render/convert passes add roughly 8–11 h plus validation/startup overhead.
Budget **18–24 h in favorable conditions, 24–30 h with retries/longer planning**.
This is an estimate, not a measured eight-case run. Do not share the GPU with
training. Network asset-cache misses and unsuccessful attempts increase time.

Validated before handoff: read-only eight-case plan, shell syntax, all four
200-demo source inventories, Python style checks, and CPU regression tests for
negative/rest-offset joints, residual-velocity rejection, source metadata reuse,
appearance-cache rejection, render flags/resume, and interrupted pair publication.
Existing converters and each asset's GPU cuMotion tests were verified separately.
The entire new 1600-demo pipeline has not been executed by the agent.
