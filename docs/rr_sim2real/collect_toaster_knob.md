# Four-method toaster-knob datasets

Entry point: `isaaclab_arena_cumotion/scripts/prepare_rr_toaster_knob_datasets.sh`.
It selects only the four knob tasks; the existing drawer/press datasets are not
replaced. Default is a read-only plan; add `--run` to execute. Use the source-host
native `.venv` (the script supplies the correct worktree/Isaac Lab import paths).

```bash
cd /home/ubuntu/code/IsaacLab-Arena-tasks
# Check the plan:
bash isaaclab_arena_cumotion/scripts/prepare_rr_toaster_knob_datasets.sh
# Run in tmux; rerun the same command to resume:
bash isaaclab_arena_cumotion/scripts/prepare_rr_toaster_knob_datasets.sh --run
```

## Matched settings

- 200 successful records per method, 800 total, with success at absolute dial
  angle >2 degrees. Stop at the first successful recorded control frame.
- Same press-toaster placement band: x in [-0.22, -0.10] m, y in [0.10, 0.14] m,
  yaw in [-10, +10] degrees; Gaussian robot reset joint jitter std=0.03 radians.
- Both box supports and toast share a single sampled pose; support height and
  all object scales stay unchanged. Supports are kinematic rigid objects:
  immobile during a trajectory, movable on reset, and included in HDF5 states
  so replay preserves their sampled poses. The source `boxx.usdc` is unchanged.
- Black fingertips; original calibrated D435; same visual randomizer as the
  previous datasets: seed 0, `/home/ubuntu/playground/assets/skies` HDRIs,
  randomized lights/table/ground materials, up to six visual-only distractors,
  0.28 m keep-clear radius around recorded task objects. No extra physics
  obstacles are introduced by the visual randomizer.
- Record states first, then replay to randomized camera videos. LeRobot v2.1:
  640x480 at 15 Hz, 7 absolute joints, EEF 9D labels. DP Zarr: the existing
  320x240 image / 10-dimensional action conversion. This is the same two-stage
  recording/rendering and conversion contract as the previous tasks.

## Outputs and recovery

Final root: `/home/ubuntu/playground/datasets/rr_sim2real`:

```text
usdcraft_turn_toaster_knob/                 + usdcraft_turn_toaster_knob_dp.zarr/
articraft_turn_toaster_knob/                + articraft_turn_toaster_knob_dp.zarr/
miniworkflow_gptsol_turn_toaster_knob/      + miniworkflow_gptsol_turn_toaster_knob_dp.zarr/
miniworkflow_astra_turn_toaster_knob/       + miniworkflow_astra_turn_toaster_knob_dp.zarr/
```

Raw HDF5, logs, staging, replay sidecars, interrupted artifacts and progress
markers stay under `datasets/rr_sim2real_aux/toaster_knob_all/<task>/`, outside
the HF upload root. A case publishes only after its full 200-demo conversion
passes validation. Previous same-name exports, if present, are archived outside
the final root through the existing recoverable publication journal.

The orchestrator collects batches of eight attempts per worker, verifies genuine
successful end states, merges exactly 200 valid demos, then renders/converts.
Failed self-collision/IK attempts are not exported. Empty rounds retry with fresh
seeds; five consecutive empty rounds stop for diagnosis, without relaxing the
task or collision threshold. Resume reuses raw demos and rendering sidecars. Fingerprints
reject changes to task code/assets/settings that would mix experimental versions.
GPU processes must be stopped before launch/resume; the script does not kill
unrelated jobs. A 40 GiB free-disk floor is checked at startup and before each case.

## Worker counts

Default: **3 collection workers, 2 replay/render workers**, all on GPU 0. Cases
run sequentially; this does not spawn 12 simultaneous simulators. Collection
workers disable cameras and stagger startup by 20 seconds. Each worker reuses
its simulator for its eight attempts, avoiding per-episode Isaac startup.

```bash
# Reduce collection concurrency, retaining completed work:
bash isaaclab_arena_cumotion/scripts/prepare_rr_toaster_knob_datasets.sh --run --workers 2
# Three render workers are configurable, but are not the default:
bash isaaclab_arena_cumotion/scripts/prepare_rr_toaster_knob_datasets.sh --run --render-workers 3
```

Concurrency/stagger changes are allowed on resume for knob cases. Random seeds
reserve three worker slots per round, preventing seed reuse when reducing the
worker count. Extra VRAM permits more workers but does not imply linear speedup:
GPU compute, RTX rendering, CPU planning and disk encoding are shared resources.
The source host reports an L40S with 46,068 MiB VRAM, not a 40 GB card.

For an isolated end-to-end smoke run (never publishes to the final root):

```bash
bash isaaclab_arena_cumotion/scripts/prepare_rr_toaster_knob_datasets.sh --run \
  --target 3 --batch-size 1 --stagger 5 \
  --work-root /home/ubuntu/playground/datasets/rr_sim2real_aux/knob_my_smoke
```

The older `qualify_toaster_knobs.sh` remains a fixed-pose single-demo test, not
the bulk dataset command. See [task geometry and qualification](turn_toaster_knob.md).

## Source-host checks, 2026-09-13

Three simultaneous camera-free collectors were exercised. Sampled total VRAM
usage was 8,541 MiB with GPU utilization at 100%; this establishes headroom, not
a 2-vs-3 throughput improvement or a peak-memory bound. Two replay workers were
also exercised. No full-scale 800-demo job was launched during preparation.

End-to-end smoke outputs under
`rr_sim2real_aux/knob_pipeline_smoke_b_20260913/`:

| Method | Valid episodes | LeRobot/DP training frames |
| --- | ---: | ---: |
| USDcraft | 3 | 537 |
| Articraft | 3 | 586 |
| miniworkflow GPTSOL | 3 | 547 |
| miniworkflow Astra | 3 | 526 |

Astra rejected one self-colliding attempt and replaced it with a new seed. All
retained episodes passed final-frame threshold, reset, synchronized support pose,
placement-range, 7D action, randomization-sidecar, LeRobot and Zarr checks. The
first USDcraft smoke exposed a disabled lid joint being parsed as a self-joint
after making the support kinematic; the overlay now marks that unused joint
inactive, and later runs had no such error. Smoke artifacts are diagnostic and
must not be used to resume the production build across these code revisions.

The final source version also completed a separate one-episode full pipeline
under `rr_sim2real_aux/knob_resume_final_smoke_20260913/` (189 training frames).
Rerunning it with collection workers changed from 3 to 2 and render workers
from 2 to 1 correctly reported `complete, skipping`, without launching simulation
or duplicating data. The GPU was idle after all preparation checks finished.
