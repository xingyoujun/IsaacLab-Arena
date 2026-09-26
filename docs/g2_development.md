# G2 development and collection

G2 now uses the data_engine public native cuMotion planner and Arena environment action executor.
The only supported collection/validation CLI is `tools/data_collection/g2.py`.
The former cuRobo collectors, shell launchers and separate replay tools have been removed.
Active workers, dataset helpers and workcell tests live in `isaaclab_arena_cumotion/g2_collection/`.
Historical implementations remain in Git history, not in the active source tree.

## Local use

From the host checkout, discover and use its running container automatically:

```bash
tools/data_collection/run_g2.sh list
tools/data_collection/run_g2.sh preflight --task stack_bowls
tools/data_collection/run_g2.sh run --task stack_bowls --run-dir outputs/g2/stack_000
tools/data_collection/run_g2.sh run --task peg_into_sleeve --run-dir outputs/g2/peg_000
tools/data_collection/run_g2.sh run --task clean_workcell_table --run-dir outputs/g2/workcell_000
```

To verify portability using the same installed image but no legacy source mounts:

```bash
tools/data_collection/run_g2.sh --isolated run --task stack_bowls --run-dir outputs/g2/portable_stack_000
```

This temporary container mounts only the checkout, including its downloaded asset cache. It does not
mount the host datasets, `/tmp`, home directory or old USD wrapper cache, and is removed on exit.
Run it serially with other GPU simulations; no Dockerfile, image rebuild or existing container changes are needed.
A fresh temporary container recompiles RTX shaders on its first camera launch, which can take minutes.
Use the normal launcher for repeated collection so the installed container retains its runtime caches.

Inside the container, invoke `/isaac-sim/python.sh tools/data_collection/g2.py` with the same arguments.
Set `PYTHONPATH` to this checkout and its Isaac Lab source, as with other Arena runtime commands.
`--assets PATH` chooses a complete downloaded/release-candidate bundle; `--device cuda:0` is the default.
The entrypoint also compares Git task, camera and motion-setting snapshots with the selected HF manifest.
CPU simulation and fallback to GenieSim or `/datasets/...` planning assets are rejected by this entrypoint.
One process owns one simulation; run attempts serially. A seed is recorded, but current G2 task layouts
are fixed qualification layouts: varying `--seed` alone does not create randomized scenes.
Arena geometry placement remains enabled to initialize the registered object poses. G2 excludes
only the optional cuRobo placement IK check; reachability is evaluated by native cuMotion during
execution. Arena loads the optional cuRobo validator only when enabled, and G2 rejects a legacy
planner import. An explicitly required cuRobo check fails configuration instead of being dropped.
The peg starts at `(0.05, -0.16)` m to clear the initial hand envelope; the original peg/sleeve
geometry and SDF collision representation are retained. A tilted peg after settling fails before planning.

`run` performs collection, independent raw audit, full three-camera replay and LeRobot export.
The same stages can be invoked as `collect`, `validate`, `render`, `export` with `--run-dir`.
`validate --official-loader` additionally checks first/middle/last samples with an installed LeRobot v2.1 loader.
Only collection needs `--task`; later stages read the saved task and asset hash. Changed task definitions
or asset releases cannot silently be used to process an older recording. Use its matching source version.
Existing raw, render and export destinations are never overwritten. Failed attempts stay local for diagnosis.

Each run has `run.json`, `environment.json`, stage logs, `raw/`, `validation.json`, `render/`, and `lerobot/`.
Raw success is checked before export; videos must match the raw SHA-256, all frame indices and camera sizes.
Rendering follows the shared `arena.transitions.v1` pre-step convention. Physics microstep rendering is
bounded, not mathematically exact; drift and visual-review requirements remain in the report.
Simulator, cameras and NVENC run on GPU. Python/I/O and the native planner's device ownership are separate;
this workflow does not claim that all processing is CUDA. `validation.json` records a separate
`runtime_device_audit`, including any PhysX CPU collision fallback warnings. The current workcell
bin mesh triggers such a fallback for a thin convex piece; task success does not certify CUDA-only
collision processing. Changing its collision asset requires a separate physics qualification.

## Assets and HF publication

Git owns code, task semantics, calibration and planning settings. The HF dataset
`xingyoujun/USDCraft-Scene` owns versioned robot/object USD dependencies and configuration snapshots.
Do not place raw HDF5, training data, output videos or model checkpoints in the asset release.
Raw task data stays on its collection machine; syncing it is not part of development synchronization.

The default local cache remains `local_assets/USDCraft-Scene`. A published immutable revision and its
manifest hash live in `tools/usdcraft_scene/release.json`; runtime never downloads implicitly.
To refresh a complete bundle's Git-owned task, motion and calibration snapshots (inside the container):

```bash
PYTHONPATH=. /isaac-sim/python.sh tools/usdcraft_scene/stage_release.py \
  --source local_assets/USDCraft-Scene --output local_assets/releases/g2_next
/isaac-sim/python.sh tools/usdcraft_scene/manage.py verify local_assets/releases/g2_next/payload
/isaac-sim/python.sh tools/usdcraft_scene/manage.py audit-usd local_assets/releases/g2_next/payload
```

The sibling `release-plan.json` pins the expected remote parent, full manifest hash, task IDs and file count.
Staging does not alter `release.json`. Publish an authorized, validated candidate with:

```bash
/isaac-sim/python.sh tools/usdcraft_scene/manage.py upload local_assets/releases/g2_next/payload \
  --release-plan local_assets/releases/g2_next/release-plan.json
```

The uploader refuses stale remote parents and changed candidates. Only a successful upload records the
returned immutable HF commit in `release.json`. Other machines download that pinned release with
`manage.py download`, verify it, and run `g2.py preflight`. Use a new destination if a previous download
has a different manifest; never overwrite an asset version used by an active collection process.
Robot, planning, room and task object assets must resolve through HF manifest IDs. Git contains no
G2 room or aluminum-block USD copies, and G2 does not fall back to historical planner/robot mounts.

## Add a task

1. Add an environment using the registered G2 embodiment and stable manifest asset IDs.
2. Extend `isaaclab_arena_cumotion/g2_collection/tasks.json` with the environment, initial configuration,
   instruction, required assets, recipe and settling policy. Record any layout change in this registry;
   HF scene snapshots are regenerated from it.
3. Add a task recipe using the shared Session and native `CumotionArmPlanner` / `EnvActionExecutor`.
   Define the new task objects in the session collision world as well. Keep collision exclusions limited to intended contact objects; physics remains active.
4. Add an independent raw-state terminal audit to `g2_collection/validation.py`, alongside the online predicate.
5. Stage updated scene/configuration snapshots, verify asset closure, then run the full unified pipeline.
   Inspect all three views before treating a pilot success as permission for large-scale collection.

G2 and Pine WM share asset resolution, public planning/execution and transition alignment. Their robot
joint dimensions, camera calibrations, task recipes and task-specific success checks remain distinct.
G2 changes do not change Pine WM's task review/qualification state.

## Local qualification (2026-09-26)

The integration checkout is `/home/ubuntu/code/IsaacLab-Arena_g2-integration`.
Each task completed one new fixed-layout pilot through `g2.py run` in an isolated container
with only this checkout mounted. The workcell and peg runs shared that temporary container's newly
compiled shader cache. No legacy dataset, host home or temporary asset directory was mounted.
All three runs report `legacy_planner_imported: false`.

Every pilot passed independent raw-state checks, full three-camera replay, LeRobot export and
first/middle/last sample loading with LeRobot 0.3.3 (v2.1 data format). The same three image indices
were visually reviewed. These are pilot qualifications, not a large-batch success-rate measurement.

| Task | Frames per camera | Camera-chain drift (mrad) | Root-position drift (mm) | Replay seconds, excluding startup | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `stack_bowls` | 1,040 | 0.266 | 0.045 | 81.3 | [portable_stack_007](../outputs/g2_framework_20260926/portable_stack_007/validation.json) |
| `peg_into_sleeve` | 555 | 0.013 | 0.044 | 91.4 | [portable_peg_001](../outputs/g2_framework_20260926/portable_peg_001/validation.json) |
| `clean_workcell_table` | 1,843 | 0.295 | 0.206 | 172.3 | [portable_workcell_001](../outputs/g2_framework_20260926/portable_workcell_001/validation.json) |

All videos are 15 FPS: head 640×400, each wrist 640×528, with CUDA camera buffers and NVENC encoding.
The canonical export has a 60-value state and 16-value action. The full evidence index is
[`qualification.json`](../outputs/g2_framework_20260926/qualification.json).
There were 55 passing targeted tests; the full Arena three-phase suite was not run.

The original pilot candidate and its evidence remain under ignored local output/release directories.
The current published asset revision is pinned in `tools/usdcraft_scene/release.json`; publication and
cleanup verification are documented in [the release record](g2_native_release_20260926.md).

Strict CUDA-only execution remains unqualified: the workcell's shared bin mesh emits a PhysX CPU
collision fallback warning, and native cuMotion's planning device is not independently verified.
This limitation is explicit in every `validation.json`; successful task/data checks do not conceal it.
No historical raw dataset is migrated by the development workflow. Git publication and HF asset publication
are independent operations.
