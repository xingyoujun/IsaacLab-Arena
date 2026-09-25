# RR upstream migration on Isaac Sim 6.0.1

Date: 2026-09-25. This migration follows the user's approval after the
[read-only assessment](upstream_audit_20260925.md). That assessment describes
the pre-migration state, not the current checkout.

## Version and scope

- Saved RR baseline: `2c42999b7` (`Preserve the completed RR sim2real baseline`).
- Merged upstream target: `aa36f191d89b1ea65fbba62b3350d6fc7d0e9b9f`.
- Branch: `chuanruiz/rr_sim2real`; local commits only, no push or PR.
- Isaac Sim remains `6.0.1.0`. The converter pin remains `0.2.0`.
  `uv lock` updated dependency resolution only; no sync, install, or upgrade was
  performed in the shared virtual environment.
- Docker, CI workflows, pre-commit configuration, and submodule gitlinks retain
  their pre-merge contents. The sibling checkout was not modified.
- Assets, camera calibration, object dimensions, task thresholds, checkpoints,
  and formal datasets were not changed for this migration. Only Articraft remains
  size-normalized, as before. PhysX remains the default backend.

This is a merge of a fixed main revision, not a claim that every new upstream
feature is compatible with Sim 6.0.1. CAP, Newton, deformable examples, and the
inherited Agibot tasks are outside the UR7e validation scope. Fresh-machine
installation and multi-rebuild experiments were not qualified.

## Compatibility decisions

1. Keep the old recorder import module for UR7e's custom seven-joint training
   export. Upstream evaluation trajectories use the new recorder separately.
2. Adapt `IdleTask` to `TaskTerminationCfg`. Use `Ur7eOpenableTask` for the three
   RR task families, with the original asset success predicate and timeout.
   Do not inherit the upstream door task's additional 5%-travel prerequisite:
   knob success is absolute rotation beyond 2 degrees in either direction.
3. Run environment configuration callbacks once, after backend selection, and
   retain the assertion that the callback returns a configuration.
4. Retain local predicate compatibility helpers and rigid-body velocity guards
   alongside upstream spatial predicates and deformable reset events.
5. Keep local deployment/runtime pins instead of adopting the Sim 6.1 Docker and
   CI changes. No environment upgrade is required to run the qualified RR paths.

The evaluation orchestrator now supports `--only`, `--workers`,
`--episodes-per-worker`, and `--record-trajectories`. Its default 12-case,
20-episodes-per-case protocol is unchanged; see [evaluation usage](eval_all_dp.md).

## Validation artifacts

All migration artifacts are isolated under
`outputs/rr_main_migration_20260925/`, excluded from git. They do not replace
`outputs/dp_eval12_20260916_retry1/` or the formal 200-demo datasets.

- `pipeline.log` and `pipeline/CASE/complete.json`: one-demo, all-case data
  pipeline checks. `published: false` means no formal dataset was overwritten.
- `initial_drawer/`, `fresh_gptsol_drawer/`, `fresh_articraft_drawer/`, and
  `fresh_usdcraft_press/`: fresh collection of the four cases for which the bulk
  pipeline replays existing source HDF5; other eight cases collect fresh in the
  bulk pipeline itself.
- `unit_tests_retry.log`: 24 orchestration/execution/timing tests passed.
- `contract_tests.log`: the UR7e termination/recording contract test passed,
  including all four knob assets at zero, exactly plus/minus 2 degrees, and
  just beyond both thresholds.
- `precommit_retry.log`: all-file host hooks passed during migration; hooks
  are rerun before the merge commit.

## Runtime results

- All twelve method/task combinations exported a fresh successful cuMotion
  demo (four standalone checks plus eight collected by the bulk pipeline).
- All twelve completed one-demo randomized background/distractor rendering,
  LeRobot export, EEF labels, DP Zarr conversion and validation. All completion
  records have `published: false`, 640 x 480 video and 15 FPS. Four existing HDF5
  sources were replayed successfully, checking backward compatibility too.
- Standalone fresh recordings contain seven-dimensional `joint_pos_target`
  arrays and `success: true`: USDcraft drawer 264 frames, GPTSOL drawer 265,
  Articraft drawer 268, USDcraft press 256. Converted frame counts can differ
  because the training pipeline trims terminal frames.
- `idle_experiment/` completed two timeout/reset episodes with four camera
  videos, canonical result/timing JSON, an HTML report and two HDF5 demos.
  Recorded kinematics include TCP and both fingertips, positions, orientations
  and velocities. `black_gripper_check.png` is a visual spot check from the
  randomized USDcraft drawer video, not a replacement for all-asset validation.

- `dp_smoke/` completed two consecutive episodes each of USDcraft drawer,
  press and knob (one worker, seed 42): each case scored 1/2. The ordered
  success/failure outcomes match the first two episodes of the old seed-42
  evaluation. Episode lengths: drawer 405/450 (old 407/450), press 420/149
  (old 420/149), knob 300/160 (old 300/159); this is a compatibility spot check, not exact trajectory
  equivalence. All six episodes have dual-camera videos, reports and diagnostic
  HDF5 trajectories. The timeout values remain 30/28/20 simulated seconds.

- `dp_dual_smoke/` completed GPTSOL knob with two independent workers, one
  episode each (seeds 42/1042), scoring 1/2. Both workers produced their own
  reports, two camera clips and trajectory HDF5; no port or file-lock collision
  occurred. A policy failure at timeout is a valid recorded episode, not an
  infrastructure failure.

All eight DP smoke episodes completed without execution errors. These checks
qualify migration paths; they do not estimate a new benchmark success rate or
claim that the full upstream three-phase regression suite passed. The previous
240-episode results remain the comparison baseline.
