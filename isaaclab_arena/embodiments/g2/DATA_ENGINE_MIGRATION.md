# G2 integration with data_engine

Date: 2026-09-25. This is a local integration of
`chuanruiz/data_engine@2b3f5f0428338cf5daad606fd4f03fee102bafd8`, which contains
upstream `main@aa36f191d89b1ea65fbba62b3350d6fc7d0e9b9f`.
The saved G2 baseline is `eb616ba` on `team/g2/dev`. No remote push was performed.

## Workspace and runtime

Integration checkout: `/home/ubuntu/code/IsaacLab-Arena_g2-integration`, branch
`xingyoujun/feature/g2-data-engine`. The original checkout remains on its saved
G2 baseline, with its original submodule and containers intact.

The integration uses its own Docker container with the existing
`isaaclab_arena:curobo` image and the incoming IsaacLab gitlink
`bb0c8e1b9af381bf13064ec3303e17db79e4b6ef`. Runtime: Isaac Sim
`6.0.1-rc.7+release.42383.32955d8d.gl`, torch `2.10.0+cu128`, Warp `1.13.0`.
This validates the existing Docker runtime, not a fresh installation of the
incoming uv lockfile or all incoming environments. No shared Python environment
or Docker image was upgraded.

Discover the container by the integration checkout's mount:

```bash
cd /home/ubuntu/code/IsaacLab-Arena_g2-integration
ARENA_CONTAINER=$(docker ps --filter "volume=$PWD" --format '{{.Names}}' | head -1)
```

Run as the host user. Non-interactive commands use `/isaac-sim/python.sh` and
`PYTHONPATH=/workspaces/isaaclab_arena:/workspaces/isaaclab_arena/submodules/IsaacLab/source/isaaclab`.
`ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1` enables Arena's existing bounded Kit
shutdown path for test and collection subprocesses.

## Compatibility changes

- Update G2's IsaacLab task imports to the GA `contrib` paths.
- Keep the G2 USD package-data entry alongside upstream's configuration-data rules.
- Use `CompositeTaskBase(subtasks_are_sequential=True)` and a G2 placement task
  preserving the old contact-force, 0.1 m/s speed, and strict root-distance checks.
  Both placements must remain valid at completion. Generic upstream placement
  behavior is unchanged.
- Return `TaskTerminationCfg` for peg insertion while keeping the 1 mm radial,
  3 mm axial and 1 degree alignment tolerances and the drop condition.
- Let the managed success term update/reset progress during collection, without
  triggering an automatic episode reset before release and retreat. The peg
  collector still checks the live insertion predicate for fifteen final steps.
- Use the canonical Arena recording module and `SimulationAppContext`. Keep the
  old `--headless` command-line option accepted by G2 scripts.
- Refresh offline replay through GA's `reset_scene_state_cadence()` API.
- Restore the Warp torch-interop namespace required by the pinned cuRobo version.
  The beta IsaacLab Mimic package used to provide this compatibility layer.
- Report both G2 TCP frame transformers to upstream trajectory recording.
  Check solver limits through GA's articulation data in the physical regression test.
- Defer the CAP USB-C Newton manager import until its task is configured. This
  keeps optional physics dependencies out of environment registration, including
  the G2 teleoperation callback; the USB-C manager's original public import path
  and cleanup behavior are preserved.

The robot USD, gains, gripper limits, TCP offsets, camera calibration, physics
backend, action ordering and LeRobot v2.1 schema are preserved. This integration
keeps G2's cuRobo collectors; it does not substitute the separate Agibot/UR7e
cuMotion descriptions or their data schemas.

## Validation artifacts

All new raw, exported data and diagnostics are isolated under
`outputs/g2_data_engine_migration_20260925/` (git-ignored). Formal datasets under
`/datasets/agibot_dataset_v1*` are not overwritten. Old replay checks use copies
of the first successful raw episode from each task.

G2 regression checks cover registration, initial pose, action/observation sizes,
both arms, drive limits, IK, collisions, gripper motion, physical cube grasp/lift,
dual keyboard control, bowl reachability, insertion geometry/physics/reset,
deferred collection termination, and sequential placement threshold/reset semantics.

### Regression results and runtime limitation

- All 17 G2 tests passed: five embodiment tests in `g2-integration-core.log`,
  then eleven physics/task tests in `g2-integration-core-retry.log` after adapting
  the drive-limit test to the GA articulation API, plus the new teleoperation CLI
  registration regression in `g2-cli-registration.log`.
- Fresh cuRobo core collection passed for one stack and one peg episode,
  seed 10000 with XY noise of +/- 0.02 m. Bowl lifts were 0.15454 m and 0.15420 m.
  The peg episode has 1133 frames; final radial error is 0.23533 mm and final
  axial error is below 0.00003 mm.
- Host `pre-commit run --all-files` passed.
- Both new episodes and both old raw episodes completed offline replay and
  LeRobot v2.1 export: 4682 total frames and 12 videos, all at 15 FPS. Every raw
  data array equals its exported Parquet array; frame indices, timestamps,
  artifact hashes and decoded video frame counts passed. Maximum TCP position
  discrepancy is 0 m for all four episodes. Head/wrist video frames were also
  visually inspected. See `export_validation.json` and `previews/`.
  Raw collection is under `fresh_raw_v3/`; the GA renderer API correction was
  verified in separate `fresh_replay/` and `fresh_export/` directories, preserving
  the original collection provenance. Old sample copies use `old_replay/` and
  `old_export/`. Each replay archives the renderer source it actually used.
- Re-executing the old bowl episode's 1208 joint actions with physics (restoring
  only its initial state) reached final task success with no failure term.
  Maximum TCP position deviation was 0.08858 mm on the right and 0.09576 mm on
  the left. See `old_stack_physics_replay.json` and `g2-old-stack-physics.log`.
- The complete pytest suite is **not green**. Standard Phase 1 first hit a
  native Kit startup crash after test collection imported `pxr` through the
  incoming RR environment registration. Its first test passed in isolation.
  A temporary runner that starts Kit before collection got 13 passes, then
  `test_all_environments.py::test_all_environments_have_default_args` failed
  importing `newton.ModelFlags` through CAP USB-C environment registration.
  The same failure reproduced in isolation. After deferring the runtime-only
  manager imports, this registration test and the G2 teleoperation callback both
  passed (`g2-registration-fixed.log`, `g2-cli-registration.log`). The image's Newton does
  not match the incoming `release-1.5` dependency; the new lockfile also requires
  Warp 1.16.0, whereas this cuRobo image has Warp 1.13.0.
- Re-running Phase 1 with Kit initialized before collection after the registration
  fix reached **151 passed, 1 failed, 83 deselected**. The failure is
  `test_asset_registry.py::test_all_assets_in_registry`: the incoming UR7e drawer
  asset requires `/home/ubuntu/playground/rr_ur/usdcraft_drawer_arena.usda`, absent
  on this host and in the container. The same missing-asset failure reproduced
  in isolation. See `g2-phase1-after-registration-fix.log` and
  `g2-all-assets-isolated.log`.
- Phase 2 and Phase 3 of the full suite remain unrun after the Phase 1 failure.
  Validating all incoming environments requires a separate image built from
  the incoming dependency set and the RR task assets (including their referenced
  USD layers), followed by all three phases and another G2 cuRobo regression run.
  The standard pytest collection/startup order also needs correction before a
  normal full-suite invocation can replace the temporary pre-initialization runner.
  These results do not establish that the whole fork works in the existing image.

Do not resume an existing production collection directory with changed sources:
the collection scripts intentionally reject mismatched provenance hashes. Use a
new run directory for the integrated version; retain the baseline checkout for
old runs. The migration's smoke episodes are compatibility checks, not a new
collection-success-rate measurement.
