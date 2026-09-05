# Keeping up with upstream Arena

Arena's `main` moves fast (48 commits in the three weeks after our baseline). Our work must stay
mergeable with it, and a sync must be a routine, not a rescue. This page has two parts: the
standing procedure, and the impact snapshot of the delta that was open on 2026-09-05.

## Principles that keep us mergeable

1. **Agibot-specific code lives in our own files.** Environments, tasks, the shared table-top
   module, the cuMotion package, the probes, `local_objects.py`, and the control-stack utilities are
   all files upstream does not touch, so they never conflict. Keep it that way: a new feature goes
   into a new file or one of ours, not into a shared upstream file.
2. **Edits to shared upstream files are the conflict surface.** Today that surface is small:
   `terms/events.py` (`_velocity_is_writable`, `reset_joint_position_and_velocity_to_defaults`),
   `tasks/predicates/spatial.py` (our predicates appended), `assets/device_library.py` and
   `assets/retargeter_library.py` (dual-arm keyboard registration), `assets/registries.py`
   (import of `local_objects`), `arena_env_builder.py` (one assert), `embodiments/agibot/agibot.py`
   (the default config), `convert_hdf5_to_lerobot.py`, `pyproject.toml`, `docs/conf.py`. Every one
   of these is either upstreamable or movable into our files; do one of the two whenever a sync
   conflicts there.
3. **Upstream what is generic.** The env-callback assert, the kinematic-aware velocity write, the
   robot-only reset event, the dual-arm keyboard device and the three Agibot bug fixes are Arena
   improvements, not team code. Once merged upstream they stop being our diff.
4. **Small commits on a branch, rebased onto `origin/main`.** Our history is currently four large
   "team baseline" commits plus an uncommitted working tree; that is the hardest shape to rebase.
   Commit the working tree in topical pieces (control stack, shared module, tasks, probes, docs),
   keep a `team/dev` branch rebased on `origin/main`, and open PRs from it.
5. **Sync on a schedule, not on demand.** Weekly `git fetch` + trial merge (below) takes ten
   minutes and turns a 48-commit surprise into a few small ones.

## The procedure

```bash
# 1. Fetch and measure the delta (nothing changes locally).
git fetch origin
MB=$(git merge-base main origin/main)
git log --oneline main..origin/main | wc -l
git diff --stat $MB origin/main | tail -1
comm -12 <(git diff --name-only $MB origin/main | sort) \
         <( (git diff --name-only $MB main; git status --short | awk '{print $NF}') | sort -u )

# 2. Trial merge in a scratch worktree (the real tree is untouched).
git worktree add --detach /tmp/arena_merge_trial main
git -C /tmp/arena_merge_trial merge --no-commit --no-ff origin/main
git -C /tmp/arena_merge_trial diff --name-only --diff-filter=U      # the conflicts
git worktree remove --force /tmp/arena_merge_trial

# 3. Read the upstream commits that touch what we depend on before merging:
#    submodules/IsaacLab pointer, embodiments/agibot, environments/, terms/, predicates/,
#    assets/*library.py, pyproject.toml / uv.lock, anything removed under scripts/.
git diff $MB origin/main -- submodules/IsaacLab pyproject.toml | head
git diff --stat $MB origin/main -- uv.lock
```

Then, in this order and only when the box is idle (no teleop or collection running):

4. Commit the working tree (topical commits) so the merge has a clean base.
5. `git merge origin/main`, resolve conflicts, keep both sides where both are additive.
6. Submodule: `git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive`
   (this host has no GitHub SSH key; the `.gitmodules` URLs are SSH).
7. Environment: if `uv.lock` changed, `uv sync` rebuilds `.venv` (tens of GB, needs network, takes
   a while). Do it in a second clone or at a quiet time; the running teleop session uses the old env.
8. Migrate our code for API changes (see the snapshot below for the current list).
9. Verify, in this order: `py_compile` of our files; the five-env zero-action smoke
   (`policy_runner.py --policy_type zero_action --num_steps 30 <env>`); the four probes on one env
   each; the config/registry pytest subset; then Phase 1. Then one teleop session through noVNC,
   because the Kit viewport path is where Isaac Lab upgrades bite.
10. Update `docs/agibot/` for anything the sync changed (flags, commands, versions).

## Impact snapshot: `origin/main` on 2026-09-05

Baseline `ec5a9773e` (2026-08-13) -> `origin/main` `af3f24b05`: 48 commits, 286 files,
+9361 / -6083 lines. Trial merge of our committed work: **3 content conflicts, all small**
(`spatial.py` import block: keep both; `terms/events.py`: our helper plus upstream's new
`ResetBackgroundPhysics` classes, keep both; `convert_hdf5_to_lerobot.py`: take upstream's
`imageio.mimwrite`, drop our torchvision fallback). None of our untracked files collide with new
upstream paths. `agibot.py`, `arena_env_builder.py`, `retargeter_library.py`, `pyproject.toml`,
`docs/conf.py` merge automatically.

### The big one: Isaac Lab 3.0 GA (#1166, `3e60147a8`)

Submodule `af1bab4dc` (3.0.0-beta2) -> `bb0c8e1b9` (`release/3.0.0`, 813 commits), Newton 1.2 ->
1.5, warp 1.13 -> 1.16, isaacteleop 1.3.131 -> 1.4.126rc1 pinned; Isaac Sim stays 6.0.1 and torch
stays 2.11.0+cu128 (so cuMotion's bundled extension should be unchanged). `uv.lock` changed by
~4000 lines: the environment must be rebuilt.

What it changes for us, verified against the GA tree:

| change | our exposure | action at sync |
| --- | --- | --- |
| `isaaclab_tasks.manager_based.manipulation.*` -> `isaaclab_tasks.contrib.*` | two imports in `agibot.py` | auto-merged by upstream's edit; nothing to do |
| `ImplicitActuatorCfg.effort_limit_sim` / `velocity_limit_sim` renamed `joint_effort_limit` / `joint_velocity_limit` (old names kept as **deprecated aliases until 4.0**) | `agibot.py` left-gripper copy (3 lines), `apply_arm_gains` in `agibot_tabletop_common.py` | still works with deprecation warnings; migrate to the new names and re-verify the gripper copy lands (print the actuator cfg after build) |
| `--headless` removed from `AppLauncher`; no `--viz` now means headless | every probe and driver passes `--headless`; docs and the collection script too | probes and drivers use `parse_known_args`, so the flag is ignored harmlessly; `policy_runner` via the Arena CLI will reject it. Drop `--headless` everywhere after the sync |
| `find_joints(..., as_proxy=True)`: `_joint_ids` is a warp array (binary action) or a torch tensor (joint action) | `RampedBinaryJointPositionAction`, `SmoothJointPositionAction`, and `probe_pinch.py` reads `term._joint_ids` | re-run the pinch probe and a teleop close; adapt `list(term._joint_ids)` |
| `RMPFlowAction` internals we subclass (`_processed_actions`, `_asset`, `_rmpflow_controller`, `_scale`, `process_actions`, `reset`) | `TargetHoldingRMPFlowAction`, `SurfaceGuardedRMPFlowAction` | unchanged in GA (file not modified); re-run the idle-drift check anyway |
| object pose access through `FrameView`; "torch/numpy imports after Kit start-up" | our modules import torch at module level but are only imported after the app starts; probes follow the AppLauncher-first pattern | none expected; smoke will tell |
| Known GA issue: Kit viewport can be blank (RTX per-env partition) | **teleop through noVNC** | run with `ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0` |
| `AGIBOT_A2D_CFG` in Isaac Lab: only the field renames, values unchanged | the frozen default | none |
| Lab's `record_demos.py` / `teleop_se3_agent.py` keep `--task`, `--teleop_device`, `--external_callback`, `--dataset_file`, `--num_demos`, `--num_success_steps`, `--step_hz` | our commands | none; drop `--headless` only |

### Other upstream changes that touch us

- **`merge_demos.py` removed (#1117)** -- our collection driver used
  `isaaclab_arena/scripts/imitation_learning/merge_demos.py`. Kept as
  `isaaclab_arena_cumotion/scripts/merge_demos.py`; the driver points at that copy.
- **Background physics reset for all library backgrounds (#1177)** -- new `ResetBackgroundPhysics`
  event on every `LibraryBackground` with nested physics roots. `robodojo_table` has only a
  collision API and the room has no physics roots, so nothing is reset; opt out with
  `reset_nested_physics=False` if a future background misbehaves. (#1179 made table backgrounds
  kinematic and #1185 reverted it: net zero.)
- **Force CPU + no Fabric after a stage rebuild (#1168)** -- applies to experiment runs that rebuild
  the stage; watch collection throughput if it triggers.
- **New geometric success predicates (#1132)** -- `object_bounds_center_over_destination`,
  `contact_force_is_upward_support`, `object_is_moving_slowly`; candidates to replace our frame-box
  and rest checks in tidy_workbench.
- **`--enable_cameras` moved into the Arena CLI group**; `patch_resolve_clone_plan_source` removed;
  `apply_arena_global_settings` + a warp CPU patch now run in `make_registered`.
- Kitchen environments renamed (`kitchenbench_*`), several new kitchen envs, docs restructured,
  uv wheel flavour dropped (only the from-source install remains, which is what this host uses).

### Recommended sequence for this delta

1. Commit the current working tree in topical commits (nothing to sync until it is committed).
2. Copy `merge_demos.py` into our package; update the collection driver (done 2026-09-05).
3. Merge `origin/main`, resolve the three conflicts, update the submodule over HTTPS.
4. Rebuild the environment (`uv sync`) when no teleop or collection is running.
5. Migrate the actuator field names, drop `--headless` from probes/drivers/docs, adapt
   `_joint_ids` reads.
6. Run the verification ladder in step 9 above, with one noVNC teleop session at the end
   (`ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0`).
7. Open the generic pieces as upstream PRs so the next sync is smaller.
