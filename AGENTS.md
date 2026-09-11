# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, OpenAI Codex, etc.) when working with code in this repository.

## Project

Isaac Lab-Arena is a composable environment-creation and policy-evaluation library for robotics simulation, built on Isaac Sim 6.0 and Isaac Lab 3.0 Beta. Status: alpha (`v0.2.x`); APIs are unstable. `main` is the active development branch.

## Skill library

Recurring multi-step workflows are captured as Agent Skills under `skills/`, grouped by audience. When a task matches a skill, prefer invoking it over re-deriving the procedure from this file.

The canonical skill sources live under `skills/developer/` and `skills/user/`. These folders describe the primary audience and validation owner; they do not restrict which workflows may reuse a skill. Codex discovers the skills through flat symlinks in `.agents/skills/`; Claude Code reads the same aliases through the committed `.claude/skills` symlink. Keep the canonical sources and flat discovery aliases synchronized when adding, removing, or renaming a skill.

Fresh-clone setup (run once):

```bash
pre-commit install    # on the host — registers git pre-commit hooks
```

## Docker environment

Commands that touch Isaac Sim or Arena's package code (tests, training, evaluation, runtime scripts) run inside the local repo clone's Docker container. The repo root is mounted at `/workspaces/isaaclab_arena`. Inside the container, `python` is aliased to `/isaac-sim/python.sh` — prefer the explicit path in `docker exec` invocations from outside the container, where the alias is not active.

Each clone gets its own container (shared image, per-clone name), so clones run in parallel. **Don't hardcode the container name** — use the `dev-container` skill to build, start, attach to, discover, or exec into the local clone's container.

Run as the host user, not root.

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && <command>"
```

Lint and format tooling (`pre-commit` and the hooks it runs — black, flake8, isort, pyupgrade, codespell) runs **on the host**.

## Repository layout

- `isaaclab_arena/` — core package: `tasks/`, `policy/`, `evaluation/`, `embodiments/`, `scene/`, `assets/`, `tests/`
- `isaaclab_arena_environments/`, `isaaclab_arena_examples/`, `isaaclab_arena_g1/`, `isaaclab_arena_gr00t/` — first-party extension packages
- `docker/` — container build and run scripts
- `submodules/` — vendored dependencies (IsaacLab, Isaac-GR00T, …)
- `osmo/` — OSMO policy-runner workflow
- `docs/` — Sphinx documentation

## Coding style

- Prefer `assert condition, "message"` over `if not condition: raise ValueError("message")` for internal invariant checks. (Formatting, imports, and typing are enforced by `pre-commit` — see `.pre-commit-config.yaml`.)
- PR bodies follow `.github/pull_request_template.md` — a one-line Summary plus 2–5 detail bullets. Resist the agent default of long, multi-section descriptions.
- Attribute docstrings should be included below the attribute, rather than in the class-level docstring.
- Copyright headers: a newly created file uses the current year alone (e.g. `2026`); a file created earlier and edited this year uses a range (e.g. `2025-2026`). Don't copy a neighbouring file's year — the pre-commit hooks (`insert-license`, `fix-new-file-copyright-year`) set and enforce this, so you generally don't hand-edit it.

## Docstrings style

- Prefer one line; a 2–3 line paragraph may follow if needed.
- The docstring should describe the function’s calling syntax and its semantics, but generally not its
    implementation details, unless those details are relevant to how the function is to be used.
- Document `Args` and `Returns`, but **not** `Raises`. Omit `Returns` when it only returns None
    or the summary already covers it.
- Don't use Sphinx-style cross-references.

## Conventions

### Coordinate-frame naming

Use active target-source notation for Arena-owned poses and transforms:

- `T_A_B` maps points from frame `B` into frame `A`.
- `T_A_B = (t_A_B, q_A_B)` consists of translation `t_A_B` and rotation
  `q_A_B`, with the rotation represented as a quaternion.
- For example, `T_W_O` maps points from object frame `O` into world frame `W`.
- Transform composition follows `T_C_A = T_C_B * T_B_A`.
- Frame letters are contextual. Define each near its first use when its meaning is not obvious.
- Preserve external API names such as Isaac Lab's `root_pose_w`. Its lowercase `_w`, `_e`, and `_b` suffixes
  denote the simulation world, local environment, and robot base frames, respectively.
- A lowercase API suffix names only the frame in which a quantity is expressed. When both source and target
  frames matter in Arena calculations, bind the value to an explicit transform name, for example
  `T_W_O = object.data.root_pose_w`.

### Wrapped Environment

`ArenaEnvBuilder.make_registered()` returns the gym-wrapped env (not the base env). Use `env.unwrapped` explicitly to access Isaac Lab-specific attributes (`cfg`, `device`, `step_dt`, etc.) that are not forwarded by gymnasium's `OrderEnforcing` wrapper:

```python
env = arena_builder.make_registered()   # wrapped env
env.step(actions)                       # goes through OrderEnforcing
env.unwrapped.cfg                       # access Isaac Lab config
env.unwrapped.device                    # access Isaac Lab device
```

### Writing Tests

Simulation tests use an inner/outer function pattern to handle Isaac Sim's process lifecycle:

```python
def _test_foo(simulation_app):  # runs inside SimulationApp
    from isaaclab_arena.X import Y  # deferred imports after sim init
    ...
    return True  # indicates pass

def test_foo():  # pytest-visible outer function
    result = run_function_with_persistent_simulation_app(_test_foo)
    assert result
```

- **Don't** call CLI paths that may invoke `sys.exit`—such as argparse `--help`, invalid arguments, or `parser.error()`—directly inside pytest. After Isaac Sim starts, catching `SystemExit` still leaves Kit shutdown queued.
- **Instead**, run the CLI with `subprocess.run([TestConstants.python_path, ...])` and assert the child result. Use `with_subprocess` only when the child starts Isaac Sim; the marker does not create a child process.

## Boundaries

- **Never** force-push to `main` or `release/*`. **Instead**, push to a `<username>/<type>/<short-description>` branch (`<type>` ∈ `feature`, `fix`, `docs`, `refactor`, `chore`, `ci`) and open a PR against `main`.
- **Never** add AI-attribution lines to commits (no `Co-Authored-By: Claude…`, no `Generated with…`). **Instead**, sign off with `git commit -s` — DCO is the only required trailer.
- **Never** commit models, datasets, or secrets. **Instead**, keep them on the host and mount them via `./docker/run_docker.sh -d <datasets> -m <models> -e <eval>`.
- **Ask first** before changing `docker/`, `.github/workflows/`, `.pre-commit-config.yaml`, or `submodules/` — these affect every contributor. **Instead** of pushing directly, open a draft PR or raise it in the relevant channel before merging.

## RR sim2real branch and agent handoff

Read `docs/rr_sim2real/README.md` before continuing UR7e work. It covers external
assets, data contracts, collection, DP training/evaluation, measured results and
cross-machine setup. `tools/rr_sim2real/` contains only small migration references,
not datasets, checkpoints or a standalone simulator installation.

The user authorized publishing this work on 2026-09-11 to the fork
`git@github.com:xingyoujun/IsaacLab-Arena.git`, branch `chuanruiz/rr_sim2real`.
This explicit branch name is an exception to the generic naming convention.
Do not push to upstream `origin`, force-push, or create a PR unless requested.
For parallel development, create separate topic branches/worktrees from this
branch, coordinate shared-file changes, and merge with ordinary commits.

## Source host: the `IsaacLab-Arena-tasks` worktree

`/home/ubuntu/code/IsaacLab-Arena-tasks` is a git worktree of the main clone at `/home/ubuntu/code/IsaacLab-Arena`, originally on `chuanruiz/feature/task-configs` and now on `chuanruiz/rr_sim2real`. It exists for **task-configuration work that is unrelated to the Agibot benchmark work in the main clone**. Treat the two checkouts as separate projects that happen to share a `.git` and a Python environment.

- **Different task, different context.** Do not carry Agibot assumptions, gates, or pending decisions from the main clone into this worktree, and do not touch the main clone's working tree from here. Only the code in this directory is in scope.
- **Publication exception.** The original local branch `chuanruiz/feature/task-configs` is retained. RR sim2real development is now shared through the explicit fork branch above. Never touch the main clone's working tree or its active branch.
- **Shared environment, not a shared install.** `.venv` is a symlink to the main clone's `.venv` (excluded via `.git/info/exclude`). The editable install of `isaaclab_arena` resolves to the main clone, so every run from this worktree must set:

  ```bash
  cd /home/ubuntu/code/IsaacLab-Arena-tasks
  export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
  export PYTHONPATH=/home/ubuntu/code/IsaacLab-Arena-tasks:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab
  ```

  Without `PYTHONPATH`, scripts import the main clone's Arena code instead of this one. Run natively from `.venv`; the Docker section above does not apply on this host.
- **Don't modify the shared environment.** No `uv sync`, `pip install`, or submodule updates from this worktree; they would change the main clone's environment. `isaaclab` always comes from the main clone's `submodules/IsaacLab`.
- **Git hygiene.** The stash stack is shared with the main clone; prefer a WIP commit over `git stash`. Remove the worktree with `git worktree remove`, never `rm -rf`.

### Work in this worktree: the UR7e workcell

- Embodiments `ur7e_robotiq_joint_pos` (default) and `ur7e_robotiq_ik` live in `isaaclab_arena/embodiments/ur7e/`; the environment `ur7e_workcell` (table, lights, calibrated D435, robot in `pose_a`, no task) is `isaaclab_arena_environments/ur7e_workcell_environment.py`. Source of truth for the layout and camera is `/home/ubuntu/playground/rr_ur/scene.py`.
- The robot USD is `/home/ubuntu/playground/rr_ur/ur7e_usd/ur7e_gripper/Collected_ur7e_gripper/ur7e_gripper.usd` (the *collected* copy, whose sub-layers and Robotiq asset are local). `select_ur_robot_spec()` falls back to the Isaac UR5e + Robotiq 2F-85 asset with a warning if that file or its sub-layers go missing; `ARENA_UR7E_USD` overrides the path.
- This host has no display. Do not use `environment_runner.py`/`--viz kit`; `--record_viewport_video` also yields nothing headless (Isaac Lab 3.0 `render()` returns None). Record the two sensor cameras instead (D435 + the third-person `scene_cam`), on `cuda:0` (the default device):

  ```bash
  .venv/bin/python isaaclab_arena/evaluation/experiment_runner.py \
      --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/ur7e_workcell_hold_pose.json \
      --output_base_dir /home/ubuntu/playground/experiments/ur7e_workcell --enable_cameras --record_camera_video
  ```

  Camera clips are only flushed when an episode ends, so jobs must set `enable_cameras: true` in `arena_env_args` and run at least one full episode (`episode_length_s`, default 6 s = 90 steps at 15 Hz). `policy_runner.py` is broken on this host for every environment (pxr 0.25.5 in the venv shadows Isaac Sim's 0.25.11 at import time); use the experiment runner.
- Run headless smoke checks the way the tests do; for the GUI use `environment_runner.py ur7e_workcell` (add `--enable_cameras` for the D435 observation).

### Press-toaster task (rr_sim2real)

- Environment `ur7e_press_toaster` (`isaaclab_arena_environments/ur7e_press_toaster_environment.py`): the toaster `toast_rr` stands on the closed drawer unit (`pedestal=True`, +78 mm) at the lower-left of the D435 image (x -0.22..-0.10, y 0.10..0.14, yaw 0 ± 10°, paddle facing the robot); one reset event places both objects from a single sample. Success = `carriage_slide` joint > 75 % of its 47 mm travel. Gravity is disabled on the toaster bodies so the lever stays where it is left (the asset has no return spring).
- The env spawns `/home/ubuntu/playground/rr_ur/toast_rr_arena.usda`, an overlay that hides two decals whose textures are missing (they rendered black) and lightens the metallic materials.
- Driver: `isaaclab_arena_cumotion/scripts/ur7e_press_toaster_cumotion.py` (closed gripper, tool tilted 50-60° towards the toaster, straight vertical press; candidates over tilt x wrist spin are filtered with cuMotion's self-collision inspector and planned to the IK *configuration*, least joint travel wins). Measured constraints, do not re-derive: paddle facing the camera hides the toaster behind the wrist; paddle closer than ~0.43 m to the base has no self-collision-free press; 80-90° (horizontal) tilts self-collide everywhere in the reachable band; planning to a pose target let cuMotion pick a folded IK branch that pressed with the shoulder alone (10 mm).
- Concurrent training has caused RTX crashes inside `ensure_isaac_rtx_render_update`; reserve the GPU for simulation rather than diagnosing these as asset failures. A single local DP inference server plus one simulator was successfully evaluated on 2026-09-11; do not generalize this to arbitrary concurrent CUDA workloads.

### Open-drawer data collection (rr_sim2real)

- Driver: `isaaclab_arena_cumotion/scripts/ur7e_open_drawer_cumotion.py` (top-down knob grasp, `--record-dir` enables states-only HDF5 recording with `Ur7eJointRecordingActionsCfg` + `embodiments/ur7e/demo_recorders.py`; `--init-joint-std` jitters the start pose, the drawer pose is randomised by the env).
- Orchestration: `/home/ubuntu/playground/datasets/collect_ur7e_open_drawer.sh` (2 workers per round, merge, trim, `rerender_embodiment_cameras.py` for the D435 at its configured 640x480, `convert_hdf5_to_lerobot.py --yaml_file isaaclab_arena_gr00t/lerobot/config/ur7e_open_drawer_config.yaml`, `add_ur7e_eef_9d.py`, publish). Raw under `datasets/rr_sim2real_raw/open_drawer`, final LeRobot v2.1 under `datasets/rr_sim2real/open_drawer` (converted output only).
- The drawer env spawns `/home/ubuntu/playground/rr_ur/drawer_rr_arena.usda`, an overlay of the user's `drawer_rr.usdc` that filters collisions between the sliding links and the carcass (the 1 mm clearance otherwise jams the slide at many placements). Diagnose a drawer that will not open with `ur7e_open_drawer_cumotion.py --probe-only --drawer-pose X Y YAW` before touching the grasp.
- Dataset conventions: `observation.state`/`action` = 7 absolute joints `[6 arm, finger_joint]` (rad); `observation.eef_9d`/`action.eef_9d` = TCP xyz + first two rotation-matrix COLUMNS in the UR `base` frame, from URDF FK (`meta/eef_9d.json`); one video `observation.images.realsense_d435` 640x480 h264 at 15 fps. Do not resize the video: it must match the real D435 configuration. `lerobot_to_diffusion_policy_zarr.py` produces the diffusion_policy replay buffer (rot6d rows, 10-dim action with the gripper last).
