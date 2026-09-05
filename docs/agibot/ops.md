# Operations on this host

The L40S / 8 vCPU box the team works on. Everything here is host-specific; another machine needs
its own copy of this page.

## Environment

Arena runs **natively from `.venv`** (uv), not Docker (the container was retired 2026-07-28).

```bash
cd /home/ubuntu/code/IsaacLab-Arena
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y      # else imports block on an interactive EULA prompt
.venv/bin/python isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_steps 30 agibot_stack_bowls
```

- `isaacsim.core.utils.*` is not importable in this install; use `omni.usd` / `pxr` directly.
- Isaac assets are cached under `ISAAC_ASSET_ROOT` (default `/tmp/Assets/Isaac/6.0/Isaac`); the
  cuMotion registry reads the Agibot URDF from there and the Lula yamls from Arena's copies.
- Lint tooling is not installed on the host; run the pinned hooks through `uvx`:
  `uvx black@24.3.0 --line-length 120 --unstable <files>`, `uvx isort@5.13.2 --profile black
  --filter-files <files>`, `uvx flake8@7.0.0 <files>`, `uvx codespell@2.2.6 <files>`.
  In zsh, put file lists in an **array** (`FILES=(a b)`; `"${FILES[@]}"`) -- an unquoted `$FILES`
  is not word-split, and that trap once made a three-way parameter sweep run the same baseline
  three times.
- Fresh clones: submodules over HTTPS (`git -c url."https://github.com/".insteadOf="git@github.com:"
  submodule update --init --recursive`), then `git lfs pull` or the `.hdf5` test data are stubs.
- After any `uv sync`, check the venv's integrity before trusting test failures: compare every
  `*.dist-info/RECORD` against disk (a ten-line Python loop; see `upstream_sync.md`). On 2026-09-05 the
  `usd-exchange` package had lost 51 files, which made every Kit-less import of Arena fail and looked
  exactly like an upstream bug. `uv sync --extra dev --reinstall-package <name>` repairs a package.
- Tests: `.venv/bin/python -m pytest -m 'not with_cameras and not with_subprocess' isaaclab_arena/tests`
  is Phase 1 (~1080 tests, green after the default-config merge); the `run-tests` skill describes
  the three phases (its Docker wrapping does not apply here).

## Display and teleoperation

Kit-GUI work goes through **noVNC** on `:6080` (Xvfb `:99`, VNC password in `~/.vnc/passwd`).
Check for leftovers first: `ps aux | grep -E "Xvfb|x11vnc|websockify"`.

```bash
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset
x11vnc -display :99 -forever -shared -localhost -rfbauth ~/.vnc/passwd -rfbport 5900 -noxdamage -quiet
websockify --web=/usr/share/novnc 6080 localhost:5900
```

Then `DISPLAY=:99 ... --viz kit --device cuda:0` (no `--viz` means headless since Isaac Lab 3.0 GA; the old `--headless` flag is gone). **Everything runs on `cuda:0`** (user decision 2026-09-05): teleop, recording, cuMotion, probes. `--device cpu` is a debugging mode only -- it surfaces errors the GPU pipeline swallows. Arena's own teleop/record
scripts were deleted upstream; use Isaac Lab's with Arena's registration callback and **both**
device flags (`--arena_teleop_device` configures, `--teleop_device` instantiates; giving only the
first silently falls back to the 7-value keyboard and dies with `expected: 14, received: 7`):

```bash
# drive only (records nothing). ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0 works around the
# Isaac Lab 3.0 GA issue of a blank Kit viewport (RTX per-env scene partitioning).
DISPLAY=:99 ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \
  submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py --viz kit --device cuda:0 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task agibot_tidy_workbench --arena_teleop_device dual_arm_keyboard --teleop_device dual_arm_keyboard

# record
DISPLAY=:99 ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \
  submodules/IsaacLab/scripts/tools/record_demos.py --viz kit --device cuda:0 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task agibot_tidy_workbench --arena_teleop_device dual_arm_keyboard --teleop_device dual_arm_keyboard \
  --step_hz 15 --num_demos 0 --num_success_steps 10 \
  --dataset_file /home/ubuntu/playground/datasets/tidy_workbench_teleop/teleop_demos.hdf5
```

Dual-arm keyboard: Tab switches the driven arm; each arm has its own gripper latch. The CPU device
is what exposes the zero-joint-articulation crash (see the asset guide).

Kill teleop with `pkill -f teleop_se3_agent` / `pkill -f record_demos`, never `head -1 | xargs
kill`: zombie instances have silently eaten 6 cores and 10 GB of VRAM. The WebRTC livestream path
(needs `libxt6 libxaw7 libxmu6`) works but noVNC is preferred.

## Video and rendering

- Headless `env.render()` / `--record_viewport_video` is **black**; record camera sensors.
- A scene without a light asset renders black in every path.
- Faithful demo replay: `scene.reset_to(frame, is_relative=True)` + `sim.step(render=True)` per
  frame; `forward()` or `render()` alone leaves rigid bodies frozen. Disable `time_out` for replays
  longer than 20 s.
- Stream frames to the writer; accumulating them OOM-killed a run at 1280x720.
- Runtime camera prims under a moving link do not track it; anchor at `/World` and re-pose every
  frame from `body_pos_w` / `body_quat_w` (the embodiment's wrist cameras do this).

## Concurrency and resources

- **Two** Isaac Sim processes at a time. Three freezes re-render workers indefinitely.
- **No training on this host** (user directive after the 2026-09-02 freeze). Fine-tuning goes to
  another machine; the GR00T N1.6 recipe (37.5 GB VRAM at micro-batch 8, rot6d rows conversion,
  `ActionType.EEF` unimplemented -> use RELATIVE + NON_EEF) is recorded in the cumotion guide's
  dataset section and in session memory for reuse elsewhere.
- Closed-loop eval: GR00T server in its own container (`isaaclab_arena_gr00t:1_6`, `--net=host`,
  models mounted from `/home/ubuntu/playground/models`) + `policy_runner.py` client; port 5555 is
  taken, use 5757. Training and eval cannot share the GPU.

## Data layout (`/home/ubuntu/playground/`)

| path | contents |
| --- | --- |
| `datasets/agibot_arena_v0/<task>/` | delivered LeRobot v2.1 datasets (upload-ready) |
| `datasets/<task>_v0_raw/` | raw HDF5 per collection round (`run_*/worker_*`), trimmed `<task>.hdf5`, `.cameras/` sidecars (handover 167 GB, bowls 36 GB) |
| `datasets/<task>_teleop/` | human recordings (`sleeve_on_peg_teleop`, `tidy_workbench_teleop`) |
| `datasets/collect_agibot_arena_v0.sh` | collection driver (reference implementation) |
| `objects/arena_local/`, `objects/agibot_assets_v0/` | local USD assets and override layers (see the asset guide) |
| `videos/`, `stackbowls/`, `make_toast/` | tuning videos and collection logs |
| `gripper_matrix/` | the 2026-08-15 actuator cross-matrix data behind the frozen gripper config |

Never store datasets or checkpoints in a container or in `/tmp`; the session scratchpad is wiped
between sessions, so probe outputs worth keeping go under `playground/` too.

### Clean-up candidates (not deleted; user's call)

| item | size | why it can go |
| --- | --- | --- |
| `Lightwheel_OpenSource.zip` | 3.2 GB | already extracted beside it |
| `arena_bugfix_worktree/` | 0.5 GB | worktree of the bug-fix branch; the branch itself stays (removed 2026-09-05 with `git worktree remove`) |
| `stack_bowls_trace_*.csv` (9 files), `stack_bowls_*.mp4` (4), `*_status*.txt`, `*_shots/`, `push_t_*`, `eef_replay_videos/`, `cumotion_grasp_headset.mp4` | ~0.1 GB | debugging artefacts of resolved issues (2026-08) |
| `agibot_cumotion/` | small | XRDF prototypes superseded by `isaaclab_arena_cumotion/robot_description.py` |
| `objects/arena_local/{t_block*,t_pad,headset*,laptop*,spring_button*,toaster_oven_*}` | small | assets of tasks not on `main` (push_T, press_button, store_laptop, kitchen oven) |
| `objects/assets_07*`, `objects/assets_08*`, `fan_probe`, `pair_probe`, `gen_v1_100`, `newton_reference` | ~1 GB total with the rest of `objects/` | earlier USDCraft exports; only `agibot_assets_v0` and `arena_local` are referenced by code |
| repo `sbi-logs/` (ignored) | tiny | TensorBoard events from an unrelated 2026-08-15 run |
