---
name: agibot-benchmark-task
description: The fixed process for Agibot benchmark work in Arena -- bringing a new task or asset from USD to teleop, human recording, cuMotion collection and dataset, and triaging failures by layer (harness, task, layout, asset, robot). Use when the user asks to add or port an Agibot task, register or qualify a new asset, set up teleop or recording for an Agibot env, write or tune a cuMotion driver, run a collection, or when an Agibot task misbehaves ("the object flies", "the arm can't reach", "success never fires", "the arm drifts").
argument-hint: "[new-task|new-asset|triage|collect] [env or asset name]"
---

# Agibot benchmark task

The process is written up in `docs/agibot/`; this skill is the entry point. Read the relevant page
before acting, follow the gate order, and do not tune anything outside the stage you are in.

| you are asked to | read | then |
| --- | --- | --- |
| add or port a task | `docs/agibot/new_task_playbook.md` | run stages 0-3 gates before writing the environment; subclass `AgibotTabletopEnvironmentCfg`, build with `build_tabletop_stage` / `build_agibot`, start the callback with `install_agibot_control_stack` (template: `agibot_stack_bowls`) |
| bring in or resize an asset | `docs/agibot/asset_guide.md` | register with measured constants; `probe_drop_settle.py`, then `probe_pinch.py` at the intended scale, then `probe_reach.py` at the intended spot |
| debug a misbehaving task | `docs/agibot/triage.md` | run the discriminating measurement for the symptom **before** forming a hypothesis; check the refuted list before proposing any parameter change |
| touch the robot config | `docs/agibot/agibot_embodiment.md` | don't. The embodiment and control stack are frozen by user directive; report a defect instead |
| write / tune a cuMotion driver or run a collection | `docs/agibot/cumotion_guide.md` | one change -> one video -> report to the user; collection via the reference driver, two workers max |
| run anything on this host | `docs/agibot/ops.md` | uv `.venv` + EULA env vars, noVNC for Kit, both teleop device flags, data under `/home/ubuntu/playground/` |

## The probes (all `isaaclab_arena_cumotion/scripts/`, headless, `--env-arg FIELD=VALUE` overrides)

| probe | gate |
| --- | --- |
| `probe_drop_settle.py --env E` | origin convention, resting height, at-rest noise floor |
| `probe_pinch.py --env E --object O --centre_offset Z` | graspability: HELD/HELD every repeat, close peak < 1 m/s |
| `probe_reach.py --env E --point x,y,z ... --tilts 0,30,50,75` | reachable poses per arm and lean |
| `probe_staged_success.py --env E --pose NAME=x,y,z[,yaw]` | success predicate on a staged goal, before and after physics |
| `probe_gripper_span.py`, `probe_gripper_axes.py`, `probe_tool_orientation.py` | gripper geometry and cuMotion frame checks |

## Working rules (user directives)

- After one failed validation run, report the symptom and candidate causes and ask; no solo
  fix-and-retry loops.
- One change per run, one video per change, the user judges the video.
- Never modify the gripper, arm actuators or controller flags for a task problem.
- Never train on this host. Never push, open PRs or post externally without an explicit go-ahead.
- Record settled numbers in code docstrings and `docs/agibot/`; keep session memory for pending
  user decisions and preferences only.
