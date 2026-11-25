# AgiBot G2

The G2 omnipicker embodiment (`g2`, `g2.py`): a fixed-base G2 with both arms, its head and wrist cameras, and the
action interfaces below.

## Evaluation tasks (G2-Eval)

The G2 evaluation tasks come from the G2-Eval asset pack ([`xingyoujun/G2-Eval`](https://huggingface.co/datasets/xingyoujun/G2-Eval),
a gated HF dataset: request access on its page and log in with `hf auth login`) at the revision pinned by
`isaaclab_arena/g2_eval/lock.json`; each registers an environment named after its task (`pick_apple`, `pick_orange`,
`stack_bowls`, `peg_into_sleeve`, `clean_workcell_table`, `tube_rack_insert`, `gear_shaft_assembly`). Download the pack, then evaluate:

```bash
.venv/bin/python -m isaaclab_arena.g2_eval.pack download   # into local_assets/G2-Eval/, every file checked
# or keep the pack elsewhere: export G2_EVAL_ASSETS_DIR=/path/to/G2-Eval
.venv/bin/python isaaclab_arena/evaluation/policy_runner.py --headless --policy_type <policy> --num_episodes 50 pick_apple
```

These environments use the pack's robot, table and room, and the interface the tasks' training data were recorded
with: **16 absolute joint-position actions** at 15 Hz, ordered right arm (7), right gripper (1), left arm (7), left
gripper (1), with +1 open / -1 closed (`G2JointPositionActionsCfg`), and the head (640x400) and wrist (640x528)
cameras. The pack holds two kinds of task:

- `g2.eval.task.v1` (the first five): each episode draws the task's layout from its seed; success is the task's
  relations on its `geometry.json` (`isaaclab_arena/g2_eval/success.py`).
- `g2.eval.task.v2` (`tube_rack_insert`, `gear_shaft_assembly`): each episode starts from one of the task's listed
  layouts, fixed props included: episode `k` of environment `i` of `N` takes layout `(k * N + i) mod` their count, so
  the same `--num_envs` evaluates the same layouts. Success is the task's `success_check`, judged every control step
  by the data engine's judge (`isaaclab_arena/g2_eval/success_check.py`) on the objects' poses, each arm's
  `gripper_{l,r}_center_link` pose and each gripper's opening (finger angle over the open angle, 1 open); the episode
  ends as a success when it decides so, and as a failure when a query is counted past its aim (a fixed prop moved).
  Its start is the state after the episode's first control step (there are no settle steps). A v2 task may bring its
  own `workbench`: the table it was collected on (hidden, but colliding, when a look stands in for it), the robot's
  position, and a room or a look-only stand-in table and room (seen, never collided with) with its light.

## Keyboard teleoperation

`keyboard.py` provides a dual-arm keyboard controller for a G2 environment with the default differential-IK actions
(the G2-Eval environments use joint-position actions instead). The selected arm starts on the right; Tab switches arms and prints the selection in the terminal. Release and press a
movement key again after switching. Each gripper retains its own open/closed state.

| Keys | Action on the selected arm |
| --- | --- |
| Tab | Switch right/left arm |
| W / S | Forward / backward |
| A / D | Left / right |
| Q / E | Up / down |
| Z / X | Roll |
| T / G | Pitch |
| C / V | Yaw |
| K | Toggle gripper |
| L | Reset keyboard state: right arm selected, both grippers open |
| R | Reset the environment (teleop script binding) |

Dual-arm keyboard actions have 14 values: right delta pose (6), right gripper (1), left delta pose (6), left gripper
(1). Existing `eef_pos`/`eef_quat` observations describe the right TCP; `left_eef_pos`/`left_eef_quat` describe the
left TCP. `--arm_mode right` or `--arm_mode left` retains the single-arm, seven-value action interface.
