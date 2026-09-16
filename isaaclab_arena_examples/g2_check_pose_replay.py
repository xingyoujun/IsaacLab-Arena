# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare physical G2 joint-action and measured-EEF playback without modifying source datasets."""

import json
import traceback
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher


def replay(args):
    import h5py
    import numpy as np
    import torch

    from isaaclab.managers import TerminationTermCfg
    from isaaclab.utils.math import compute_pose_error, subtract_frame_transforms

    from isaaclab_arena.embodiments.g2.g2 import G2DualArmActionsCfg, G2JointPositionActionsCfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.g2_stack_bowls_environment import (
        G2StackBowlsEnvironment,
        G2StackBowlsEnvironmentCfg,
    )

    source_report = json.loads((args.raw.parent / "report.json").read_text())
    description = G2StackBowlsEnvironment().build(
        G2StackBowlsEnvironmentCfg(
            hdr=None,
            table_height_m=source_report["configuration"]["table_height_m"],
            episode_length_s=600,
            bowl_positions=source_report["configuration"]["bowl_positions"],
        )
    )
    action_cfg = G2JointPositionActionsCfg()
    if args.mode == "eef":
        action_cfg = G2DualArmActionsCfg()
        for term in (action_cfg.right_arm_action, action_cfg.left_arm_action):
            term.controller.use_relative_mode = False
            term.scale = 1.0
    description.embodiment.action_config = action_cfg
    original_callback = description.env_cfg_callback
    terms = {}

    def configure(cfg):
        cfg = original_callback(cfg)
        for name, term in vars(cfg.terminations).items():
            if isinstance(term, TerminationTermCfg):
                terms[name] = term
                setattr(cfg.terminations, name, None)
        return cfg

    description.env_cfg_callback = configure
    env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
    base = env.unwrapped
    robot = base.scene["robot"]
    report = {
        "mode": args.mode,
        "raw": str(args.raw),
        "physics_replay": True,
        "initial_state_restores": 1,
        "pose_target": "next recorded pre-action pose; final sample held",
        "gripper_commands": "original action at current timestep",
        "repeat": args.repeat,
    }
    try:
        env.reset()
        with h5py.File(args.raw, "r") as dataset:
            episode = dataset["data/demo_0"]
            actions = torch.tensor(episode["actions"][:], device=base.device)
            reference = torch.tensor(episode["core/eef_pose"][:], device=base.device)
            initial = {
                category: {
                    name: {
                        key: torch.tensor(value[0], device=base.device).unsqueeze(0) for key, value in fields.items()
                    }
                    for name, fields in objects.items()
                }
                for category, objects in episode["initial_state"].items()
            }
        base.scene.reset_to(initial, env_ids=torch.tensor([0], dtype=torch.int32, device=base.device), is_relative=True)
        robot.set_joint_position_target_index(target=robot.data.joint_pos.torch.clone())
        base.scene.write_data_to_sim()
        base.sim.forward()
        base.scene.update(base.step_dt)
        errors, successes, failures = [], [], set()
        for index in range(len(actions)):
            target = reference[min(index + 1, len(actions) - 1)]
            action = actions[index].clone()
            if args.mode == "eef":
                action[:7] = target[:7]
                action[8:15] = target[7:]
            for _ in range(args.repeat):
                env.step(action.unsqueeze(0))
                successes.append(bool(terms["success"].func(base, **terms["success"].params)[0]))
                for name, term in terms.items():
                    if name != "success" and not term.time_out and bool(term.func(base, **term.params)[0]):
                        failures.add(name)
            frame_error = []
            for offset, name in ((0, "ee_frame"), (7, "left_ee_frame")):
                sensor = base.scene[name]
                pos, quat = subtract_frame_transforms(
                    robot.data.root_pos_w.torch,
                    robot.data.root_quat_w.torch,
                    sensor.data.target_pos_w.torch[:, 0],
                    sensor.data.target_quat_w.torch[:, 0],
                )
                pos_error, rot_error = compute_pose_error(
                    pos,
                    quat,
                    target[offset : offset + 3].unsqueeze(0),
                    target[offset + 3 : offset + 7].unsqueeze(0),
                    rot_error_type="axis_angle",
                )
                frame_error.extend((float(pos_error.norm()), float(rot_error.norm())))
            errors.append(frame_error)
            if index % 200 == 0:
                print(f"REPLAY mode={args.mode} frame={index}/{len(actions)} errors={frame_error}", flush=True)
        values = np.array(errors)
        report.update(
            steps=len(actions) * args.repeat,
            step_dt=base.step_dt,
            final_success=all(successes[-10:]),
            failure_terms=sorted(failures),
            error_columns=["right_position_m", "right_rotation_rad", "left_position_m", "left_rotation_rad"],
            max_errors=values.max(0).tolist(),
            mean_errors=values.mean(0).tolist(),
            p95_errors=np.quantile(values, 0.95, axis=0).tolist(),
            final_bowl_positions={
                name: base.scene[name].data.root_pos_w.torch[0].tolist() for name in ("bowl_1", "bowl_2", "bowl_3")
            },
        )
    except Exception as exc:
        report["error"] = str(exc)
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(report, indent=2))
        print("REPLAY_RESULT", json.dumps(report), flush=True)
        env.close()


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--mode", choices=("joint", "eef"), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.repeat > 0 and not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    launcher = get_app_launcher(args)
    try:
        replay(args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
