# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnose authored mechanism dynamics using external forces, never robot-demonstration labels."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kind", choices=["kettle", "toaster"], required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--enable_cameras", action="store_true")
args = parser.parse_args()
args.headless = True
cameras_enabled = args.enable_cameras
launcher = AppLauncher(args)


def main():
    import torch

    from data_engine.motion.cumotion.executor import JointActionInterface
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.pine_wm_interactions_environment import (
        PineWmKettleEnvironment,
        PineWmKettleEnvironmentCfg,
        PineWmToasterEnvironment,
        PineWmToasterEnvironmentCfg,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    factory, config = (
        (PineWmKettleEnvironment, PineWmKettleEnvironmentCfg)
        if args.kind == "kettle"
        else (PineWmToasterEnvironment, PineWmToasterEnvironmentCfg)
    )
    arena = factory().build(config(enable_cameras=cameras_enabled))
    env = (
        ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(device=args.device, solve_relations=False))
        .make_registered()
        .unwrapped
    )
    env.reset()
    interface = JointActionInterface(env)
    asset = env.scene[args.kind]
    print("MECHANISM", asset.joint_names, asset.body_names, flush=True)
    latch = env.interaction_latches[args.kind]
    print("INITIAL", asset.data.joint_pos.torch.cpu().tolist(), flush=True)
    samples = []

    def step(phase, steps, force_body=None, force=None, torque=None):
        forces = torch.zeros((1, asset.num_bodies, 3), device=env.device)
        torques = torch.zeros_like(forces)
        if force_body:
            index = asset.body_names.index(force_body)
            if force is not None:
                forces[0, index] = torch.tensor(force, device=env.device)
            if torque is not None:
                torques[0, index] = torch.tensor(torque, device=env.device)
        asset.set_external_force_and_torque(forces, torques, is_global=False)
        for _ in range(steps):
            interface.step()
            samples.append(
                dict(phase=phase, q=asset.data.joint_pos.torch[0].cpu().tolist(), engaged=list(latch.engaged[0]))
            )
        print(phase, samples[-1], flush=True)

    step("rest_locked" if args.kind == "kettle" else "rest", 30)
    if args.kind == "kettle":
        step("press_release", 15, "lid_release", [0, 0, -4])
        step("spring_open", 60)
        step("close_lid", 45, "lid", torque=[0.5, 0, 0])
        step("relatched", 30)
        step("press_release_again", 15, "lid_release", [0, 0, -4])
        step("spring_open_again", 60)
    else:
        step("lower_carriage", 30, "carriage", [0, 0, -12])
        step("latched", 30)
        step("press_cancel", 15, "cancel_button", [0, 4, 0])
        step("spring_return", 60)
    report = dict(
        kind=args.kind,
        mode="external_force_mechanism_probe",
        robot_demonstration=False,
        joint_names=asset.joint_names,
        samples=samples,
        events=latch.events,
    )
    report["events"] = list(latch.events)
    env.reset()
    step("reset_rest", 30)
    (args.output / "reset.json").write_text(json.dumps(samples[-30:], indent=2) + "\n")

    def final(phase, joint):
        rows = [sample for sample in samples if sample["phase"] == phase]
        return rows[-1]["q"][asset.joint_names.index(joint)], rows[-1]["engaged"][0]

    if args.kind == "kettle":
        closed, held = final("rest_locked", "lid_hinge")
        opened, released = final("spring_open", "lid_hinge")
        reclosed, relatched = final("relatched", "lid_hinge")
        reopened, rereleased = final("spring_open_again", "lid_hinge")
        reset_q, reset_latch = final("reset_rest", "lid_hinge")
        checks = dict(
            initially_locked=held and abs(closed) < 0.02,
            spring_opened=not released and 1.2 < opened < 1.42,
            relatched=relatched and abs(reclosed) < 0.02,
            released_again=not rereleased and 1.2 < reopened < 1.42,
            reset=reset_latch and abs(reset_q) < 0.02,
        )
    else:
        held_q, held = final("latched", "carriage_slide")
        returned, released = final("spring_return", "carriage_slide")
        reset_q, reset_latch = final("reset_rest", "carriage_slide")
        checks = dict(
            held_after_force_removed=held and abs(held_q - 0.06) < 0.001,
            returned_after_cancel=not released and abs(returned) < 0.005,
            reset=not reset_latch and abs(reset_q) < 0.005,
        )
    report.update(checks=checks, success=all(checks.values()))
    (args.output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    assert report["success"], checks


code = 0
try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    code = 1
finally:
    launcher.app.close(exit_code=code)
