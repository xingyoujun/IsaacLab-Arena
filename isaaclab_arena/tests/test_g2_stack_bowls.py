# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G2 tabletop reachability and Tab-switched dual-arm control regression checks."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from isaaclab_arena.tests.test_g2_embodiment import requires_g2_asset
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_g2_dual_keyboard(simulation_app):
    import torch

    import carb.input
    import omni
    from isaaclab.devices.keyboard import Se3KeyboardCfg

    from isaaclab_arena.embodiments.g2.keyboard import G2DualArmKeyboard

    # Headless Kit has no window; mock only the OS event subscription, exercising the real device logic.
    with (
        patch.object(omni, "appwindow", MagicMock(), create=True),
        patch.object(carb.input, "acquire_input_interface", return_value=MagicMock()),
    ):
        keyboard = G2DualArmKeyboard(Se3KeyboardCfg(sim_device="cpu", pos_sensitivity=0.05))

    def event(key, pressed=True):
        keyboard._on_keyboard_event(
            SimpleNamespace(
                input=SimpleNamespace(name=key),
                type=carb.input.KeyboardEventType.KEY_PRESS if pressed else carb.input.KeyboardEventType.KEY_RELEASE,
            )
        )

    event("W")
    event("K")
    command = keyboard.advance()
    assert command.shape == (14,) and command[0] > 0 and command[6] == -1
    assert torch.count_nonzero(command[7:13]) == 0 and command[13] == 1
    event("K", False)
    event("TAB")
    event("TAB", False)
    event("W", False)
    command = keyboard.advance()
    assert torch.count_nonzero(command[:6]) == 0 and torch.count_nonzero(command[7:13]) == 0
    assert command[6] == -1 and command[13] == 1
    event("A")
    event("K")
    command = keyboard.advance()
    assert command[8] > 0 and command[6] == -1 and command[13] == -1
    event("L")
    event("A", False)
    command = keyboard.advance()
    assert torch.count_nonzero(command[:6]) == 0 and torch.count_nonzero(command[7:13]) == 0
    assert command[6] == 1 and command[13] == 1 and keyboard.active_arm == 0
    return True


def _test_g2_stack_bowls_reach(simulation_app):
    import torch

    import warp as wp

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.g2_stack_bowls_environment import (
        G2StackBowlsEnvironment,
        G2StackBowlsEnvironmentCfg,
    )

    cfg = G2StackBowlsEnvironmentCfg(hdr=None)
    description = G2StackBowlsEnvironment().build(cfg)
    args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        assert env.action_space.shape == (1, 14)
        assert base.action_manager.active_terms == [
            "right_arm_action",
            "right_gripper_action",
            "left_arm_action",
            "left_gripper_action",
        ]
        action = torch.zeros((1, 14), device=base.device)
        action[:, [6, 13]] = 1
        for _ in range(30):
            obs, *_ = env.step(action)
        assert "left_eef_pos" in obs["policy"]
        initial_bowls = {
            name: wp.to_torch(base.scene[name].data.root_pos_w)[0].clone() for name in ("bowl_1", "bowl_2", "bowl_3")
        }
        print("Settled bowls:", {name: pos.tolist() for name, pos in initial_bowls.items()})
        for name, pos in initial_bowls.items():
            assert -0.02 < pos[2] < 0.15, f"{name} did not settle on tabletop: {pos}"
        for offset, sensor, bowl_names in (
            (0, "ee_frame", ("bowl_1", "bowl_2")),
            (7, "left_ee_frame", ("bowl_3", "bowl_2")),
        ):
            inactive_sensor = "left_ee_frame" if offset == 0 else "ee_frame"
            inactive_start = wp.to_torch(base.scene[inactive_sensor].data.target_pos_w)[0, 0].clone()
            print(sensor, "initial TCP:", wp.to_torch(base.scene[sensor].data.target_pos_w)[0, 0].tolist())
            for name in bowl_names:
                # Reach immediately above the bowl rim without colliding with the bowl.
                target = initial_bowls[name] + torch.tensor([0.0, 0.0, 0.10], device=base.device)
                for _ in range(160):
                    current = wp.to_torch(base.scene[sensor].data.target_pos_w)[0, 0]
                    if torch.norm(target - current) < 0.01:
                        break
                    action[:, :6] = 0
                    action[:, 7:13] = 0
                    action[0, offset : offset + 3] = ((target - current) * 0.4).clamp(-0.04, 0.04)
                    env.step(action)
                error = torch.norm(wp.to_torch(base.scene[sensor].data.target_pos_w)[0, 0] - target).item()
                print(f"{sensor} -> {name}: {error:.4f} m")
                assert error < 0.025, f"{sensor} cannot reach {name}: {error:.4f} m"
            inactive_end = wp.to_torch(base.scene[inactive_sensor].data.target_pos_w)[0, 0]
            assert torch.norm(inactive_end - inactive_start) < 0.02, "Inactive arm drifted"
        return True
    finally:
        env.close()


def test_g2_dual_keyboard():
    assert run_function_with_persistent_simulation_app(_test_g2_dual_keyboard, headless=True)


@requires_g2_asset
def test_g2_stack_bowls_reach():
    assert run_function_with_persistent_simulation_app(_test_g2_stack_bowls_reach, headless=True)
