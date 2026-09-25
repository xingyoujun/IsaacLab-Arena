# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check G2 placement thresholds and sequential success across the task API migration."""

from isaaclab_arena.tests.test_g2_embodiment import requires_g2_asset
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_g2_stack_contract(simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena_environments.g2_stack_bowls_environment import (
        G2StackBowlsEnvironment,
        G2StackBowlsEnvironmentCfg,
    )

    task = G2StackBowlsEnvironment().build(G2StackBowlsEnvironmentCfg(hdr=None)).task
    cfg = task.get_termination_cfg()
    assert cfg.subtasks_are_sequential and cfg.desired_subtask_success_state == [True, True]

    def array(value):
        return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))

    scene = {}
    for name, height in (("bowl_2", 0), ("bowl_1", 0.03), ("bowl_3", 0.06)):
        scene[name] = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=array([[0, 0, height]]), root_lin_vel_w=array([[0, 0, 0]]))
        )
    for subtask in task.subtasks:
        scene[subtask.contact_sensor_name] = SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=array([[[[0, 0, 1]]]]))
        )
    env = SimpleNamespace(scene=scene, num_envs=1, device="cpu")
    first = task.subtasks[0]
    assert first.is_placed(env).item()
    position = scene["bowl_1"].data.root_pos_w.torch
    velocity = scene["bowl_1"].data.root_lin_vel_w.torch
    force = scene[first.contact_sensor_name].data.force_matrix_w.torch
    position[0, 0] = first.max_separation[0]
    assert not first.is_placed(env).item(), "The original XY limit is strict"
    position[0, 0] = 0
    velocity[0, 0] = 0.1
    assert not first.is_placed(env).item(), "A moving bowl must not count as placed"
    velocity.zero_()
    force[..., 2] = 0.1
    assert not first.is_placed(env).item(), "Contact force must exceed the original threshold"
    force[..., 2] = 1

    tracker = ProgressTracker(
        cfg.success,
        num_envs=1,
        device="cpu",
        env=env,
        subtasks_are_sequential=True,
        desired_subtask_success_state=[True, True],
    )
    tracker.step(env, torch.tensor([0]))
    assert not tracker.is_complete().item(), "Two placements cannot advance in the same step"
    tracker.step(env, torch.tensor([1]))
    assert tracker.is_complete().item()
    position[0, 0] = 0.5
    tracker.step(env, torch.tensor([2]))
    assert not tracker.is_complete().item(), "Both placements must still hold at the final step"
    position[0, 0] = 0
    tracker.reset(torch.tensor([0]))
    assert not tracker.is_complete().item()
    tracker.step(env, torch.tensor([0]))
    assert not tracker.is_complete().item(), "Reset must discard both subtask histories"
    return True


@requires_g2_asset
def test_g2_stack_contract():
    assert run_function_with_persistent_simulation_app(_test_g2_stack_contract, headless=True)


def _test_g2_cli_registration(simulation_app):
    import gymnasium as gym
    from unittest.mock import patch

    from isaaclab_arena.environments.isaaclab_interop import environment_registration_callback

    argv = [
        "teleop_se3_agent.py",
        "--task",
        "g2_stack_bowls",
        "--arena_teleop_device",
        "keyboard",
        "--device",
        "cpu",
        "--num_envs",
        "1",
    ]
    with patch("sys.argv", argv):
        assert environment_registration_callback() == []
    cfg = gym.spec("g2_stack_bowls").kwargs["env_cfg_entry_point"]
    assert cfg.scene.num_envs == 1
    assert "keyboard" in cfg.teleop_devices.devices
    assert cfg.actions.right_arm_action is not None and cfg.actions.left_arm_action is not None
    return True


@requires_g2_asset
def test_g2_cli_registration():
    assert run_function_with_persistent_simulation_app(_test_g2_cli_registration, headless=True)
