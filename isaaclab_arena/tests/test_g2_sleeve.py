# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check sleeve success geometry, physical seating, and reset behavior."""

from isaaclab_arena.tests.test_g2_embodiment import requires_g2_asset
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_sleeve_success_geometry(simulation_app):
    import math
    import torch
    from types import SimpleNamespace

    import warp as wp
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

    from isaaclab_arena.tasks.sleeve_task import peg_is_inserted

    # Seated, beside bore, above opening, partly inserted, over-penetrated, wrong end, tilted, axial spin.
    positions = torch.tensor(
        [[0, 0, 0], [0.02, 0, 0], [0, 0, 0.09], [0, 0, 0.03], [0, 0, -0.01]] + [[0, 0, 0]] * 3,
        dtype=torch.float32,
    )
    rotations = torch.tensor([[0, 0, 0, 1]] * 8, dtype=torch.float32)
    rotations[5] = torch.tensor([1, 0, 0, 0])
    rotations[6] = torch.tensor([math.sin(0.1), 0, 0, math.cos(0.1)])
    rotations[7] = torch.tensor([0, 0, 1, 0])
    # Rotate and translate both objects to verify the predicate uses the sleeve frame.
    angles = torch.full((8,), 0.4)
    sleeve_quat = quat_from_euler_xyz(angles, angles, angles)
    sleeve_pos = torch.tensor([[2.0, -3.0, 1.0]] * 8)

    def asset(pos, quat):
        return SimpleNamespace(data=SimpleNamespace(root_pos_w=wp.from_torch(pos), root_quat_w=wp.from_torch(quat)))

    env = SimpleNamespace(
        scene={
            "sleeve": asset(sleeve_pos, sleeve_quat),
            "peg": asset(sleeve_pos + quat_apply(sleeve_quat, positions), quat_mul(sleeve_quat, rotations)),
        }
    )
    result = peg_is_inserted(env, SceneEntityCfg("sleeve"), SceneEntityCfg("peg"), 0.001, 0.003, 1.0)
    assert result.tolist() == [True, False, False, False, False, False, False, True]
    return True


def _test_g2_sleeve_physics(simulation_app):
    import torch

    import warp as wp

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.tasks.sleeve_task import peg_is_inserted
    from isaaclab_arena_environments.g2_sleeve_environment import G2SleeveEnvironment, G2SleeveEnvironmentCfg

    description = G2SleeveEnvironment().build(G2SleeveEnvironmentCfg())
    args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "2", "--device", "cpu"])
    env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        sleeve = base.scene["sleeve"]
        peg = base.scene["peg"]
        params = description.task.get_termination_cfg().success.params
        action = torch.zeros(env.action_space.shape, device=base.device)
        action[:, [6, 13]] = 1
        for _ in range(30):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated.any() and not truncated.any()
        initial_sleeve = wp.to_torch(sleeve.data.root_pos_w).clone()
        initial_peg = wp.to_torch(peg.data.root_pos_w).clone()
        assert not peg_is_inserted(base, **params).any()
        assert torch.allclose(initial_sleeve - base.scene.env_origins, torch.tensor([[-0.10, 0, 0.009]]), atol=0.001)
        assert torch.allclose(initial_peg[:, 2], torch.full((2,), 0.0), atol=0.005)

        # A test-only peg placement above the bore followed by gravity verifies physical insertion.
        # This is not a robot policy demonstration.
        pose = torch.zeros((2, 7), device=base.device)
        pose[:, :3] = initial_sleeve + torch.tensor([0, 0, 0.09], device=base.device)
        pose[:, 6] = 1
        peg.write_root_pose_to_sim(pose)
        peg.write_root_velocity_to_sim(torch.zeros((2, 6), device=base.device))
        for _ in range(120):
            base.scene.write_data_to_sim()
            base.sim.step()
            base.scene.update(base.physics_dt)
        print("Inserted peg positions:", wp.to_torch(peg.data.root_pos_w).tolist())
        assert peg_is_inserted(base, **params).all()
        assert torch.allclose(wp.to_torch(sleeve.data.root_pos_w), initial_sleeve, atol=0.001)
        _, _, terminated, _, _ = env.step(action)
        assert terminated.all()
        env.reset()
        for _ in range(20):
            env.step(action)
        assert not peg_is_inserted(base, **params).any()
        assert torch.allclose(wp.to_torch(peg.data.root_pos_w), initial_peg, atol=0.005)
        return True
    finally:
        env.close()


def test_sleeve_success_geometry():
    assert run_function_with_persistent_simulation_app(_test_sleeve_success_geometry, headless=True)


@requires_g2_asset
def test_g2_sleeve_physics():
    assert run_function_with_persistent_simulation_app(_test_g2_sleeve_physics, headless=True)
