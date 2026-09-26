# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import tqdm

import pytest
import warp as wp

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_STEPS = 10
HEADLESS = True
INITIAL_POSITION_EPS = 1e-6

from isaaclab_arena.assets.usdcraft_scene import resolve_asset

try:
    resolve_asset("g2")
    _G2_AVAILABLE = True
except AssertionError:
    _G2_AVAILABLE = False
requires_g2_asset = pytest.mark.skipif(
    not _G2_AVAILABLE,
    reason="Download the pinned USDCraft-Scene release before running G2 tests",
)


def get_g2_test_environment(num_envs: int = 1, arm_mode=None):
    """Returns a kitchen scene with the G2 embodiment for testing."""
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.embodiments.g2.g2 import G2Embodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("kitchen")()

    pick_up_object = asset_registry.get_asset_by_name("cracker_box")()
    pick_up_object.set_initial_pose(
        Pose(
            position_xyz=(0.4, 0.0, 0.1),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )

    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
        parent_asset=background,
        object_type=ObjectType.RIGID,
    )

    # Genie Sim places the G2 ~0.66 m back from a 0.8 m-high table; the kitchen counter top is at z ~ 0 and its
    # floor at z = -0.895, where the chassis wheels rest.
    robot_init_position = (-0.7, 0.0, -0.895)
    embodiment = G2Embodiment(arm_mode=arm_mode or ArmMode.RIGHT)
    embodiment.set_initial_pose(Pose(position_xyz=robot_init_position, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    scene = Scene(assets=[background, pick_up_object, destination_location])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="g2_kitchen_test",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(pick_up_object, destination_location, background),
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()

    return env, robot_init_position


def _test_g2_registered(simulation_app) -> bool:
    """The G2 embodiment is reachable through the asset registry under its name."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.g2.g2 import G2Embodiment

    try:
        assert AssetRegistry().get_asset_by_name("g2") is G2Embodiment
        assert "embodiment" in G2Embodiment.tags
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True


def _test_g2_initial_position(simulation_app) -> bool:
    """The fixed-base G2 stays at its configured initial position."""

    env, robot_init_position = get_g2_test_environment(num_envs=1)

    try:
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        robot_position = wp.to_torch(env.unwrapped.scene["robot"].data.root_link_pose_w)[0, :3].cpu().numpy()
        robot_position_error = np.linalg.norm(robot_position - np.array(robot_init_position))
        print(f"Robot position error: {robot_position_error}")
        assert robot_position_error < INITIAL_POSITION_EPS, "G2 ended up at the wrong position."

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_g2_action_and_observation_spaces(simulation_app) -> bool:
    """Action space is 7 (relative IK pose) + 1 (gripper) and the observation terms are finite."""

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, _ = get_g2_test_environment(num_envs=2)

    try:
        action_shape = env.action_space.shape
        print(f"G2 action space shape: {action_shape}")
        assert action_shape[-1] == 7, f"Expected 6-DoF relative pose + 1 gripper action, got {action_shape}"

        with torch.inference_mode():
            step_zeros_and_call(env, NUM_STEPS)

            robot_data = env.unwrapped.scene["robot"].data
            joint_pos = wp.to_torch(robot_data.joint_pos)
            joint_vel = wp.to_torch(robot_data.joint_vel)
            assert not torch.any(torch.isnan(joint_pos)), "Joint positions should not contain NaN"
            assert not torch.any(torch.isnan(joint_vel)), "Joint velocities should not contain NaN"

            obs, _, _, _, _ = env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
            policy_obs = obs["policy"]
            for key in ("eef_pos", "eef_quat", "left_gripper_pos", "right_gripper_pos", "joint_pos", "joint_vel"):
                assert key in policy_obs, f"Missing observation term {key}"
                assert torch.isfinite(policy_obs[key]).all(), f"Observation {key} contains non-finite values"

            # Grippers start open (Genie Sim default posture).
            right_gripper = policy_obs["right_gripper_pos"]
            assert torch.allclose(
                right_gripper, torch.full_like(right_gripper, 0.785), atol=0.05
            ), f"Right gripper should start open at 0.785 rad, got {right_gripper}"

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_g2_arm_reaches_goal(simulation_app) -> bool:
    """The right arm reaches a target in front of the robot through relative differential IK."""

    env, _ = get_g2_test_environment(num_envs=1)

    target_position = torch.tensor([0.15, -0.35, 0.3], device=env.unwrapped.device)
    position_tolerance = 0.05

    try:
        with torch.inference_mode():
            ee_frame = env.unwrapped.scene["ee_frame"]
            env_origin = env.unwrapped.scene.env_origins[0]

            initial_ee_pos = wp.to_torch(ee_frame.data.target_pos_w)[0, 0, :] - env_origin
            print(f"Initial EE position: {initial_ee_pos.cpu().numpy()}")
            print(f"Target position: {target_position.cpu().numpy()}")

            num_reach_steps = 200
            for _ in range(num_reach_steps):
                current_ee_pos = wp.to_torch(ee_frame.data.target_pos_w)[0, 0, :] - env_origin
                # Relative IK: [dx, dy, dz, droll, dpitch, dyaw, gripper]; move a fraction of the remaining distance.
                action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                action[0, :3] = (target_position - current_ee_pos) * 0.2
                env.step(action)

            final_ee_pos = wp.to_torch(ee_frame.data.target_pos_w)[0, 0, :] - env_origin
            position_error = torch.norm(final_ee_pos - target_position).item()
            print(f"Final EE position: {final_ee_pos.cpu().numpy()}, error {position_error:.4f} m")
            assert (
                position_error < position_tolerance
            ), f"Arm didn't reach target. Error: {position_error:.4f}m, tolerance: {position_tolerance}m"

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_g2_left_arm_builds(simulation_app) -> bool:
    """The left-arm variant builds and steps."""
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode

    env, _ = get_g2_test_environment(num_envs=1, arm_mode=ArmMode.LEFT)
    try:
        with torch.inference_mode():
            for _ in range(NUM_STEPS):
                env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        env.close()
    return True


@requires_g2_asset
def test_g2_registered():
    result = run_function_with_persistent_simulation_app(_test_g2_registered, headless=HEADLESS)
    assert result, f"Test {_test_g2_registered.__name__} failed"


@requires_g2_asset
def test_g2_initial_position():
    result = run_function_with_persistent_simulation_app(_test_g2_initial_position, headless=HEADLESS)
    assert result, f"Test {_test_g2_initial_position.__name__} failed"


@requires_g2_asset
def test_g2_action_and_observation_spaces():
    result = run_function_with_persistent_simulation_app(_test_g2_action_and_observation_spaces, headless=HEADLESS)
    assert result, f"Test {_test_g2_action_and_observation_spaces.__name__} failed"


@requires_g2_asset
def test_g2_arm_reaches_goal():
    result = run_function_with_persistent_simulation_app(_test_g2_arm_reaches_goal, headless=HEADLESS)
    assert result, f"Test {_test_g2_arm_reaches_goal.__name__} failed"


@requires_g2_asset
def test_g2_left_arm_builds():
    result = run_function_with_persistent_simulation_app(_test_g2_left_arm_builds, headless=HEADLESS)
    assert result, f"Test {_test_g2_left_arm_builds.__name__} failed"


if __name__ == "__main__":
    test_g2_registered()
    test_g2_initial_position()
    test_g2_action_and_observation_spaces()
    test_g2_arm_reaches_goal()
    test_g2_left_arm_builds()
