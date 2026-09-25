# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physics-level checks of the G2 embodiment: joint drives, collision setup, gripper configuration and grasping."""

import math
import os
import torch
import traceback

import pytest
import warp as wp

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True

_G2_USD = os.path.join(
    os.environ.get("GENIESIM_ASSETS_DIR", "/datasets/GenieSimAssets"), "robot", "G2_omnipicker", "robot_fix.usda"
)
requires_g2_asset = pytest.mark.skipif(
    not os.path.isfile(_G2_USD),
    reason=f"G2 omnipicker asset not found at {_G2_USD}; set GENIESIM_ASSETS_DIR to a GenieSimAssets checkout",
)

ARM_R_JOINTS = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
GRIPPER_R_JOINTS = ["idx81_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1"]


def _joint_ids(robot, names):
    return [robot.joint_names.index(n) for n in names]


def get_g2_grasp_test_environment(num_envs: int = 1):
    """G2 on the Genie table with one small cube spawned in front of the robot for grasp tests."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.embodiments.g2.g2 import G2Embodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.utils.pose import Pose

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(num_envs)])
    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("genie_benchmark_table")()
    ground_plane = asset_registry.get_asset_by_name("ground_plane")(
        initial_pose=Pose(position_xyz=(0.0, 0.0, -0.74), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    )
    cube = asset_registry.get_asset_by_name("dex_cube")()
    cube.set_initial_pose(Pose(position_xyz=(0.0, -0.3, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    embodiment = G2Embodiment(arm_mode=ArmMode.RIGHT)
    embodiment.set_initial_pose(Pose(position_xyz=(-0.75, 0.0, -0.74), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    env = IsaacLabArenaEnvironment(
        name="g2_grasp_test",
        embodiment=embodiment,
        scene=Scene(assets=[background, ground_plane, cube]),
        task=NoTask(),
    )
    env = ArenaEnvBuilder(env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    env.reset()
    return env


def _zero_action(env):
    return torch.zeros(env.action_space.shape, device=env.unwrapped.device)


def _move_ee_to(env, target_pos_b, steps: int = 120, gripper: float = 1.0, tol: float = 0.01) -> float:
    """Drive the right TCP to ``target_pos_b`` (robot base frame) with relative IK; returns the final error [m]."""
    ee_frame = env.unwrapped.scene["ee_frame"]
    robot = env.unwrapped.scene["robot"]
    for _ in range(steps):
        ee_pos_b = wp.to_torch(ee_frame.data.target_pos_w)[0, 0] - wp.to_torch(robot.data.root_pos_w)[0]
        delta = target_pos_b - ee_pos_b
        if torch.norm(delta) < tol:
            break
        step = delta * 0.5
        if torch.norm(step) > 0.04:
            step = step / torch.norm(step) * 0.04
        action = _zero_action(env)
        action[0, :3] = 2.0 * step  # the IK term scales deltas by 0.5
        action[0, 6] = gripper
        env.step(action)
    ee_pos_b = wp.to_torch(ee_frame.data.target_pos_w)[0, 0] - wp.to_torch(robot.data.root_pos_w)[0]
    return torch.norm(target_pos_b - ee_pos_b).item()


def _hold(env, steps: int, gripper: float):
    for _ in range(steps):
        action = _zero_action(env)
        action[0, 6] = gripper
        env.step(action)


def _test_g2_joint_drives(simulation_app) -> bool:
    """Actuator gains and limits resolve to the vendor USD values for every joint group."""
    from isaaclab_arena.embodiments.g2.g2 import G2_OMNIPICKER_CFG

    env = get_g2_grasp_test_environment()
    try:
        robot = env.unwrapped.scene["robot"]
        assert robot.num_joints == 46, f"G2 omnipicker should expose 46 joints, got {robot.num_joints}"
        covered = sum(len(act.joint_names) for act in robot.actuators.values())
        assert covered == robot.num_joints, f"actuators cover {covered}/{robot.num_joints} joints"

        expected = {
            "arms": (10000.0, 1000.0, 60.0, 14),
            "waist": (100000.0, 10000.0, 100.0, 5),
            "head": (500.0, 50.0, 50.0, 3),
            "grippers": (100.0, 20.0, 15.0, 4),
            "gripper_passive": (0.0, 0.02, 10.0, 12),
        }
        for group, (stiffness, damping, effort, count) in expected.items():
            actuator = robot.actuators[group]
            assert len(actuator.joint_names) == count, f"{group}: {len(actuator.joint_names)} joints, want {count}"
            assert torch.allclose(actuator.stiffness, torch.full_like(actuator.stiffness, stiffness)), group
            assert torch.allclose(actuator.damping, torch.full_like(actuator.damping, damping)), group
            effort_limits = robot.data.joint_effort_limits.torch[:, _joint_ids(robot, actuator.joint_names)]
            assert torch.allclose(effort_limits, torch.full_like(effort_limits, effort)), group

        arm_vel_limit = robot.data.joint_vel_limits.torch[:, _joint_ids(robot, robot.actuators["arms"].joint_names)]
        assert torch.allclose(arm_vel_limit, torch.full_like(arm_vel_limit, math.pi)), "arm velocity limit"

        # Joint limits come from the USD (degrees there, radians here).
        limits = wp.to_torch(robot.data.joint_pos_limits)[0]
        outer = robot.joint_names.index("idx81_gripper_r_outer_joint1")
        inner = robot.joint_names.index("idx71_gripper_r_inner_joint1")
        assert torch.allclose(limits[outer], torch.tensor([0.0, math.radians(45.0)], device=limits.device), atol=1e-3)
        assert torch.allclose(limits[inner], torch.tensor([math.radians(-45.0), 0.0], device=limits.device), atol=1e-3)

        # The arm holds the default posture under the PD drive (gravity disabled): no sag or drift.
        _hold(env, 30, gripper=1.0)
        arm_ids = _joint_ids(robot, ARM_R_JOINTS)
        joint_pos = wp.to_torch(robot.data.joint_pos)[0, arm_ids]
        default_pos = wp.to_torch(robot.data.default_joint_pos)[0, arm_ids]
        err = (joint_pos - default_pos).abs().max().item()
        print(f"Right arm hold error after 30 steps: {err:.4f} rad")
        assert err < 0.01, f"arm drifts from its default posture: max error {err:.4f} rad"
        assert G2_OMNIPICKER_CFG.spawn.articulation_props.enabled_self_collisions is False
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_g2_arm_tracks_ik_targets(simulation_app) -> bool:
    """Relative IK moves the TCP through a small square of waypoints with centimetre accuracy."""
    env = get_g2_grasp_test_environment()
    try:
        ee_frame = env.unwrapped.scene["ee_frame"]
        robot = env.unwrapped.scene["robot"]
        start = wp.to_torch(ee_frame.data.target_pos_w)[0, 0] - wp.to_torch(robot.data.root_pos_w)[0]
        print(f"TCP start (base frame): {start.tolist()}")
        offsets = [(0.1, 0.0, 0.05), (0.1, -0.1, 0.05), (0.0, -0.1, 0.15), (0.0, 0.0, 0.0)]
        for dx, dy, dz in offsets:
            target = start + torch.tensor([dx, dy, dz], device=start.device)
            err = _move_ee_to(env, target)
            print(f"waypoint {(dx, dy, dz)}: final error {err:.4f} m")
            assert err < 0.02, f"IK tracking error {err:.4f} m at offset {(dx, dy, dz)}"
        joint_vel = wp.to_torch(robot.data.joint_vel)[0]
        assert torch.isfinite(joint_vel).all(), "joint velocities must stay finite"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_g2_collision_setup(simulation_app) -> bool:
    """Robot links carry collision geometry, the finger pads included, and the robot rests on the table
    without penetrating it (fixed base, no contact-driven drift)."""
    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics

    env = get_g2_grasp_test_environment()
    try:
        stage = sim_utils.get_current_stage()
        robot_root = "/World/envs/env_0/Robot"
        links_with_collision = set()
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if path.startswith(robot_root) and prim.HasAPI(UsdPhysics.CollisionAPI):
                link = path[len(robot_root) + 1 :].split("/")[0]
                links_with_collision.add(link)
        print(f"Links with collision geometry: {len(links_with_collision)}")
        required = [
            "gripper_r_inner_link1",
            "gripper_r_outer_link1",
            "gripper_r_inner_link2",
            "gripper_r_outer_link2",
            "arm_r_link7",
            "body_link5",
            "head_link3",
        ]
        missing = [name for name in required if name not in links_with_collision]
        assert not missing, f"links without collision geometry: {missing}"

        # A physics material with friction 1.0 / combine mode max was applied at spawn.
        material_prim = stage.GetPrimAtPath(f"{robot_root}/material")
        assert material_prim.IsValid(), "robot physics material prim missing"
        assert math.isclose(material_prim.GetAttribute("physics:staticFriction").Get(), 1.0)
        assert material_prim.GetAttribute("physxMaterial:frictionCombineMode").Get() == "max"

        # Fixed base: root pose does not drift under contact with the table over 60 steps.
        robot = env.unwrapped.scene["robot"]
        root_before = wp.to_torch(robot.data.root_pos_w)[0].clone()
        _hold(env, 60, gripper=1.0)
        root_after = wp.to_torch(robot.data.root_pos_w)[0]
        drift = torch.norm(root_after - root_before).item()
        assert drift < 1e-4, f"fixed base drifted {drift:.5f} m"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_g2_gripper_open_close(simulation_app) -> bool:
    """The binary gripper action opens both finger joints to +/-45 deg and closes them to 0."""
    env = get_g2_grasp_test_environment()
    try:
        robot = env.unwrapped.scene["robot"]
        ids = _joint_ids(robot, GRIPPER_R_JOINTS)
        _hold(env, 20, gripper=1.0)
        opened = wp.to_torch(robot.data.joint_pos)[0, ids]
        print(f"open: {opened.tolist()}")
        assert abs(opened[0].item() - 0.785) < 0.03 and abs(opened[1].item() + 0.785) < 0.03, "gripper not open"

        _hold(env, 30, gripper=-1.0)
        closed = wp.to_torch(robot.data.joint_pos)[0, ids]
        print(f"closed: {closed.tolist()}")
        assert closed.abs().max().item() < 0.03, "gripper not closed"

        _hold(env, 30, gripper=1.0)
        reopened = wp.to_torch(robot.data.joint_pos)[0, ids]
        assert abs(reopened[0].item() - 0.785) < 0.03, "gripper did not reopen"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _test_g2_gripper_grasps_and_lifts_cube(simulation_app) -> bool:
    """Grasp a 4 cm cube from the table top with the right gripper and lift it 15 cm; the cube must follow."""
    env = get_g2_grasp_test_environment()
    try:
        robot = env.unwrapped.scene["robot"]
        cube = env.unwrapped.scene["dex_cube"]
        ids = _joint_ids(robot, GRIPPER_R_JOINTS)
        robot_root = wp.to_torch(robot.data.root_pos_w)[0]

        _hold(env, 10, gripper=1.0)
        cube_pos_w = wp.to_torch(cube.data.root_pos_w)[0].clone()
        cube_pos_b = cube_pos_w - robot_root
        print(f"cube (base frame): {cube_pos_b.tolist()}")
        up = torch.tensor([0.0, 0.0, 1.0], device=cube_pos_b.device)

        err = _move_ee_to(env, cube_pos_b + 0.15 * up, gripper=1.0)
        assert err < 0.02, f"pre-grasp pose not reached ({err:.3f} m)"
        err = _move_ee_to(env, cube_pos_b + 0.005 * up, gripper=1.0, steps=80)
        print(f"grasp pose error: {err:.4f} m")
        assert err < 0.03, f"grasp pose not reached ({err:.3f} m)"

        _hold(env, 30, gripper=-1.0)
        fingers = wp.to_torch(robot.data.joint_pos)[0, ids]
        print(f"fingers on cube: {fingers.tolist()}")
        # Fingers are blocked by the cube: they must stop well before the fully closed 0 rad.
        assert fingers[0].item() > 0.05, "fingers closed completely; the cube was not between them"

        lift_target = cube_pos_b + 0.20 * up
        _move_ee_to(env, lift_target, gripper=-1.0, steps=120)
        _hold(env, 15, gripper=-1.0)
        cube_z_after = wp.to_torch(cube.data.root_pos_w)[0, 2].item()
        lifted = cube_z_after - cube_pos_w[2].item()
        print(f"cube lifted by {lifted:.3f} m")
        assert lifted > 0.10, f"cube did not follow the gripper (lifted {lifted:.3f} m)"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


@requires_g2_asset
def test_g2_joint_drives():
    assert run_function_with_persistent_simulation_app(_test_g2_joint_drives, headless=HEADLESS)


@requires_g2_asset
def test_g2_arm_tracks_ik_targets():
    assert run_function_with_persistent_simulation_app(_test_g2_arm_tracks_ik_targets, headless=HEADLESS)


@requires_g2_asset
def test_g2_collision_setup():
    assert run_function_with_persistent_simulation_app(_test_g2_collision_setup, headless=HEADLESS)


@requires_g2_asset
def test_g2_gripper_open_close():
    assert run_function_with_persistent_simulation_app(_test_g2_gripper_open_close, headless=HEADLESS)


@requires_g2_asset
def test_g2_gripper_grasps_and_lifts_cube():
    assert run_function_with_persistent_simulation_app(_test_g2_gripper_grasps_and_lifts_cube, headless=HEADLESS)
