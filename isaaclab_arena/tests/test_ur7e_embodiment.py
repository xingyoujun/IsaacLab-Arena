# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the UR7e + Robotiq 2F-85 embodiment (no simulation)."""

import math


def test_ur7e_embodiments_registered_in_asset_registry() -> None:
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.ur7e.ur7e import (
        Ur7eRobotiqDifferentialIKEmbodiment,
        Ur7eRobotiqJointPositionEmbodiment,
    )

    reg = AssetRegistry()
    assert reg.get_asset_by_name("ur7e_robotiq_joint_pos") is Ur7eRobotiqJointPositionEmbodiment
    assert reg.get_asset_by_name("ur7e_robotiq_ik") is Ur7eRobotiqDifferentialIKEmbodiment


def test_scene_cfg_has_robot_and_ee_frame() -> None:
    from isaaclab_arena.embodiments.ur7e.ur7e import UR5E_STAND_IN_SPEC, UR7E_SPEC, make_ur7e_scene_cfg

    for spec in (UR7E_SPEC, UR5E_STAND_IN_SPEC):
        cfg = make_ur7e_scene_cfg(spec)
        assert cfg.robot.spawn.usd_path == spec.usd_path
        assert cfg.robot.init_state.joint_pos["shoulder_pan_joint"] == 1.4309711456298828
        assert cfg.ee_frame.prim_path.endswith(spec.gripper_prim + "/base_link")
        assert cfg.ee_frame.target_frames[0].name == "end_effector"


def test_joint_position_embodiment_action_and_observation() -> None:
    from isaaclab_arena.embodiments.ur7e.ur7e import UR5E_STAND_IN_SPEC, Ur7eRobotiqJointPositionEmbodiment

    emb = Ur7eRobotiqJointPositionEmbodiment(robot_spec=UR5E_STAND_IN_SPEC)
    assert emb.action_config.arm_action.joint_names == [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    assert emb.action_config.gripper_action.joint_names == ["finger_joint"]
    assert emb.observation_config.policy.concatenate_terms is False
    assert hasattr(emb.observation_config.policy, "eef_pos")
    assert emb.event_config.reset_robot_joints.mode == "reset"


def test_set_initial_joint_pose_round_trip() -> None:
    from isaaclab_arena.embodiments.ur7e.ur7e import (
        _UR7E_JOINT_NAMES,
        UR5E_STAND_IN_SPEC,
        Ur7eRobotiqJointPositionEmbodiment,
    )

    emb = Ur7eRobotiqJointPositionEmbodiment(robot_spec=UR5E_STAND_IN_SPEC)
    values = [0.1 * i for i in range(len(_UR7E_JOINT_NAMES))]
    emb.set_initial_joint_pose(values)
    assert emb.scene_config.robot.init_state.joint_pos == dict(zip(_UR7E_JOINT_NAMES, values))


def test_camera_orientation_matches_calibrated_axes() -> None:
    from isaaclab_arena.embodiments.ur7e.ur7e import (
        D435_CAMERA_LOOK_AT,
        D435_CAMERA_POS,
        D435_CAMERA_UP,
        Ur7eCameraCfg,
        look_at_quat_xyzw,
    )

    x, y, z, w = look_at_quat_xyzw(D435_CAMERA_POS, D435_CAMERA_LOOK_AT, D435_CAMERA_UP)
    assert math.isclose(x * x + y * y + z * z + w * w, 1.0, abs_tol=1e-9)
    # Rotate the camera's -Z axis into the world and compare with the calibrated optical axis.
    optical_axis = (
        -(2.0 * (x * z + w * y)),
        -(2.0 * (y * z - w * x)),
        -(1.0 - 2.0 * (x * x + y * y)),
    )
    expected = (0.6276, -0.0076, -0.7785)
    assert all(math.isclose(a, b, abs_tol=2e-3) for a, b in zip(optical_axis, expected))

    cam = Ur7eCameraCfg().realsense_d435
    assert (cam.width, cam.height) == (640, 480)
    assert cam.spawn.horizontal_aperture_offset < 0.0
    assert cam.offset.convention == "opengl"
