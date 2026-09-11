# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the UR7e + Robotiq 2F-85 embodiment."""

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

UR_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
"""The six UR arm joints in kinematic-chain order."""

GRIPPER_DRIVE_JOINT = "finger_joint"
"""The single actuated Robotiq 2F-85 joint; the other finger joints are passive mimics."""

GRIPPER_CLOSED_JOINT_POS = torch.pi / 4
"""finger_joint angle (rad) commanded for a closed gripper; also the 1.0 point of ``gripper_pos``."""


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the six arm joint positions in ``UR_ARM_JOINT_NAMES`` order."""
    robot = env.scene[asset_cfg.name]
    joint_indices = [robot.data.joint_names.index(name) for name in UR_ARM_JOINT_NAMES]
    return wp.to_torch(robot.data.joint_pos)[:, joint_indices]


def gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the gripper opening as one value per env, 0 for open and 1 for closed."""
    robot = env.scene[asset_cfg.name]
    joint_index = robot.data.joint_names.index(GRIPPER_DRIVE_JOINT)
    return wp.to_torch(robot.data.joint_pos)[:, joint_index : joint_index + 1] / GRIPPER_CLOSED_JOINT_POS


def ee_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Return the tool-centre-point position (x, y, z) in the world frame from the ``ee_frame`` sensor."""
    ee_frame = env.scene[ee_frame_cfg.name]
    return wp.to_torch(ee_frame.data.target_pos_w)[:, 0, :]


def ee_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Return the tool-centre-point orientation quaternion in the world frame from the ``ee_frame`` sensor."""
    ee_frame = env.scene[ee_frame_cfg.name]
    return wp.to_torch(ee_frame.data.target_quat_w)[:, 0, :]
