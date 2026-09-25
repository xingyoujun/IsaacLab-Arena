# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time-aligned physical joint and TCP observations for compact G2 recordings."""

import torch

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_arena.progress_tracking.task_success import TaskSuccessTerm


class G2CollectionSuccessTerm(TaskSuccessTerm):
    """Update and reset task progress while allowing collection to finish the release motion."""

    def __call__(
        self,
        env,
        success_objectives,
        subtasks_are_sequential=False,
        desired_subtask_success_state=None,
    ):
        success = super().__call__(env, success_objectives, subtasks_are_sequential, desired_subtask_success_state)
        return torch.zeros_like(success)


class G2CoreRecorder(RecorderTerm):
    """Capture pre-action absolute joints and right/left TCP poses in world and robot-base frames."""

    def record_pre_step(self):
        robot = self._env.scene["robot"]
        world, relative = [], []
        for name in ("ee_frame", "left_ee_frame"):
            sensor = self._env.scene[name]
            pos = sensor.data.target_pos_w.torch[:, 0]
            quat = sensor.data.target_quat_w.torch[:, 0]
            base_pos, base_quat = subtract_frame_transforms(
                robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, pos, quat
            )
            world.extend((pos, quat))
            relative.extend((base_pos, base_quat))
        return "core", {
            "joint_position": robot.data.joint_pos.torch.clone(),
            "joint_velocity": robot.data.joint_vel.torch.clone(),
            "eef_pose": torch.cat(relative, dim=-1),
            "eef_pose_world": torch.cat(world, dim=-1),
        }


def core_recorder_cfg():
    """Create the pre-action G2 recorder term configuration."""
    return RecorderTermCfg(class_type=G2CoreRecorder)
