# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Demo recorder terms for UR7e demonstrations: Arena's action/state recorder plus ``joint_pos_target``.

``joint_pos_target`` holds the joint position targets the drives were actually given at the end of
each control step, for the 7 joints of ``Ur7eJointRecordingActionsCfg`` in action-vector order. It
is the label that lets scripted (cuMotion) and any future teleoperated demonstrations share one
action space (see the Agibot counterpart, ``isaaclab_arena.embodiments.agibot.demo_recorders``).
"""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.ur7e.observations import GRIPPER_DRIVE_JOINT, UR_ARM_JOINT_NAMES
from isaaclab_arena.utils.isaaclab_utils.recorders import (
    PostStepFlatPolicyActionObservationRecorderCfg,
    PreStepFlatCameraObservationsRecorderCfg,
)

JOINT_POS_TARGET_KEY = "joint_pos_target"
"""HDF5 key of the recorded joint position targets."""

UR7E_ACTION_JOINT_NAMES: tuple[str, ...] = UR_ARM_JOINT_NAMES + (GRIPPER_DRIVE_JOINT,)
"""Joints of the 7-dim joint-space action vector, in order."""


class PostStepJointPositionTargetRecorder(RecorderTerm):
    """Record the joint position targets applied in the last substep of each control step."""

    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)
        robot = env.scene.articulations[cfg.asset_name]
        ids, _ = robot.find_joints(list(UR7E_ACTION_JOINT_NAMES), preserve_order=True)
        self._joint_ids = [int(i) for i in (ids.torch.tolist() if hasattr(ids, "torch") else ids)]
        self._robot = robot

    def record_post_step(self):
        targets = self._robot.data.joint_pos_target.torch[:, self._joint_ids]
        return JOINT_POS_TARGET_KEY, targets.clone()


@configclass
class PostStepJointPositionTargetRecorderCfg(RecorderTermCfg):
    """Configuration for the joint position target recorder."""

    class_type: type[RecorderTerm] = PostStepJointPositionTargetRecorder

    asset_name: str = "robot"
    """Scene key of the articulation whose targets are recorded."""


@configclass
class Ur7eDemoRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Arena's demo recorder plus the joint position target label, for UR7e recordings."""

    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()
    """Camera images; set to None for states-only recording."""

    record_post_step_flat_policy_action_observations = PostStepFlatPolicyActionObservationRecorderCfg()

    record_post_step_joint_pos_target = PostStepJointPositionTargetRecorderCfg()


def ur7e_demo_recorder_cfg(with_cameras: bool) -> Ur7eDemoRecorderManagerCfg:
    """The recorder manager config for a UR7e recording, with or without camera streams."""
    cfg = Ur7eDemoRecorderManagerCfg()
    if not with_cameras:
        cfg.record_pre_step_flat_camera_observations = None
    return cfg
