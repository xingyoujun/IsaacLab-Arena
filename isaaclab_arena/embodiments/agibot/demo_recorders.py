# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Demo recorder terms that give teleoperated and scripted Agibot demonstrations one action label.

A teleoperated demonstration records what the operator commanded -- relative end-effector deltas
plus a binary gripper (14 values) -- while a cuMotion demonstration records absolute joint targets
(20 values). The robot, physics, observations and states are identical on both paths; only that
label differs, and it is what keeps the two kinds of demonstration out of one training set.

What both paths have in common underneath is the joint position target the drives were actually
given: RMPFlow writes one every physics substep from the operator's delta, and the joint-space
action term writes the planner's target directly. ``PostStepJointPositionTargetRecorder`` records
that target, for the 20 joints in ``AgibotDualArmJointActionsCfg`` order, at the end of every
control step under the key ``joint_pos_target``. For a cuMotion demo it equals ``actions``; for a
teleop demo it is the joint-space action the operator's motion amounted to. The LeRobot conversion
reads this key, so demonstrations from either path convert to the same action space.
"""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.agibot.agibot import AgibotDualArmJointActionsCfg
from isaaclab_arena.utils.isaaclab_utils.recorders import (
    PostStepFlatPolicyActionObservationRecorderCfg,
    PreStepFlatCameraObservationsRecorderCfg,
)

JOINT_POS_TARGET_KEY = "joint_pos_target"
"""HDF5 key of the recorded joint position targets."""


def agibot_action_joint_patterns() -> list[list[str]]:
    """The joint-name patterns of the four joint-space action terms, in action-vector order."""
    cfg = AgibotDualArmJointActionsCfg()
    return [
        list(cfg.left_arm_action.joint_names),
        list(cfg.left_gripper_action.joint_names),
        list(cfg.right_arm_action.joint_names),
        list(cfg.right_gripper_action.joint_names),
    ]


class PostStepJointPositionTargetRecorder(RecorderTerm):
    """Record the joint position targets applied in the last substep of each control step.

    The joints are the ones ``AgibotDualArmJointActionsCfg`` drives, resolved term by term with
    ``preserve_order`` so the columns match the 20-dim joint-space action vector exactly.
    """

    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)
        robot = env.scene.articulations[cfg.asset_name]
        self._joint_ids: list[int] = []
        for patterns in agibot_action_joint_patterns():
            ids, _ = robot.find_joints(patterns, preserve_order=True)
            self._joint_ids.extend(int(i) for i in (ids.torch.tolist() if hasattr(ids, "torch") else ids))
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
class AgibotDemoRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Arena's demo recorder plus the joint position target label, for every Agibot recording.

    Used both by ``record_demos.py`` (registered as the environment's ``demo_recorder_config``)
    and by the cuMotion drivers, so the two kinds of demonstration carry the same keys.
    """

    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()
    """Camera images; set to None when the environment has no cameras (states-only recording)."""

    record_post_step_flat_policy_action_observations = PostStepFlatPolicyActionObservationRecorderCfg()

    record_post_step_joint_pos_target = PostStepJointPositionTargetRecorderCfg()


def agibot_demo_recorder_cfg(with_cameras: bool) -> AgibotDemoRecorderManagerCfg:
    """The recorder manager config for an Agibot recording, with or without camera streams."""
    cfg = AgibotDemoRecorderManagerCfg()
    if not with_cameras:
        cfg.record_pre_step_flat_camera_observations = None
    return cfg
