# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Legacy Agibot hardware profiles retained for existing tasks; distinct from G2."""

import os
import pathlib

import isaaclab_arena
from data_engine.motion.cumotion.cumotion_embodiment_cfg import CumotionEmbodimentCfg

# Isaac Lab downloads its RMPFlow assets from Nucleus, but cuMotion's loader reads real files off
# disk, so the local Isaac asset cache is used directly (for the URDF). The Lula robot
# descriptions come from Arena's own copies instead: Arena's default Agibot rest pose mirrors
# the left wrist (see agibot.py), and cuMotion has to read descriptions that state that pose --
# planning one arm against the stock file would place the *other* arm 0.77 rad away from where
# it actually is, and that arm is a fixed obstacle in the collision model.
ISAAC_ASSET_ROOT = os.environ.get("ISAAC_ASSET_ROOT", "/tmp/Assets/Isaac/6.0/Isaac")
_AGIBOT_RMPFLOW_DIR = f"{ISAAC_ASSET_ROOT}/IsaacLab/Controllers/RmpFlowAssets/agibot"
_ARENA_AGIBOT_RMPFLOW_DIR = pathlib.Path(isaaclab_arena.__file__).parent / "embodiments" / "agibot" / "rmpflow"

# The Agibot's arm links are named Link<n>_<side>; ``left_base_link`` is the *hand* base (child of
# Link7_l through the fixed Joint_hand_l), not the arm mount, which is ``base_link_l``.
_AGIBOT_LEFT_SELF_COLLISION_IGNORE = {
    "Link1_l": ["Link2_l", "Link3_l"],
    "Link2_l": ["Link3_l", "Link4_l"],
    "Link3_l": ["Link4_l", "Link5_l"],
    "Link4_l": ["Link5_l", "Link6_l"],
    "Link5_l": ["Link6_l", "Link7_l", "left_base_link"],
    "Link6_l": ["Link7_l", "left_base_link"],
    "Link7_l": ["left_base_link"],
}

_AGIBOT_RIGHT_SELF_COLLISION_IGNORE = {
    "Link1_r": ["Link2_r", "Link3_r"],
    "Link2_r": ["Link3_r", "Link4_r"],
    "Link3_r": ["Link4_r", "Link5_r"],
    "Link4_r": ["Link5_r", "Link6_r"],
    "Link5_r": ["Link6_r", "Link7_r", "right_base_link"],
    "Link6_r": ["Link7_r", "right_base_link"],
    "Link7_r": ["right_base_link"],
}

AGIBOT_LEFT_ARM_CUMOTION_CFG = CumotionEmbodimentCfg(
    lula_robot_description=str(_ARENA_AGIBOT_RMPFLOW_DIR / "agibot_left_arm_gripper.yaml"),
    robot_urdf=f"{_AGIBOT_RMPFLOW_DIR}/agibot.urdf",
    tool_frame="gripper_center",
    arm_joint_names=[f"left_arm_joint{i}" for i in range(1, 8)],
    gripper_joint_names=["left_hand_joint1", "left_.*_Support_Joint"],
    gripper_open_pos=0.994,
    gripper_closed_pos=0.0,
    self_collision_ignore=_AGIBOT_LEFT_SELF_COLLISION_IGNORE,
    # Both measured with scripts/probe_gripper_axes.py, not read off a rest pose: the wrist-to-tool
    # vector is +z in tool coordinates and the jaws separate along y, on both hands.
    tool_approach_axis="+z",
    jaw_axis="+y",
)

AGIBOT_RIGHT_ARM_CUMOTION_CFG = CumotionEmbodimentCfg(
    lula_robot_description=str(_ARENA_AGIBOT_RMPFLOW_DIR / "agibot_right_arm_gripper.yaml"),
    robot_urdf=f"{_AGIBOT_RMPFLOW_DIR}/agibot.urdf",
    tool_frame="right_gripper_center",
    arm_joint_names=[f"right_arm_joint{i}" for i in range(1, 8)],
    gripper_joint_names=["right_hand_joint1", "right_.*_Support_Joint"],
    gripper_open_pos=0.994,
    gripper_closed_pos=0.0,
    self_collision_ignore=_AGIBOT_RIGHT_SELF_COLLISION_IGNORE,
    tool_approach_axis="+z",
    jaw_axis="+y",
)
