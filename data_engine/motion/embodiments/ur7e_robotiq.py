# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""UR7e/Robotiq profiles, including Pine WM wrist-camera collision hardware."""

import pathlib

import isaaclab_arena
from data_engine.motion.cumotion.cumotion_embodiment_cfg import CumotionEmbodimentCfg

_ARENA_UR7E_RMPFLOW_DIR = pathlib.Path(isaaclab_arena.__file__).parent / "embodiments" / "ur7e" / "rmpflow"

_UR7E_SELF_COLLISION_IGNORE = {
    "base_link_inertia": ["shoulder_link", "upper_arm_link"],
    "shoulder_link": ["upper_arm_link", "forearm_link"],
    "upper_arm_link": ["forearm_link", "wrist_1_link"],
    "forearm_link": ["wrist_1_link", "wrist_2_link"],
    "wrist_1_link": ["wrist_2_link", "wrist_3_link", "tool0"],
    "wrist_2_link": ["wrist_3_link", "tool0"],
    "wrist_3_link": ["tool0"],
}


def create_ur7e_robotiq_cfg() -> CumotionEmbodimentCfg:
    from isaaclab_arena.embodiments.ur7e.observations import (
        GRIPPER_CLOSED_JOINT_POS,
        GRIPPER_DRIVE_JOINT,
        UR_ARM_JOINT_NAMES,
    )
    from isaaclab_arena.embodiments.ur7e.ur7e import select_ur_robot_spec

    return CumotionEmbodimentCfg(
        lula_robot_description=str(_ARENA_UR7E_RMPFLOW_DIR / "ur7e_robotiq.yaml"),
        robot_urdf=str(_ARENA_UR7E_RMPFLOW_DIR / "ur7e_robotiq.urdf"),
        # cuMotion drives the flange frame; the Robotiq base link sits on it in the simulator. The
        # tool centre point is 0.1628 m further along +z (TCP_OFFSET_FROM_GRIPPER_BASE_M).
        tool_frame="tool0",
        sim_tool_body=select_ur_robot_spec().gripper_base_body_name,
        arm_joint_names=list(UR_ARM_JOINT_NAMES),
        gripper_joint_names=[GRIPPER_DRIVE_JOINT],
        gripper_open_pos=0.0,
        gripper_closed_pos=float(GRIPPER_CLOSED_JOINT_POS),
        self_collision_ignore=_UR7E_SELF_COLLISION_IGNORE,
        # Robotiq 2F-85: fingers along the gripper base +z, jaws separate along its y (URDF knuckle
        # joints sit at y = -/+ 0.0306). Checked at run time by ur7e_open_drawer_cumotion.py.
        tool_approach_axis="+z",
        jaw_axis="+y",
        gripper_ramp_seconds=0.5,
    )


def create_pine_wm_cfg() -> CumotionEmbodimentCfg:
    """Use a separate gripper-origin frame and include pine_wm camera hardware in planning."""
    from copy import deepcopy

    cfg = deepcopy(create_ur7e_robotiq_cfg())
    cfg.robot_urdf = str(_ARENA_UR7E_RMPFLOW_DIR / "pine_wm_ur7e.urdf")
    cfg.lula_robot_description = str(_ARENA_UR7E_RMPFLOW_DIR / "pine_wm_ur7e.yaml")
    cfg.tool_frame = "pine_wm_gripper_base"
    cfg.sim_tool_body = "base_link"
    for link in ("wrist_1_link", "wrist_2_link", "wrist_3_link"):
        cfg.self_collision_ignore[link].append("pine_wm_gripper_base")
    cfg.self_collision_ignore["tool0"] = ["pine_wm_gripper_base"]
    return cfg
