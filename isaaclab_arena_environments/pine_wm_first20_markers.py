# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visible task cues whose poses are included in recorded and replayed scene state."""

import math
import numpy as np
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose


def marker_specs(task):
    """Return marker sizes without introducing collision geometry or target-slot affordances."""
    if task in {"T012", "T021", "T031"}:
        return [(0.12, 0.003, 0.0005)] * 4
    if task == "T014":
        return [(0.05, 0.004, 0.0005), (0.017, 0.004, 0.0005), (0.017, 0.004, 0.0005)]
    if task == "T143":
        return [(0.56, 0.003, 0.0005), (0.017, 0.004, 0.0005), (0.017, 0.004, 0.0005)]
    if task == "T017":
        return [(0.03, 0.02, 0.0005), (0.03, 0.02, 0.0005)]
    return []


def add_markers(arena, task):
    """Add collision-free kinematic cues so standard state recording captures every pose."""
    for i, size in enumerate(marker_specs(task)):
        color = (0.95, 0.45, 0.04) if i != 1 or task != "T017" else (0.02, 0.65, 0.85)
        arena.scene.add_asset(
            Object(
                name=f"task_marker_{i}",
                object_type=ObjectType.RIGID,
                initial_pose=Pose(position_xyz=(0, 0, 0.70)),
                spawner_cfg=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
                ),
            )
        )


def place_markers(task, goals, pose, put):
    """Place target frames, direction arrows and distinguishable reference-box labels."""

    def marker(i, point, yaw=0, rotation=None):
        q = Rotation.from_euler("z", yaw).as_quat() if rotation is None else rotation
        put(f"task_marker_{i}", point, q)

    if task in {"T012", "T021", "T031"}:
        xy = goals["target"]
        z = 0.7406
        if task == "T021":
            z = pose("push_mat")[2] + 0.0036
        for i, (dx, dy, yaw) in enumerate(
            [(0, -0.06, 0), (0, 0.06, 0), (-0.06, 0, math.pi / 2), (0.06, 0, math.pi / 2)]
        ):
            marker(i, [xy[0] + dx, xy[1] + dy, z], yaw)
    elif task in {"T014", "T143"}:
        yaw = goals["target_yaw"] if task == "T014" else 0.0
        R = Rotation.from_euler("z", yaw)
        c = pose("coaster")[:3] + R.apply([0.09, 0, 0]) if task == "T014" else np.array([*goals["target"], 0.7406])
        c[2] = 0.7406
        marker(0, c, yaw)
        tip = 0.025 if task == "T014" else 0.28
        marker(1, c + R.apply([tip - 0.006, 0.006, 0]), yaw - math.pi / 4)
        marker(2, c + R.apply([tip - 0.006, -0.006, 0]), yaw + math.pi / 4)
    elif task == "T017":
        for i, name in enumerate(["box_a", "box_b"]):
            p = pose(name)
            marker(
                i,
                p[:3] + [0, goals.get("box_marker_offset_y_m", -0.0906), goals.get("box_marker_height_m", 0.05)],
                rotation=Rotation.from_euler("x", math.pi / 2).as_quat(),
            )
