# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Open the drawer, put the block inside, close the drawer -- with the Agibot.

The USDCraft tabletop drawer cabinet stands on the RoboDojo table with its drawer front toward the
robot, so the drawer pulls out into the work band; the metal block starts on the table beside it.
The three stages are Arena's open / pick-and-place / close tasks in sequence, exactly as the
microwave task before it, but the bar handle here is pinchable (an 11 x 11 mm bar standing 20.5 mm
proud of the drawer front), so the drawer starts fully closed and the robot pulls it open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.local_objects import DrawerCabinet
from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.agibot_tabletop_common import (
    TABLE_TOP_Z,
    AgibotTabletopEnvironmentCfg,
    build_agibot,
    build_tabletop_stage,
    build_teleop_device,
    install_agibot_control_stack,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

DRAWER_FRONT_TOWARD_ROBOT_ROTATION_XYZW = (0.0, 0.0, -0.7071068, 0.7071068)
"""Yaw -90 deg: the asset's -y (drawer front) turns to world -x, toward the robot. Asset (x, y)
maps to world (cx + y, cy - x)."""

CABINET_POSITION_XY = (0.45, 0.0)
"""Cabinet origin (centre of the carcass's underside). The carcass spans world x 0.335..0.565,
y +/-0.15; the handle travels from x 0.302 (closed) to 0.142 (fully out), so the pull happens
inside the 0.15-0.30 work band, and the open drawer's accessible cavity spans x 0.18..0.335."""

_BLOCK_POSITION_XY = (0.22, -0.28)
"""The block starts in the work band on the robot's right, clear of the drawer's swept volume
(y within +/-0.137 of the cabinet's centre line)."""


def handle_world_position(
    openness: float,
    cabinet_xy: tuple[float, float] = CABINET_POSITION_XY,
    table_top_z: float = TABLE_TOP_Z,
) -> tuple[float, float, float]:
    """World position of the handle bar's centre at a given openness (0 closed .. 1 fully out)."""
    joint_pos = DrawerCabinet.DRAWER_CLOSED_JOINT_POS_M + openness * (
        DrawerCabinet.DRAWER_OPEN_JOINT_POS_M - DrawerCabinet.DRAWER_CLOSED_JOINT_POS_M
    )
    hx, hy, hz = DrawerCabinet.HANDLE_CENTRE_AT_JOINT_ZERO_M
    return (cabinet_xy[0] + hy - joint_pos, cabinet_xy[1] - hx, table_top_z + hz)


def drawer_floor_world_position(
    openness: float,
    cabinet_xy: tuple[float, float] = CABINET_POSITION_XY,
    table_top_z: float = TABLE_TOP_Z,
) -> tuple[float, float, float]:
    """World position of the centre of the drawer's floor at a given openness."""
    joint_pos = DrawerCabinet.DRAWER_CLOSED_JOINT_POS_M + openness * (
        DrawerCabinet.DRAWER_OPEN_JOINT_POS_M - DrawerCabinet.DRAWER_CLOSED_JOINT_POS_M
    )
    (_, _), (y_min, y_max), (z_floor, _) = DrawerCabinet.DRAWER_INTERIOR_LOCAL_M
    return (cabinet_xy[0] + 0.5 * (y_min + y_max) - joint_pos, cabinet_xy[1], table_top_z + z_floor)


@dataclass
class AgibotDrawerBlockEnvironmentCfg(AgibotTabletopEnvironmentCfg):
    """Configure the Agibot drawer environment."""

    arm_mode: str = "dual"

    cabinet_x: float = CABINET_POSITION_XY[0]
    cabinet_y: float = CABINET_POSITION_XY[1]
    """Cabinet origin on the table (its drawer front faces -x)."""

    object: str = "metal_block"
    """Asset that goes into the drawer. The 48 mm metal block by default."""

    object_x: float = _BLOCK_POSITION_XY[0]
    object_y: float = _BLOCK_POSITION_XY[1]
    """Where the object starts on the table."""

    reset_openness: float = 0.0
    """Drawer openness at reset (0 closed .. 1 fully out). Closed: the handle is pinchable."""

    open_threshold: float = 0.8
    """Openness above which the drawer counts as open (stage 1 done): 128 mm of its 160 mm travel."""

    closed_threshold: float = 0.1
    """Openness below which the drawer counts as closed (stage 3 done): within 16 mm of flush."""


@register_environment
class AgibotDrawerBlockEnvironment(ArenaEnvironmentFactory[AgibotDrawerBlockEnvironmentCfg]):
    """Open the drawer, put the block in, close the drawer."""

    name: str = "agibot_drawer_block"
    _legacy_argparse_cfg_type = AgibotDrawerBlockEnvironmentCfg

    def build(self, cfg: AgibotDrawerBlockEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_composite_tasks.agibot_open_put_close_door_task import (
            AgibotOpenPutCloseDoorTask,
        )
        from isaaclab_arena.utils.pose import Pose

        background, surroundings, light = build_tabletop_stage(self, cfg, TABLE_TOP_Z)
        cabinet = self.asset_registry.get_asset_by_name("drawer_cabinet")()
        cabinet.set_initial_pose(
            Pose(
                position_xyz=(cfg.cabinet_x, cfg.cabinet_y, TABLE_TOP_Z),
                rotation_xyzw=DRAWER_FRONT_TOWARD_ROBOT_ROTATION_XYZW,
            )
        )
        pick_object = self.asset_registry.get_asset_by_name(cfg.object)()
        # Bottom-origin parts (the block) sit on the table; centre-origin ones (the bowl) are lifted.
        object_half_height = getattr(type(pick_object), "HALF_HEIGHT_M", 0.0)
        pick_object.set_initial_pose(
            Pose(
                position_xyz=(cfg.object_x, cfg.object_y, TABLE_TOP_Z + object_half_height),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        # The cabinet articulation owns the drawer body; this read-only reference gives the place
        # task its live pose and geometry.
        drawer = ObjectReference(
            name="drawer",
            parent_asset=cabinet,
            prim_path="{ENV_REGEX_NS}/drawer_cabinet/Links/drawer",
        )
        embodiment = build_agibot(self, cfg)
        teleop_device = build_teleop_device(self, cfg)
        scene = Scene(assets=[background, cabinet, pick_object, drawer, surroundings, light])

        open_task = OpenDoorTask(
            openable_object=cabinet,
            openness_threshold=cfg.open_threshold,
            reset_openness=cfg.reset_openness,
            task_description="Pull the drawer open.",
        )
        put_task = PickAndPlaceTask(
            pick_up_object=pick_object,
            destination_object=cabinet,
            destination_location=drawer,
            background_scene=background,
            task_description=f"Put the {cfg.object} into the drawer.",
        )
        close_task = CloseDoorTask(
            openable_object=cabinet,
            closedness_threshold=cfg.closed_threshold,
            reset_openness=cfg.reset_openness,
            task_description="Push the drawer closed.",
        )

        def env_cfg_callback(env_cfg):
            install_agibot_control_stack(env_cfg, cfg)
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=AgibotOpenPutCloseDoorTask(
                openable_object=cabinet,
                subtasks=[open_task, put_task, close_task],
                episode_length_s=180.0,
                viewer_cfg=embodiment.get_head_viewer_cfg() if cfg.head_view else None,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
