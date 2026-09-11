# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Open-drawer task on the UR7e workcell: a small drawer unit on the slotted table, pull knob facing the camera."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset, register_environment
from isaaclab_arena.embodiments.ur7e.ur7e import UR7E_READY_JOINT_POS
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose, PoseRange
from isaaclab_arena_environments.ur7e_workcell_environment import (
    ROBOT_BASE_XY,
    TABLE_TOP_HEIGHT_M,
    build_lights,
    build_table,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

DRAWER_USD_PATH = os.environ.get("ARENA_DRAWER_USD", "/home/ubuntu/playground/rr_ur/drawer_rr_arena.usda")
"""Overlay of drawer_rr.usdc that filters the sliding links' collisions against the carcass and top panel.

With the original asset the drawer box runs inside the carcass with 1 mm of side clearance, and at
roughly half of the sampled placements PhysX jammed the slide solid (a joint written to 50 mm snapped
back to 0 before anything touched it). The slide is guided by the prismatic joint, so those contacts
carry no information; filtering them frees the drawer at every placement tested. Tighter contact
offsets did not help.
"""

DRAWER_GPT56_USD_PATH = os.environ.get(
    "ARENA_DRAWER_GPT56_USD", "/home/ubuntu/playground/rr_ur/drawer_rr_gpt56_arena.usda"
)
"""Overlay of drawer_rr_gpt56_sol_high_image_mesh.usd (the image-reconstructed comparison asset).

The overlay deactivates the PhysicsScene the asset ships with and moves the articulation root from the
plain root Xform onto the fixed carcass body so the fixed-base articulation can be anchored. Same
footprint as drawer_rr (199 x 229 x 78 mm), knob centre 7.5 mm higher; the drawer and knob are one
rigid body and the slide's travel is negative (-155 mm), which the Openable affordance normalises.
"""

# Drawer unit on the table. The asset's drawer front and pull knob face its -Y; yaw -90 deg turns
# that face towards world -X, i.e. towards the calibrated D435 (which sits at x = -0.55 looking +X).
# The prismatic joint then opens towards -X as well, so the robot pulls the knob towards the camera.
DRAWER_YAW_RAD = -math.pi / 2
DRAWER_FAR_XY = (0.18, 0.15)
"""Farthest placement (upper-left of the D435 image), fully visible and clear of the robot."""
DRAWER_NEAR_XY = (0.05, 0.0)
"""Nearest placement, towards the robot and the camera (centre of the D435 image)."""
DRAWER_YAW_JITTER_RAD = math.radians(10.0)


@register_asset
class DrawerRR(LibraryObject, Openable):
    """A 20 x 23 x 8 cm single-drawer unit with a spherical pull knob and a 15 cm prismatic travel.

    The carcass is the articulation root and is fixed in place so pulling the knob slides the drawer
    instead of dragging the whole unit.
    """

    name = "drawer_rr"
    tags = ["object", "openable"]
    usd_path = DRAWER_USD_PATH
    object_type = ObjectType.ARTICULATION
    spawn_cfg_addon = {"articulation_props": sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)}

    openable_joint_name = "carcass_to_drawer_fascia"
    openable_threshold = 0.5
    knob_body = "pull_knob"
    """Articulation body the pull knob belongs to."""
    knob_offset_local = (0.0, -0.005, 0.0)
    """Centre of the spherical grip in ``knob_body``'s frame (the asset's -y faces the operator)."""

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            openable_joint_name=self.openable_joint_name,
            openable_threshold=self.openable_threshold,
        )


@register_asset
class DrawerRRGpt56(DrawerRR):
    """The image-reconstructed comparison drawer (``drawer_rr_gpt56``), same footprint as ``drawer_rr``."""

    name = "drawer_rr_gpt56"
    usd_path = DRAWER_GPT56_USD_PATH
    openable_joint_name = "DrawerSlide"
    knob_body = "Drawer"
    knob_offset_local = (0.0, -0.1001, 0.0405)


@dataclass
class Ur7eOpenDrawerEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the UR7e open-drawer environment."""

    embodiment: str = "ur7e_robotiq_ik"
    teleop_device: str | None = None
    light_scale: float = 1.0
    drawer_asset: str = "drawer_rr"
    """Registered drawer asset to place on the table (``drawer_rr`` or ``drawer_rr_gpt56``)."""
    openness_threshold: float = 0.5
    """Fraction of the 15 cm travel the drawer must be pulled out for success."""
    episode_length_s: float = 20.0
    randomize_drawer_pose: bool = True
    """Sample the drawer position between ``DRAWER_NEAR_XY`` and ``DRAWER_FAR_XY`` (plus yaw jitter) on every reset."""
    drawer_x: float = DRAWER_FAR_XY[0]
    drawer_y: float = DRAWER_FAR_XY[1]
    drawer_yaw_deg: float = math.degrees(DRAWER_YAW_RAD)
    drawer_z: float = TABLE_TOP_HEIGHT_M
    """Fixed drawer pose used when ``randomize_drawer_pose`` is False."""


@register_environment
class Ur7eOpenDrawerEnvironment(ArenaEnvironmentFactory[Ur7eOpenDrawerEnvironmentCfg]):
    """UR7e workcell with a drawer unit on the table; the task is to pull the drawer open."""

    name: str = "ur7e_open_drawer"
    _legacy_argparse_cfg_type = Ur7eOpenDrawerEnvironmentCfg

    def build(self, cfg: Ur7eOpenDrawerEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask

        table_blocks = build_table()
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        lights = build_lights(self.asset_registry, cfg.light_scale)

        drawer = self.asset_registry.get_asset_by_name(cfg.drawer_asset)()
        if cfg.randomize_drawer_pose:
            drawer.set_initial_pose(
                PoseRange(
                    position_xyz_min=(DRAWER_NEAR_XY[0], DRAWER_NEAR_XY[1], TABLE_TOP_HEIGHT_M),
                    position_xyz_max=(DRAWER_FAR_XY[0], DRAWER_FAR_XY[1], TABLE_TOP_HEIGHT_M),
                    rpy_min=(0.0, 0.0, DRAWER_YAW_RAD - DRAWER_YAW_JITTER_RAD),
                    rpy_max=(0.0, 0.0, DRAWER_YAW_RAD + DRAWER_YAW_JITTER_RAD),
                )
            )
        else:
            yaw = math.radians(cfg.drawer_yaw_deg)
            drawer.set_initial_pose(
                Pose(
                    position_xyz=(cfg.drawer_x, cfg.drawer_y, cfg.drawer_z),
                    rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
                )
            )

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(ROBOT_BASE_XY[0], ROBOT_BASE_XY[1], TABLE_TOP_HEIGHT_M)))
        # Start with the arm raised so the drawer can be placed anywhere in its range without touching it.
        embodiment.set_joint_initial_pos(UR7E_READY_JOINT_POS)

        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device is not None else None
        )

        scene = Scene(assets=[*table_blocks, drawer, ground_plane, *lights])

        task = OpenDoorTask(
            openable_object=drawer,
            openness_threshold=cfg.openness_threshold,
            reset_openness=0.0,
            episode_length_s=cfg.episode_length_s,
            task_description="Grasp the knob and pull the drawer open.",
        )

        def set_viewer(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.8, -1.6, 1.7), lookat=(0.0, 0.0, TABLE_TOP_HEIGHT_M))
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=set_viewer,
        )


@dataclass
class Ur7eOpenDrawerGpt56EnvironmentCfg(Ur7eOpenDrawerEnvironmentCfg):
    """The open-drawer environment with the image-reconstructed comparison drawer."""

    drawer_asset: str = "drawer_rr_gpt56"


@register_environment
class Ur7eOpenDrawerGpt56Environment(
    Ur7eOpenDrawerEnvironment, ArenaEnvironmentFactory[Ur7eOpenDrawerGpt56EnvironmentCfg]
):
    """Same workcell, layout and task as ``ur7e_open_drawer`` with ``drawer_rr_gpt56`` as the drawer."""

    name: str = "ur7e_open_drawer_gpt56"
    _legacy_argparse_cfg_type = Ur7eOpenDrawerGpt56EnvironmentCfg
