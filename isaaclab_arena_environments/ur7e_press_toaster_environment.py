# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Press-toaster task on the UR7e workcell: a two-slot toaster on the slotted table, carriage lever facing the camera."""

from __future__ import annotations

import math
import os
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset, register_environment
from isaaclab_arena.embodiments.ur7e.ur7e import UR7E_READY_JOINT_POS
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.ur7e_workcell_environment import (
    ROBOT_BASE_XY,
    TABLE_TOP_HEIGHT_M,
    build_lights,
    build_table,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

TOASTER_USD_PATH = os.environ.get("ARENA_TOASTER_USD", "/home/ubuntu/playground/rr_ur/toast_rr_arena.usda")
"""Overlay of toast_rr.usdc that hides its two decals (their textures are missing, they rendered as black
patches) and lightens the metallic materials. Geometry, joints and physics are the original asset's."""

LEVER_TRAVEL_M = 0.047
"""Full travel of the carriage lever along the prismatic joint (0 = rest at the top, 0.047 = fully pressed)."""
LEVER_PADDLE_CENTRE_LOCAL = (-0.002, -0.116, 0.124)
"""Centre of the 48 x 20 x 9 mm lever paddle in the toaster frame, at rest. It sits on the toaster's -Y face."""
TOASTER_EXTENTS_M = (0.143, 0.252, 0.168)
"""Toaster bounding box in its own frame (x width, y depth incl. the paddle, z height)."""

# Toaster on the table. The asset's lever paddle and browning dial sit on its -Y face, which at yaw 0 faces
# the robot (mounted at the table's -Y edge). The robot then presses the lever from its own side and the
# calibrated D435 (at x = -0.55 looking +X) sees the paddle and the fingertips from the side; with the
# paddle facing the camera instead, the wrist hid the whole toaster during the press.
TOASTER_YAW_RAD = 0.0
TOASTER_FAR_XY = (-0.10, 0.14)
"""Corner of the placement band nearest the robot and farthest from the camera."""
TOASTER_NEAR_XY = (-0.22, 0.10)
"""Corner of the placement band nearest the camera and the robot.

The band was read off a D435 render grid with the toaster on its pedestal: it keeps the whole toaster in the
lower-left part of the image (further along +Y or x it leaves the frame on the left or the top, and it must
stay clear of the arm's approach path on the right of the image). With the paddle closer than about 0.43 m
to the robot base (toaster y below 0.10) every tilt folds the wrist into the shoulder and no press is planned.
"""
TOASTER_YAW_JITTER_RAD = math.radians(10.0)

PEDESTAL_ASSET = "drawer_rr"
PEDESTAL_HEIGHT_M = 0.078
"""Height of the drawer unit used as a pedestal, so the toaster's base sits this far above the table."""
PEDESTAL_YAW_OFFSET_RAD = math.pi
"""The pedestal's pull knob faces away from the robot (the drawer front is the asset's -Y face)."""


def randomize_stacked_object_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfgs: list[SceneEntityCfg],
    z_offsets: list[float],
    yaw_offsets: list[float],
) -> None:
    """Draw one pose per environment and place several assets on it, each with its own z and yaw offset.

    Args:
        env: The environment.
        env_ids: Environments being reset.
        pose_range: ``{"x": (lo, hi), "y": ..., "z": ..., "yaw": ...}`` sampled uniformly (radians for yaw).
        asset_cfgs: Assets to place, in stacking order.
        z_offsets: Height added to the sampled z for each asset.
        yaw_offsets: Yaw added to the sampled yaw for each asset (radians).
    """
    if env_ids is None:
        return
    assert len(asset_cfgs) == len(z_offsets) == len(yaw_offsets), "one z and yaw offset per asset"
    device = env.device
    n = len(env_ids)
    lo_hi = [pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "yaw")]
    sample = torch.stack([torch.empty(n, device=device).uniform_(lo, hi) for lo, hi in lo_hi], dim=-1)  # x, y, z, yaw
    origins = env.scene.env_origins[env_ids]
    for asset_cfg, dz, dyaw in zip(asset_cfgs, z_offsets, yaw_offsets):
        asset = env.scene[asset_cfg.name]
        yaw = sample[:, 3] + dyaw
        pose = torch.zeros(n, 7, device=device)
        pose[:, :2] = sample[:, :2] + origins[:, :2]
        pose[:, 2] = sample[:, 2] + dz + origins[:, 2]
        # Isaac Lab 3.0 root poses are (x, y, z, qx, qy, qz, qw).
        pose[:, 5] = torch.sin(yaw / 2)
        pose[:, 6] = torch.cos(yaw / 2)
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


@register_asset
class ToasterRR(LibraryObject, Openable):
    """A 14 x 24 x 17 cm two-slot toaster whose carriage lever slides 47 mm down a prismatic joint.

    The housing is the articulation root and is fixed in place so pressing the lever moves the carriage
    instead of tipping the toaster. The lever has no return spring in the asset, so gravity is disabled on
    its bodies: the lever stays wherever it is left, at the top after a reset and at the bottom once pressed,
    like a latched toaster carriage. The joint's own drive (damping 1 N s/m) brings it to rest.
    """

    name = "toaster_rr"
    tags = ["object", "openable"]
    usd_path = TOASTER_USD_PATH
    object_type = ObjectType.ARTICULATION
    spawn_cfg_addon = {
        "articulation_props": sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    }

    openable_joint_name = "carriage_slide"
    openable_threshold = 0.75

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


@dataclass
class Ur7ePressToasterEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the UR7e press-toaster environment."""

    embodiment: str = "ur7e_robotiq_ik"
    teleop_device: str | None = None
    light_scale: float = 1.0
    pressed_threshold: float = 0.75
    """Fraction of the 47 mm lever travel the carriage must be pushed down for success."""
    episode_length_s: float = 20.0
    randomize_toaster_pose: bool = True
    """Sample the toaster position between ``TOASTER_NEAR_XY`` and ``TOASTER_FAR_XY`` (plus yaw jitter) on every reset."""
    toaster_x: float = TOASTER_FAR_XY[0]
    toaster_y: float = TOASTER_FAR_XY[1]
    toaster_yaw_deg: float = math.degrees(TOASTER_YAW_RAD)
    toaster_z: float = TABLE_TOP_HEIGHT_M
    """Fixed toaster pose used when ``randomize_toaster_pose`` is False (``toaster_z`` is the table top;
    the pedestal height is added on top of it)."""
    pedestal: bool = True
    """Stand the toaster on the closed drawer unit, raising the lever by ``PEDESTAL_HEIGHT_M``.

    Raising the toaster lets the arm press with the wrist extended rather than folded against the shoulder.
    """


@register_environment
class Ur7ePressToasterEnvironment(ArenaEnvironmentFactory[Ur7ePressToasterEnvironmentCfg]):
    """UR7e workcell with a toaster on the table; the task is to press the carriage lever down."""

    name: str = "ur7e_press_toaster"
    _legacy_argparse_cfg_type = Ur7ePressToasterEnvironmentCfg

    def build(self, cfg: Ur7ePressToasterEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask

        table_blocks = build_table()
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        lights = build_lights(self.asset_registry, cfg.light_scale)

        toaster = self.asset_registry.get_asset_by_name("toaster_rr")()
        stack = [toaster]
        z_offsets = [PEDESTAL_HEIGHT_M if cfg.pedestal else 0.0]
        yaw_offsets = [0.0]
        if cfg.pedestal:
            pedestal = self.asset_registry.get_asset_by_name(PEDESTAL_ASSET)(instance_name="pedestal")
            stack.insert(0, pedestal)
            z_offsets.insert(0, 0.0)
            yaw_offsets.insert(0, PEDESTAL_YAW_OFFSET_RAD)
        yaw = math.radians(cfg.toaster_yaw_deg)
        for asset, dz, dyaw in zip(stack, z_offsets, yaw_offsets):
            asset.set_initial_pose(
                Pose(
                    position_xyz=(cfg.toaster_x, cfg.toaster_y, cfg.toaster_z + dz),
                    rotation_xyzw=(0.0, 0.0, math.sin((yaw + dyaw) / 2), math.cos((yaw + dyaw) / 2)),
                )
            )
        stack_event = None
        if cfg.randomize_toaster_pose:
            # One sample places the whole stack; per-asset PoseRange events would draw independently.
            stack_event = EventTermCfg(
                func=randomize_stacked_object_poses,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (TOASTER_NEAR_XY[0], TOASTER_FAR_XY[0]),
                        "y": (TOASTER_NEAR_XY[1], TOASTER_FAR_XY[1]),
                        "z": (TABLE_TOP_HEIGHT_M, TABLE_TOP_HEIGHT_M),
                        "yaw": (TOASTER_YAW_RAD - TOASTER_YAW_JITTER_RAD, TOASTER_YAW_RAD + TOASTER_YAW_JITTER_RAD),
                    },
                    "asset_cfgs": [SceneEntityCfg(asset.name) for asset in stack],
                    "z_offsets": z_offsets,
                    "yaw_offsets": yaw_offsets,
                },
            )

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(ROBOT_BASE_XY[0], ROBOT_BASE_XY[1], TABLE_TOP_HEIGHT_M)))
        # Start with the arm raised so the toaster can be placed anywhere in its range without touching it.
        embodiment.set_joint_initial_pos(UR7E_READY_JOINT_POS)

        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device is not None else None
        )

        scene = Scene(assets=[*table_blocks, *stack, ground_plane, *lights])

        # The lever's joint runs from the rest position at the top to fully pressed, so "open" == pressed.
        task = OpenDoorTask(
            openable_object=toaster,
            openness_threshold=cfg.pressed_threshold,
            reset_openness=0.0,
            episode_length_s=cfg.episode_length_s,
            task_description="Press the toaster lever down.",
        )

        def set_viewer(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.8, -1.6, 1.7), lookat=(0.0, 0.0, TABLE_TOP_HEIGHT_M))
            if stack_event is not None:
                # Appended after the per-asset pose events, so it has the last word on where the stack is.
                env_cfg.events.reset_toaster_stack = stack_event
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=set_viewer,
        )
