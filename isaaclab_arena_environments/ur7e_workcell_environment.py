# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The UR7e + Robotiq 2F-85 workcell replicated from ``rr_ur/scene.py``.

A 1 m square slotted aluminium-profile table with its top at 0.75 m, the robot mounted at the middle of the
table's -Y edge with the base rim flush with the edge, three lights, and the calibrated RealSense
D435 provided by the embodiment. The scene carries no task; it is the starting point for tasks on
this workcell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.tasks.no_task import NoTask

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

TABLE_SIZE_M = 1.0
"""Table top edge length."""
TABLE_TOP_HEIGHT_M = 0.75
"""Height of the table's upper surface."""
TABLE_TOP_THICKNESS_M = 0.05
LEG_SECTION_M = 0.06
UR_BASE_RADIUS_M = 0.075
"""UR base radius; the base is inset by it so the base rim is flush with the table edge."""

ROBOT_BASE_XY = (0.0, -TABLE_SIZE_M / 2 + UR_BASE_RADIUS_M)
"""Robot mounting point on the table top, middle of the -Y edge."""

# Slotted aluminium-profile table top: parallel extrusions running along X with T-slot grooves between them.
SLOT_PITCH_M = 0.045
"""Centre-to-centre distance between neighbouring grooves."""
SLOT_WIDTH_M = 0.008
"""Opening width of one groove."""
SLOT_DEPTH_M = 0.008
"""Depth of the groove below the profile surface."""

PROFILE_COLOR = (0.74, 0.75, 0.77)
PROFILE_METALLIC = 0.45
PROFILE_ROUGHNESS = 0.35
"""Brushed anodised aluminium look for the profile faces."""
SLOT_FLOOR_COLOR = (0.30, 0.31, 0.33)
"""Darker groove floor so the slots read as recesses."""
LEG_COLOR = (0.25, 0.26, 0.28)

# Lights: a bright, cool-white lab ceiling. Dome does most of the work, the distant lights add soft direction.
DOME_INTENSITY = 900.0
DOME_COLOR = (0.90, 0.92, 0.97)
KEY_INTENSITY = 500.0
KEY_COLOR = (0.97, 0.98, 1.0)
FILL_INTENSITY = 250.0


@dataclass
class Ur7eWorkcellEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the UR7e workcell environment."""

    embodiment: str = "ur7e_robotiq_joint_pos"
    teleop_device: str | None = None
    light_scale: float = 1.0
    """Multiplier on all three light intensities; matches ``--light-scale`` in ``rr_ur/scene.py``."""
    episode_length_s: float = 6.0
    """Episode time-out. Short by default so headless camera-video recording flushes a clip per episode."""


def _euler_xyz_deg_to_quat_xyzw(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """Convert a USD ``rotateXYZ`` (degrees, applied X then Y then Z about fixed axes) to an xyzw quaternion."""
    hx, hy, hz = (math.radians(a) / 2.0 for a in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(hx), math.sin(hx), math.cos(hy), math.sin(hy), math.cos(hz), math.sin(hz)
    # q = qz * qy * qx
    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    return (x, y, z, w)


def _table_block(
    name: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    color: tuple[float, float, float] = PROFILE_COLOR,
    metallic: float = PROFILE_METALLIC,
    roughness: float = PROFILE_ROUGHNESS,
) -> Object:
    """Return a static, collidable cuboid with a preview-surface material, placed at ``center``."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    block = Object(
        name=name,
        prim_path="{ENV_REGEX_NS}/Table/" + name,
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic, roughness=roughness),
        ),
    )
    block.set_initial_pose(Pose(position_xyz=center))
    return block


def build_table() -> list[Object]:
    """Return the slotted table top (base slab plus aluminium profiles) and four legs as static blocks.

    The slab's upper face is the groove floor; the profiles sit on it and their upper faces form the
    table surface at ``TABLE_TOP_HEIGHT_M``.
    """
    slab_thickness = TABLE_TOP_THICKNESS_M - SLOT_DEPTH_M
    slab_top = TABLE_TOP_HEIGHT_M - SLOT_DEPTH_M
    blocks = [
        _table_block(
            "SlotFloor",
            size=(TABLE_SIZE_M, TABLE_SIZE_M, slab_thickness),
            center=(0.0, 0.0, slab_top - slab_thickness / 2),
            color=SLOT_FLOOR_COLOR,
            metallic=0.3,
            roughness=0.6,
        )
    ]
    profile_width = SLOT_PITCH_M - SLOT_WIDTH_M
    num_profiles = int(round(TABLE_SIZE_M / SLOT_PITCH_M))
    first_center_y = -(num_profiles - 1) * SLOT_PITCH_M / 2
    for i in range(num_profiles):
        blocks.append(
            _table_block(
                f"Profile{i:02d}",
                size=(TABLE_SIZE_M, profile_width, SLOT_DEPTH_M),
                center=(0.0, first_center_y + i * SLOT_PITCH_M, slab_top + SLOT_DEPTH_M / 2),
            )
        )
    leg_height = TABLE_TOP_HEIGHT_M - TABLE_TOP_THICKNESS_M
    inset = TABLE_SIZE_M / 2 - LEG_SECTION_M / 2 - 0.02
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        blocks.append(
            _table_block(
                f"Leg{i}",
                size=(LEG_SECTION_M, LEG_SECTION_M, leg_height),
                center=(sx * inset, sy * inset, leg_height / 2),
                color=LEG_COLOR,
                metallic=0.6,
                roughness=0.5,
            )
        )
    return blocks


def build_lights(asset_registry, light_scale: float = 1.0) -> list:
    """Return the dome, key and fill lights of the workcell, intensities multiplied by ``light_scale``.

    Light directions follow rr_ur/scene.py; intensities are tuned against the real D435 capture.
    """
    import isaaclab.sim as sim_utils

    assert light_scale >= 0.0, f"light_scale must be non-negative, got {light_scale}"
    k = light_scale
    dome_light = asset_registry.get_asset_by_name("light")(
        spawner_cfg=sim_utils.DomeLightCfg(intensity=DOME_INTENSITY * k, color=DOME_COLOR)
    )
    key_light = asset_registry.get_asset_by_name("directional_light")(
        instance_name="key_light",
        prim_path="/World/KeyLight",
        spawner_cfg=sim_utils.DistantLightCfg(intensity=KEY_INTENSITY * k, angle=5.0, color=KEY_COLOR),
    )
    key_light.set_orientation(_euler_xyz_deg_to_quat_xyzw(-40.0, 0.0, 30.0))
    fill_light = asset_registry.get_asset_by_name("directional_light")(
        instance_name="fill_light",
        prim_path="/World/FillLight",
        spawner_cfg=sim_utils.DistantLightCfg(intensity=FILL_INTENSITY * k, angle=5.0),
    )
    fill_light.set_orientation(_euler_xyz_deg_to_quat_xyzw(-20.0, 0.0, -120.0))
    return [dome_light, key_light, fill_light]


class IdleTask(NoTask):
    """A task with no goal that only times out, so episodes (and per-episode video clips) end."""

    name = "ur7e_idle"

    def __init__(self, episode_length_s: float):
        super().__init__()
        self.episode_length_s = episode_length_s

    def get_termination_cfg(self):
        from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

        return TaskTerminationCfg(timeout_s=self.episode_length_s)


@register_environment
class Ur7eWorkcellEnvironment(ArenaEnvironmentFactory[Ur7eWorkcellEnvironmentCfg]):
    """Static UR7e workcell: table, lights, calibrated camera, robot in ``pose_a``. No task."""

    name: str = "ur7e_workcell"
    _legacy_argparse_cfg_type = Ur7eWorkcellEnvironmentCfg

    def build(self, cfg: Ur7eWorkcellEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        k = cfg.light_scale

        table_blocks = build_table()
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()

        lights = build_lights(self.asset_registry, k)

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(ROBOT_BASE_XY[0], ROBOT_BASE_XY[1], TABLE_TOP_HEIGHT_M)))

        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device is not None else None
        )

        scene = Scene(assets=[*table_blocks, ground_plane, *lights])

        def set_viewer(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.8, -1.6, 1.7), lookat=(0.0, 0.0, TABLE_TOP_HEIGHT_M))
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=IdleTask(episode_length_s=cfg.episode_length_s),
            teleop_device=teleop_device,
            env_cfg_callback=set_viewer,
        )
