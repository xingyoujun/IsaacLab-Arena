# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Calibrated pine_wm workcell from isim_scene_20260925, ready for task composition."""

from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


@dataclass
class PineWmEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the pine_wm workcell without changing the original RR environments."""

    asset_root: str | None = None
    """USDCraft-Scene bundle root; defaults to ARENA_USDCRAFT_SCENE_ROOT. Explicit legacy roots remain supported."""
    light_scale: float = 1.0
    episode_length_s: float = 6.0


def build_pine_wm_table() -> list:
    """Build a slotted aluminium table at the calibrated height, with the robot mounting plate."""
    from isaaclab_arena_environments.ur7e_workcell_environment import _table_block

    # Recessed slot floor and 25 parallel profiles; upper surface remains exactly 0.74 m.
    blocks = [_table_block("SlotFloor", (1.0, 1.0, 0.043), (0.0, 0.0, 0.7115), (0.24, 0.25, 0.26), 0.65, 0.45)]
    for index in range(25):
        center_y = -0.48 + index * 0.04
        blocks.append(
            _table_block(
                f"Profile{index:02d}", (1.0, 0.0355, 0.007), (0.0, center_y, 0.7365), (0.66, 0.68, 0.70), 0.75, 0.32
            )
        )
    parts = []
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        parts.append((f"Leg{i}", (0.06, 0.06, 0.69), (sx * 0.45, sy * 0.45, 0.345)))
    for name, size, center in parts:
        blocks.append(_table_block(name, size, center, color=(0.52, 0.54, 0.56), metallic=0.7, roughness=0.38))
    blocks.append(_table_block("Mount", (0.20, 0.20, 0.01), (0.0, -0.425, 0.745), (0.02, 0.02, 0.02), 0.0, 0.6))
    return blocks


def build_pine_wm_lights(asset_registry, light_scale: float) -> list:
    """Build broad neutral laboratory lighting with soft, slightly warm highlights."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena_environments.ur7e_workcell_environment import _euler_xyz_deg_to_quat_xyzw

    assert light_scale >= 0.0
    dome = asset_registry.get_asset_by_name("light")(
        spawner_cfg=sim_utils.DomeLightCfg(intensity=650.0 * light_scale, color=(0.98, 0.97, 0.94))
    )
    lights = [dome]
    for name, intensity, angle, color, euler in (
        ("key", 450.0, 18.0, (1.0, 0.98, 0.94), (-40.0, 0.0, 30.0)),
        ("fill", 200.0, 25.0, (1.0, 1.0, 1.0), (-20.0, 0.0, -120.0)),
    ):
        light = asset_registry.get_asset_by_name("directional_light")(
            instance_name=f"pine_wm_{name}",
            prim_path=f"/World/PineWmLights/{name}",
            spawner_cfg=sim_utils.DistantLightCfg(intensity=intensity * light_scale, angle=angle, color=color),
        )
        light.set_orientation(_euler_xyz_deg_to_quat_xyzw(*euler))
        lights.append(light)
    return lights


@register_environment
class PineWmEnvironment(ArenaEnvironmentFactory[PineWmEnvironmentCfg]):
    """UR7e, slotted metal table, mounting hardware, D435 and two wrist D405 cameras."""

    name = "pine_wm"
    _legacy_argparse_cfg_type = PineWmEnvironmentCfg

    def build(self, cfg: PineWmEnvironmentCfg):
        """Compose the calibrated scene with an idle task and continuous joint control."""
        from isaaclab_arena.embodiments.ur7e.pine_wm import ROBOT_POSITION, PineWmUr7eEmbodiment
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.ur7e_workcell_environment import IdleTask

        assert cfg.episode_length_s > 0
        embodiment = PineWmUr7eEmbodiment(enable_cameras=cfg.enable_cameras, asset_root=cfg.asset_root)
        embodiment.set_initial_pose(Pose(position_xyz=ROBOT_POSITION))
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[*build_pine_wm_table(), *build_pine_wm_lights(self.asset_registry, cfg.light_scale)]),
            task=IdleTask(episode_length_s=cfg.episode_length_s),
        )
