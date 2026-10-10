# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pine workcell with a USDCraft segmented cable, one plug pinned to the table, and a post to wrap it around."""

import math
import os
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.pine_wm_environment import PineWmEnvironment, PineWmEnvironmentCfg

CABLE_USD_ENV = "ARENA_PINE_WM_CABLE_USD"
"""Environment variable naming the flattened PhysX ``isaac/model.usdc`` of a USDCraft cable record."""
TABLE_TOP_Z = 0.74


@dataclass
class PineWmCableWrapEnvironmentCfg(PineWmEnvironmentCfg):
    cable_usd: str | None = None
    """Flattened PhysX USD of the cable; defaults to ARENA_PINE_WM_CABLE_USD."""
    anchor_xy: tuple[float, float] = (-0.10, 0.0)
    """World position of the pinned plug (the cable's articulation root)."""
    cable_yaw_deg: float = 0.0
    """Direction in which the cable leaves the pinned plug; 0 lays it along +x."""
    anchor_height_m: float = 0.0035
    """Pinned plug axis height above the table: the plug grip radius."""
    post_offset: tuple[float, float] = (0.05, 0.035)
    """Post centre relative to the pinned plug, in the cable frame (along, left of the cable)."""
    post_radius_m: float = 0.015
    post_height_m: float = 0.05
    episode_length_s: float = 120.0


def post_xy(cfg: PineWmCableWrapEnvironmentCfg) -> tuple[float, float]:
    """World xy of the post axis, from its offset in the pinned cable's frame."""
    yaw = math.radians(cfg.cable_yaw_deg)
    along, left = cfg.post_offset
    return (
        cfg.anchor_xy[0] + along * math.cos(yaw) - left * math.sin(yaw),
        cfg.anchor_xy[1] + along * math.sin(yaw) + left * math.cos(yaw),
    )


@register_environment
class PineWmCableWrapEnvironment(ArenaEnvironmentFactory[PineWmCableWrapEnvironmentCfg]):
    """UR7e Pine workcell with a cable whose first plug is fixed in place and a static post."""

    name = "pine_wm_cable_wrap_post"
    _legacy_argparse_cfg_type = PineWmCableWrapEnvironmentCfg

    def build(self, cfg: PineWmCableWrapEnvironmentCfg):
        from scipy.spatial.transform import Rotation

        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.embodiments.ur7e.ur7e import UR7E_READY_JOINT_POS
        from isaaclab_arena.utils.pose import Pose

        cable_usd = cfg.cable_usd or os.environ.get(CABLE_USD_ENV)
        assert cable_usd and os.path.isfile(cable_usd), f"Set cable_usd or {CABLE_USD_ENV} to the cable's model.usdc"
        arena = PineWmEnvironment().build(cfg)
        arena.name = self.name
        arena.embodiment.set_joint_initial_pos(UR7E_READY_JOINT_POS)
        # USDCraft cables run along their local +z from the root plug, and the joint frames give the
        # audio cable a 17 mm rest bow along local +x. Lay +z along the table and turn the bow to -y,
        # in the table plane and away from the post; a bow along -z would press it into the table.
        rotation = (
            Rotation.from_euler("z", cfg.cable_yaw_deg, degrees=True)
            * Rotation.from_euler("y", 90, degrees=True)
            * Rotation.from_euler("z", -90, degrees=True)
        )
        cable = Object(
            name="cable",
            usd_path=cable_usd,
            object_type=ObjectType.ARTICULATION,
            initial_pose=Pose(
                position_xyz=(*cfg.anchor_xy, TABLE_TOP_Z + cfg.anchor_height_m),
                rotation_xyzw=tuple(float(v) for v in rotation.as_quat()),
            ),
            # Pinning the root plug stands in for a connector plugged into a fixed jack.
            spawn_cfg_addon={"articulation_props": sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)},
        )
        post = Object(
            name="cable_post",
            prim_path="{ENV_REGEX_NS}/CablePost",
            object_type=ObjectType.BASE,
            spawner_cfg=sim_utils.CylinderCfg(
                radius=cfg.post_radius_m,
                height=cfg.post_height_m,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.1), roughness=0.5),
            ),
        )
        post.set_initial_pose(Pose(position_xyz=(*post_xy(cfg), TABLE_TOP_Z + cfg.post_height_m / 2)))
        arena.scene.add_asset(cable)
        arena.scene.add_asset(post)
        return arena
