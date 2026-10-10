# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pine workcell with a USDCraft surface-deformable towel lying flat on the table, for folding."""

import os
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.pine_wm_environment import PineWmEnvironment, PineWmEnvironmentCfg

TOWEL_USD_ENV = "ARENA_PINE_WM_TOWEL_USD"
"""Environment variable naming the flattened PhysX ``isaac/model.usdc`` of a USDCraft towel record."""


@dataclass
class PineWmFoldTowelEnvironmentCfg(PineWmEnvironmentCfg):
    towel_usd: str | None = None
    """Flattened PhysX USD of the towel; defaults to ARENA_PINE_WM_TOWEL_USD."""
    towel_position: tuple[float, float, float] = (0.0, 0.05, 0.745)
    """Towel origin in the world; the table's upper surface is at 0.74 m."""
    towel_yaw_deg: float = 0.0
    episode_length_s: float = 120.0


@register_environment
class PineWmFoldTowelEnvironment(ArenaEnvironmentFactory[PineWmFoldTowelEnvironmentCfg]):
    """UR7e Pine workcell with one free PhysX surface-deformable towel and no attachments."""

    name = "pine_wm_fold_towel"
    _legacy_argparse_cfg_type = PineWmFoldTowelEnvironmentCfg

    def build(self, cfg: PineWmFoldTowelEnvironmentCfg):
        from scipy.spatial.transform import Rotation

        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.deformable_object import DeformableObject
        from isaaclab_arena.embodiments.ur7e.ur7e import UR7E_READY_JOINT_POS
        from isaaclab_arena.utils.physics_backend import PhysicsBackend
        from isaaclab_arena.utils.pose import Pose

        towel_usd = cfg.towel_usd or os.environ.get(TOWEL_USD_ENV)
        assert towel_usd and os.path.isfile(towel_usd), f"Set towel_usd or {TOWEL_USD_ENV} to the towel's model.usdc"
        arena = PineWmEnvironment().build(cfg)
        arena.name = self.name
        arena.embodiment.set_joint_initial_pos(UR7E_READY_JOINT_POS)
        rotation = tuple(float(v) for v in Rotation.from_euler("z", cfg.towel_yaw_deg, degrees=True).as_quat())
        # The USD carries its own surface-deformable body, material and collision schemas; spawn it unchanged.
        towel = DeformableObject(
            name="towel",
            spawner_cfg=sim_utils.UsdFileCfg(usd_path=towel_usd),
            initial_pose=Pose(position_xyz=cfg.towel_position, rotation_xyzw=rotation),
            physics_backend=PhysicsBackend.PHYSX,
        )
        arena.scene.add_asset(towel)
        return arena
