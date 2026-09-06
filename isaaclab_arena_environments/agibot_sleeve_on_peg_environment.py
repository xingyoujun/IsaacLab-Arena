# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.agibot_tabletop_common import (
    REACH_X_BAND_M,
    TABLE_TOP_Z,
    AgibotTabletopEnvironmentCfg,
    build_agibot,
    build_tabletop_stage,
    build_teleop_device,
    install_agibot_control_stack,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# The platform stands with its peg at x 0.225, the middle of the 0.15-0.30 work band
# (``REACH_X_BAND_M``, re-measured 2026-09-06). Its long side runs along x, so the deck spans
# 0.105-0.345 by +/-0.09 in y; the deck's far end pokes past the band, which is fine -- only the
# peg is a grasp/release point. Until 2026-09-06 the peg sat at 0.40, on the far third of the
# table, because of a stale reach measurement.
_PLATFORM_POSITION_XY = (0.225, 0.0)

_SLEEVE_X_BAND_M = REACH_X_BAND_M
"""Where the sleeve may land along x: the work band."""

_SLEEVE_ABS_Y_BAND_M = (0.135, 0.21)
"""Where the sleeve may land in |y|: outside the platform's 90 mm half-depth plus the sleeve's
own radius and a contact margin, and inside the y the bowls of stack_bowls are reached at."""

_SLEEVE_TABLE_CLEARANCE_M = 0.001
"""The sleeve is dropped from 1 mm up rather than written flush, so it never spawns pressed
into the table."""


def _place_sleeve(env, env_ids, asset_name: str, side: str, z_m: float, abs_y_min_m: float) -> None:
    """Reset event: stand the sleeve at a random spot on the table, off the platform.

    x is drawn uniformly across the reach band and |y| across the band flanking the platform, on
    the robot's left, right, or either at random. The sleeve stays upright with a random yaw,
    which on a body of revolution changes nothing physical.
    """
    import torch

    import isaaclab.utils.math as math_utils

    count = len(env_ids)
    x = torch.empty(count).uniform_(*_SLEEVE_X_BAND_M)
    y = torch.empty(count).uniform_(abs_y_min_m, _SLEEVE_ABS_Y_BAND_M[1])
    if side == "left":
        pass  # +y is the robot's left
    elif side == "right":
        y = -y
    else:
        y = y * (torch.randint(0, 2, (count,)) * 2 - 1)
    yaw = torch.rand(count) * 2.0 * math.pi - math.pi
    quat = math_utils.quat_from_euler_xyz(torch.zeros(count), torch.zeros(count), yaw).to(env.device)
    position = torch.stack([x, y, torch.full((count,), z_m)], dim=-1).to(env.device)
    root_pose = torch.cat([position + env.scene.env_origins[env_ids], quat], dim=-1).float()
    asset = env.scene[asset_name]
    asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(count, 6, device=env.device), env_ids=env_ids)


def _apply_factory_part_physics(part, kinematic: bool) -> None:
    """Give an assembly part the rigid-body and collision settings of Arena's Factory peg and hole.

    Args:
        part: The scene object whose spawn config is patched in place.
        kinematic: Keep the part kinematic (the fixture) or let it move (the held part).
    """
    from dataclasses import replace

    from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg

    from isaaclab_arena.assets.object_utils import RIGID_BODY_PROPS_HIGH_PRECISION

    part.object_cfg.spawn.rigid_props = replace(RIGID_BODY_PROPS_HIGH_PRECISION, kinematic_enabled=kinematic)
    part.object_cfg.spawn.collision_props = CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)


def _apply_factory_scene_physics(env_cfg) -> None:
    """Apply the scene-level physics of ``assembly_env_cfg_callback`` while keeping 15 Hz control.

    The assembly callback rebuilds ``env_cfg.sim`` wholesale and moves control to 30 Hz; this
    patches the same physics fields in place and keeps the step count per control step so the
    control rate stays at Arena's 15 Hz default.

    Args:
        env_cfg: The compiled environment configuration, patched in place.
    """
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

    env_cfg.sim.dt = 1 / 60
    env_cfg.sim.physics = PhysxCfg(
        solver_type=1,
        max_position_iteration_count=192,
        max_velocity_iteration_count=1,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.01,
        friction_correlation_distance=0.00625,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
        gpu_max_num_partitions=1,
    )
    env_cfg.sim.physics_material = RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
    env_cfg.decimation = 4
    print(f"[factory physics] sim dt {env_cfg.sim.dt:.4f}, decimation {env_cfg.decimation}, 192 position iterations")


@dataclass
class AgibotSleeveOnPegEnvironmentCfg(AgibotTabletopEnvironmentCfg):
    """Configure the Agibot sleeve-on-peg assembly environment.

    The sleeve lands on either side of the platform by default, so both arms are driven (the
    shared ``arm_mode`` default of ``"dual"``).
    """

    sleeve_side: str = "both"
    """Which side of the platform the sleeve starts on: ``"left"``, ``"right"`` or ``"both"``
    (chosen at random on every reset). Restrict it to make a single-arm dataset."""

    platform_scale: float = 1.0
    """Uniform scale applied to the platform and its peg; the sleeve is not scaled, and the
    seated height scales with the platform.

    Left at 1.0 -- scaling the peg down does not make the sleeve fit. The sleeve shell's
    convexDecomposition collider (16 hulls) bulges into the bore to r 10.6 mm against the
    mesh's 12.5 mm, so at 0.9 (peg r 10.8 mm) a seated sleeve is still interfering and gets
    flung off; the fix has to be on the sleeve's collider, not the peg's size."""

    sleeve_abs_y_min_m: float = _SLEEVE_ABS_Y_BAND_M[0]
    """Inner edge of the |y| band the sleeve lands in. 0.135 just clears the platform; a
    horizontal (side) grasp with the 104 mm open jaws needs the sleeve at least 0.17 from the
    centreline before cuMotion can plan the descent beside the platform (plan probe, +15 cm:
    0/14 descents at |y| 0.136, 8/14 at 0.17, 12/12 at 0.20)."""

    table_raise_m: float = 0.0
    """Extra height of the work surface over the stack_bowls table (0.6232), platform and sleeve
    following. Kept as a knob because an IK scan showed the Agibot can only approach the table
    at >= 30 deg of lean, and horizontal (side) grasps -- the only ones that keep a cylinder upright
    in flat jaws -- become reachable as the surface rises (0 poses at +0, 120 at +10 cm, 148 at
    +15 cm over nine sleeve spots; over the seat 0 / 12 / 16)."""

    factory_physics: bool = False
    """Run the two assembly parts and the physics scene the way Arena's ``peg_insert`` does.

    Copied from the Factory peg/hole assets and ``assembly_env_cfg_callback``, nothing else: both
    parts get ``RIGID_BODY_PROPS_HIGH_PRECISION`` (192 position iterations) and a 5 mm contact
    offset with zero rest offset, the scene runs at a 1/60 s step with the TGS solver at 192
    position iterations and a friction-1.0 default material. Control stays at 15 Hz (decimation
    4 instead of 8). Off by default while it is being compared against the stock physics."""

    platform_fixed: bool = True
    """Hold the platform kinematic. False lets its 1.8 kg body sit on the table under gravity
    and friction, where a jammed insert can shove it."""


@register_environment
class AgibotSleeveOnPegEnvironment(ArenaEnvironmentFactory[AgibotSleeveOnPegEnvironmentCfg]):
    """Put the loose sleeve back onto the platform's peg with the Agibot.

    The platform stands fixed in front of the robot; on every reset the sleeve is set down at a
    random spot on the table beside it, and the task is to slide it back over the peg.
    """

    name: str = "agibot_sleeve_on_peg"
    _legacy_argparse_cfg_type = AgibotSleeveOnPegEnvironmentCfg

    def build(self, cfg: AgibotSleeveOnPegEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.sleeve_on_peg_task import SleeveOnPegTask
        from isaaclab_arena.utils.pose import Pose

        table_top_z = TABLE_TOP_Z + cfg.table_raise_m
        background, surroundings, light = build_tabletop_stage(self, cfg, table_top_z)

        platform_asset = self.asset_registry.get_asset_by_name("peg_platform")
        platform = platform_asset(scale=(cfg.platform_scale,) * 3)
        platform.set_initial_pose(
            Pose(position_xyz=(*_PLATFORM_POSITION_XY, table_top_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        if not cfg.platform_fixed:
            platform.object_cfg.spawn.rigid_props = RigidBodyPropertiesCfg(kinematic_enabled=False)
        if cfg.factory_physics:
            _apply_factory_part_physics(platform, kinematic=cfg.platform_fixed)

        # Nominal pose on the robot's left; the reset event below re-places it every episode.
        sleeve_z = table_top_z + _SLEEVE_TABLE_CLEARANCE_M
        sleeve = self.asset_registry.get_asset_by_name("peg_sleeve")()
        sleeve.set_initial_pose(
            Pose(
                position_xyz=(_PLATFORM_POSITION_XY[0], _SLEEVE_ABS_Y_BAND_M[0] + 0.03, sleeve_z),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        if cfg.factory_physics:
            _apply_factory_part_physics(sleeve, kinematic=False)

        embodiment = build_agibot(self, cfg)
        teleop_device = build_teleop_device(self, cfg)
        scene = Scene(assets=[background, platform, sleeve, surroundings, light])

        assert cfg.sleeve_side in ("left", "right", "both"), f"Unknown sleeve_side {cfg.sleeve_side!r}"
        assert (
            _SLEEVE_ABS_Y_BAND_M[0] <= cfg.sleeve_abs_y_min_m < _SLEEVE_ABS_Y_BAND_M[1]
        ), f"sleeve_abs_y_min_m {cfg.sleeve_abs_y_min_m} must lie in {_SLEEVE_ABS_Y_BAND_M}"

        def env_cfg_callback(env_cfg):
            """Install the standard control stack (guarding the possibly raised table), then
            attach the sleeve's random placement after its own reset event."""
            from isaaclab.managers import EventTermCfg

            install_agibot_control_stack(env_cfg, cfg, surface_z=table_top_z)
            if cfg.factory_physics:
                _apply_factory_scene_physics(env_cfg)
            env_cfg.events.place_sleeve = EventTermCfg(
                func=_place_sleeve,
                mode="reset",
                params={
                    "asset_name": sleeve.name,
                    "side": cfg.sleeve_side,
                    "z_m": sleeve_z,
                    "abs_y_min_m": cfg.sleeve_abs_y_min_m,
                },
            )
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=SleeveOnPegTask(
                sleeve=sleeve,
                platform=platform,
                seated_offset_m=tuple(v * cfg.platform_scale for v in platform_asset.SLEEVE_SEATED_OFFSET_M),
                episode_length_s=120.0,
                viewer_cfg=embodiment.get_head_viewer_cfg() if cfg.head_view else None,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
