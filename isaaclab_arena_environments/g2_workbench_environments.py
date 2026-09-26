# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G2 workbench scenes for inspecting and teleoperating Arena's existing assets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.assets.usdcraft_scene import resolve_asset
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class G2WorkbenchEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the common G2 workbench and dual-arm keyboard interface."""

    table_height_m: float = 0.75
    """Floor-to-tabletop height; the tabletop remains at world z=0."""
    robot_position: list[float] | None = None
    """Optional base x/y/z override."""
    arm_mode: str = "dual_arm"
    teleop_device: str | None = None
    hdr: str | None = None
    light_intensity: float = 1000.0

    def __post_init__(self):
        assert self.table_height_m > 0, "table_height_m must be positive"
        assert self.robot_position is None or len(self.robot_position) == 3, "robot_position must be x/y/z"
        assert self.arm_mode in ("dual_arm", "left", "right"), "Unsupported arm_mode"
        assert self.light_intensity >= 0, "light_intensity must be non-negative"


class _G2WorkbenchEnvironment:
    """Build a scene without task success terms or automatic task execution."""

    object_layout: tuple[tuple[str, str, float, float, float], ...] = ()
    """Registry name, instance name, bounding-box centre x/y, and yaw in degrees."""
    scale_overrides: dict[str, tuple[float, float, float]] = {}
    """Explicit per-instance scales, where a library asset needs a larger work surface."""

    def make_object(self, registry_name, params):
        """Construct an object, allowing task-local asset overrides."""
        cls = self.asset_registry.get_asset_by_name(registry_name)
        if registry_name in {"peg", "hole", "small_gear", "cordless_drill_ycb_robolab", "bin_b04_vomp_robolab"}:
            path = str(resolve_asset(f"g2_{registry_name}"))
            cls = type(f"Bundled{cls.__name__}", (cls,), {"usd_path": path})
        return cls(**params)

    def build(self, cfg: G2WorkbenchEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Place the selected library objects on the shared tabletop."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.background import Background
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        table = self.asset_registry.get_asset_by_name("genie_benchmark_table")(height_m=cfg.table_height_m)
        room = Background(
            name="g2_workroom",
            usd_path=str(resolve_asset("g2_room")),
            object_min_z=-cfg.table_height_m,
            initial_pose=Pose(position_xyz=(0.0, 0.0, -cfg.table_height_m), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )
        ground = self.asset_registry.get_asset_by_name("ground_plane")(
            initial_pose=Pose(position_xyz=(0.0, 0.0, -cfg.table_height_m - 0.08), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())

        objects = []
        for registry_name, instance_name, x, y, yaw_deg in self.object_layout:
            params = {"instance_name": instance_name}
            if instance_name in self.scale_overrides:
                params["scale"] = self.scale_overrides[instance_name]
            obj = self.make_object(registry_name, params)
            half_yaw = math.radians(yaw_deg) / 2
            rotation = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))
            bounds = obj.get_bounding_box().enclosing_after_rotation(rotation)
            centre = bounds.center[0]
            # Asset origins vary; place the actual geometry just above the support surface.
            obj.set_initial_pose(
                Pose(
                    position_xyz=(x - centre[0].item(), y - centre[1].item(), 0.003 - bounds.min_point[0, 2].item()),
                    rotation_xyzw=rotation,
                )
            )
            objects.append(obj)

        embodiment = self.asset_registry.get_asset_by_name("g2")(
            enable_cameras=cfg.enable_cameras, arm_mode=ArmMode(cfg.arm_mode)
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=tuple(cfg.robot_position or [-0.65, 0.0, -cfg.table_height_m]),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        device = self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device else None

        def configure(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.25, -1.7, 1.1), lookat=(-0.3, 0.0, 0.2))
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[table, room, ground, light, *objects]),
            teleop_device=device,
            env_cfg_callback=configure,
        )


@dataclass
class G2PegInsertEnvironmentCfg(G2WorkbenchEnvironmentCfg):
    """Configure the provisional peg/hole scene."""


@register_environment
class G2PegInsertEnvironment(_G2WorkbenchEnvironment, ArenaEnvironmentFactory[G2PegInsertEnvironmentCfg]):
    """Preview Arena's peg/hole assembly assets as a provisional sleeve-task scene."""

    name = "g2_peg_insert"
    _legacy_argparse_cfg_type = G2PegInsertEnvironmentCfg
    object_layout = (
        ("peg", "peg", -0.10, -0.16, 0.0),
        ("hole", "hole", -0.10, 0.16, 0.0),
    )


@dataclass
class G2TurnKnobEnvironmentCfg(G2WorkbenchEnvironmentCfg):
    """Configure the stand-mixer knob scene."""


@register_environment
class G2TurnKnobEnvironment(_G2WorkbenchEnvironment, ArenaEnvironmentFactory[G2TurnKnobEnvironmentCfg]):
    """Preview Arena's articulated stand mixer and its speed-control knob."""

    name = "g2_turn_knob"
    _legacy_argparse_cfg_type = G2TurnKnobEnvironmentCfg
    object_layout = (("stand_mixer", "stand_mixer", -0.02, 0.0, 180.0),)
