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

# Which loose part goes into which container, by asset name.
ASSIGNMENTS = (("metal_block", "green_tray"), ("bearing_assembly", "gray_tray"))
"""Raw stock into the green parts tray, finished parts into the grey sorting tray. The wrench and its
blue storage bin were dropped on 2026-09-06 (user): a 6 mm-thick wrench is too thin for a reliable
teleoperated pinch. The raw stock is the 48 mm ``metal_block`` (dex cube with the billet's look);
the flat 24 mm billet was too low to see and pinch from the operator view."""
FINISHED_PART_ALTERNATIVES = ("bearing_assembly", "small_gear_centred")
"""Assets that may stand in for the finished part. The USDCraft bearing is not graspable by the
Agibot (a 9 mm grip band over a wide flange; see the memory notes), so Arena's Factory small gear
-- 44 mm across at its library scale, SDF collider, 50 g -- is offered instead (re-centred as
``small_gear_centred``, since the library asset's origin is 50 mm off its geometry)."""

# Layout (2026-09-06, work-band review): loose parts scatter in the 0.15-0.30 work band across
# y +/-0.22; the containers flank it. Release points were IK-scanned on this date (poses out of
# 48 per lean, tilts 30/50/75): a side container at (0.22, +/-0.45, z 0.77-0.85) gives the arm on
# its side 16-18 poses and the other arm none, so each side container is served by its own arm,
# which can still fetch a part from the far half of the band (10 poses at (0.18, -/+0.20)). The
# grey tray sits beyond the band at x 0.42 where both arms have ~10 poses at lean 50-75. The old
# far-side row at x 0.52 had 0 release poses for most leans. At 70 % the containers are 224 x 154
# (grey), 230 x 160 (green) and 286 x 209 x 140 mm (bin); yaw 90 turns a long side along y, so
# the trays' inner edges stay >= 85 mm clear of the part band (the scatter's keep-out rejects any
# part footprint that reaches them).
_CONTAINER_LAYOUT = {
    # name: (x, y, yaw_deg); yaw 90 turns a container's long side along y
    "green_tray": (0.22, 0.45, 90.0),  # raw stock, robot's left
    "gray_tray": (0.22, -0.45, 90.0),  # finished parts, robot's right
}

_PART_X_BAND_M = REACH_X_BAND_M
"""Where the loose parts may land along x: the work band."""

_PART_Y_BAND_M = (-0.22, 0.22)
"""Where the loose parts may land along y."""

_PART_MIN_SEPARATION_M = 0.18
"""Smallest allowed distance between part origins: the wrench is 254 mm long at 150 %."""

_PART_TABLE_CLEARANCE_M = 0.002


def _scatter_parts(
    env,
    env_ids,
    asset_names: list[str],
    half_extents_xy: list[tuple[float, float]],
    keep_out_rects: list[tuple[float, float, float, float]],
    z_m: float,
    x_band: tuple,
    y_band: tuple,
    min_sep: float,
    margin: float = 0.02,
):
    """Reset event: drop the loose parts at random, non-overlapping spots in the work band.

    Draws of position and yaw are rejected until every pair of origins is ``min_sep`` apart and no
    part's yawed footprint reaches into a container's footprint (``keep_out_rects`` as
    ``(x_min, x_max, y_min, y_max)``, grown by ``margin``). A part spawned into a container's
    collider is ejected by PhysX, and a 254 mm wrench spawned at the band's edge with a random
    yaw did exactly that. Parts stay flat.
    """
    import torch

    import isaaclab.utils.math as math_utils

    count = len(asset_names)
    half = torch.tensor(half_extents_xy)  # (count, 2): along the part's own long and short axes
    for cur_env in env_ids.tolist():
        for _ in range(2000):
            xy = torch.stack([torch.empty(count).uniform_(*x_band), torch.empty(count).uniform_(*y_band)], dim=-1)
            yaw = torch.rand(count) * 2.0 * math.pi - math.pi
            separations = torch.cdist(xy, xy) + torch.eye(count)
            if float(separations.min()) < min_sep:
                continue
            # Axis-aligned half-widths of each yawed footprint.
            c, s_ = yaw.cos().abs(), yaw.sin().abs()
            ext_x = half[:, 0] * c + half[:, 1] * s_
            ext_y = half[:, 0] * s_ + half[:, 1] * c
            clear = True
            for x_min, x_max, y_min, y_max in keep_out_rects:
                overlap_x = (xy[:, 0] + ext_x > x_min - margin) & (xy[:, 0] - ext_x < x_max + margin)
                overlap_y = (xy[:, 1] + ext_y > y_min - margin) & (xy[:, 1] - ext_y < y_max + margin)
                if bool((overlap_x & overlap_y).any()):
                    clear = False
                    break
            if clear:
                break
        for index, name in enumerate(asset_names):
            quat = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), yaw[index : index + 1]).to(env.device)
            position = torch.tensor([[float(xy[index, 0]), float(xy[index, 1]), z_m]], device=env.device)
            root_pose = torch.cat([position + env.scene.env_origins[cur_env : cur_env + 1], quat], dim=-1).float()
            asset = env.scene[name]
            ids = torch.tensor([cur_env], device=env.device)
            asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)


def _part_scale(cfg, part_name: str) -> float:
    """The configured uniform scale of a loose part."""
    if part_name == "wrench":
        return cfg.wrench_scale
    if part_name == "metal_billet":
        return cfg.billet_scale
    if part_name == "metal_block":
        return cfg.block_scale
    return cfg.finished_part_scale if part_name == "small_gear_centred" else cfg.bearing_scale


@dataclass
class AgibotTidyWorkbenchEnvironmentCfg(AgibotTabletopEnvironmentCfg):
    """Configure the Agibot workbench-tidying environment."""

    parts: str = ",".join(part for part, _ in ASSIGNMENTS)
    """Comma-separated names of the loose parts to spawn; each brings its container. Drop entries
    to make a simpler task. (A plain string because the CLI bridge cannot parse tuple fields.)"""

    container_scale: float = 0.7
    """Uniform scale of the containers (user's choice: the stock ones dwarf the parts)."""

    billet_scale: float = 1.2
    """Scale of the metal billet (80 x 40 x 20 mm at 1.0; 96 x 48 x 24 at this default).

    User decision 2026-09-06 after the mid-air pinch gate (3 repeats, right arm, 0.5 s ramp):
    1.0 held 3/3 (close peak 0.55 m/s), 1.2 held 3/3 (0.66 m/s), 1.5 (120 x 60 x 30) was ejected
    3/3 at 3.4-4.6 m/s whichever side was pinched. 1.2 is the largest that passes."""

    block_scale: float = 1.0
    """Scale of the raw-stock cube (48 mm at 1.0). The gripper's usable object width tops out near
    55-60 mm (billet 60 mm wide: ejected 3/3), so keep this at or below ~1.15."""

    bearing_scale: float = 1.0
    """Scale of the bearing assembly (72 mm across at 1.0). 1.0 is held 3/3 (the fingers cradle
    the flange, pad gap 86 mm); 1.2 is wedged between the knuckles with the pads never touching;
    1.5 (108 mm, wider than the 104 mm jaw opening) is shot out at 6-7 m/s."""

    finished_part: str = "small_gear_centred"
    """Which asset plays the finished part: ``small_gear_centred`` (Arena's Factory gear,
    re-centred; held 3/3 in the mid-air pinch probe like the billet) or ``bearing_assembly`` (the
    USDCraft bearing, which the Agibot cannot hold in any tested form -- see the memory notes)."""

    finished_part_scale: float = 2.0
    """Scale of the small gear when it is the finished part; 2.0 is Arena's library scale
    (44 mm across, 50 mm tall). Mid-air pinch gate 2026-09-06 (right arm, 0.5 s ramp): 2.0 held
    3/3 (close peak 1.6 m/s at 1.67, 1.0 and 0.5 s ramps alike -- the SDF gear takes a harder kick
    than the convex billet but stays in the pads), 2.5 held 1/3, 3.0 (the 150 % the user asked for) was ejected 3/3 at 2-5
    m/s. 2.0 is the largest that holds."""

    wrench_scale: float = 1.2
    """Scale of the wrench: 203 x 40 x 7 mm. At 1.5 it is 254 mm long and does not lie flat inside any
    0.7-scale container (the bin's inside is ~190 x 270 mm). cuMotion lifts it 3/5 at 1.2, the
    failures being IK on the wrench's yaw, not the pinch."""


@register_environment
class AgibotTidyWorkbenchEnvironment(ArenaEnvironmentFactory[AgibotTidyWorkbenchEnvironmentCfg]):
    """Sort the loose parts on the bench into their containers with the Agibot.

    Three containers stand around the work area -- a tool tray, a raw-parts tray and a storage
    bin -- and on every reset a wrench, a metal billet and a bearing assembly are scattered on the
    table in front of the robot. The task is to put each into the container that belongs to it.
    """

    name: str = "agibot_tidy_workbench"
    _legacy_argparse_cfg_type = AgibotTidyWorkbenchEnvironmentCfg

    def build(self, cfg: AgibotTidyWorkbenchEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import torch

        import isaaclab.utils.math as math_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.tidy_workbench_task import TidyWorkbenchTask
        from isaaclab_arena.utils.pose import Pose

        background, surroundings, light = build_tabletop_stage(self, cfg)

        assert (
            cfg.finished_part in FINISHED_PART_ALTERNATIVES
        ), f"finished_part must be one of {FINISHED_PART_ALTERNATIVES}"
        assignment_table = [
            (cfg.finished_part if part == "bearing_assembly" else part, container) for part, container in ASSIGNMENTS
        ]
        wanted = [
            cfg.finished_part if name.strip() == "bearing_assembly" else name.strip()
            for name in cfg.parts.split(",")
            if name.strip()
        ]
        assignments = [(part, container) for part, container in assignment_table if part in wanted]
        assert len(assignments) == len(wanted), f"Unknown part in {wanted}; known: {[p for p, _ in ASSIGNMENTS]}"

        containers = {}
        for _, container_name in assignments:
            x, y, yaw_deg = _CONTAINER_LAYOUT[container_name]
            quat = math_utils.quat_from_euler_xyz(
                torch.zeros(1), torch.zeros(1), torch.tensor([math.radians(yaw_deg)])
            )[0]
            container = self.asset_registry.get_asset_by_name(container_name)(scale=(cfg.container_scale,) * 3)
            container.set_initial_pose(Pose(position_xyz=(x, y, TABLE_TOP_Z), rotation_xyzw=tuple(quat.tolist())))
            containers[container_name] = container

        part_z = TABLE_TOP_Z + _PART_TABLE_CLEARANCE_M
        parts = {}
        for index, (part_name, _) in enumerate(assignments):
            scale = _part_scale(cfg, part_name)
            part = self.asset_registry.get_asset_by_name(part_name)(scale=(scale,) * 3)
            # Nominal spots along the band; the scatter event re-places them on every reset.
            part.set_initial_pose(
                Pose(position_xyz=(0.22, -0.18 + 0.18 * index, part_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            )
            parts[part_name] = part

        embodiment = build_agibot(self, cfg)
        teleop_device = build_teleop_device(self, cfg)
        scene = Scene(assets=[background, *containers.values(), *parts.values(), surroundings, light])

        def env_cfg_callback(env_cfg):
            """Install the standard control stack, then scatter the parts after their own reset
            events."""
            from isaaclab.managers import EventTermCfg

            install_agibot_control_stack(env_cfg, cfg)
            # Each part's half-extents along its own axes, and each container's axis-aligned
            # footprint (they are only ever yawed by 0 or 90 deg), both at the configured scales.
            part_half = []
            for part_name in parts:
                kind = type(parts[part_name])
                if hasattr(kind, "HALF_EXTENTS_M"):
                    hx, hy = kind.HALF_EXTENTS_M[:2]
                else:
                    hx = hy = kind.RADIUS_M
                scale = _part_scale(cfg, part_name)
                part_half.append((hx * scale, hy * scale))
            keep_out = []
            for container_name, container in containers.items():
                x, y, yaw_deg = _CONTAINER_LAYOUT[container_name]
                hx, hy = (v * cfg.container_scale for v in type(container).HALF_EXTENTS_XY_M)
                if round(yaw_deg) % 180 == 90:
                    hx, hy = hy, hx
                keep_out.append((x - hx, x + hx, y - hy, y + hy))
            env_cfg.events.scatter_parts = EventTermCfg(
                func=_scatter_parts,
                mode="reset",
                params={
                    "asset_names": list(parts),
                    "half_extents_xy": part_half,
                    "keep_out_rects": keep_out,
                    "z_m": part_z,
                    "x_band": _PART_X_BAND_M,
                    "y_band": _PART_Y_BAND_M,
                    "min_sep": _PART_MIN_SEPARATION_M,
                },
            )
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=TidyWorkbenchTask(
                assignments=[(parts[p], containers[c]) for p, c in assignments],
                episode_length_s=180.0,
                viewer_cfg=embodiment.get_head_viewer_cfg() if cfg.head_view else None,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
