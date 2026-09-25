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

# A bowl's origin is at its geometry centre, 30.1 mm above its base.
_BOWL_Z = TABLE_TOP_Z + 0.0301

# A triangle in front of the robot, matching how RoboDojo's demo spreads the bowls across the
# space in front of the arms rather than off to one side.
#
# Every x stays inside ``REACH_X_BAND_M`` (0.15-0.30, the main work band) after the +/-0.02
# jitter: near bowl at 0.19 (0.17-0.21), far pair at 0.27 (0.25-0.29). Until 2026-09-06 the
# triangle sat at 0.37-0.43, on the far third of the table, because of a stale reach measurement;
# the re-measured reach is better here (see ``REACH_X_BAND_M``).
#
# Spacing has to beat the measured 0.11 m bowl diameter *after* jitter, not before it. The first
# two were 0.14 m apart, which the jitter can close to 0.10 -- so some resets spawned two bowls
# already interpenetrating, which pre-loads a contact and makes them spring apart at the first
# touch. The closest pair here is 0.179 m, so the worst case is 0.139 m.
_BOWL_POSITIONS_XY = ((0.27, -0.16), (0.27, 0.16), (0.19, 0.00))

_BOWL_X_BAND_M = REACH_X_BAND_M
"""The x band the arm was measured to reach at table height; jittered bowls are clamped into it."""

_BOWL_MIN_SEPARATION_M = 0.14
"""Smallest allowed distance between jittered bowls: the 0.11 m diameter plus a contact margin."""


def _jitter_bowls(
    env, env_ids, asset_names: list[str], nominal_xy: list[tuple[float, float]], z_m: float, xy_half_m: list[float]
) -> None:
    """Reset event: re-place the bowls with a biased xy jitter, upright, a minimum distance apart.

    Per-bowl uniform offsets, with x clamped into the arm's measured reach band -- the nominal
    positions sit near its edges, so an unbiased sample wastes resets on unreachable bowls (the
    same lesson the toast rack's asymmetric range encodes). Draws are rejected until every pair
    is at least ``_BOWL_MIN_SEPARATION_M`` apart, since bowls spawned interpenetrating pre-load a
    contact and spring apart at the first touch. Orientation stays upright; only the yaw spins,
    which on a body of revolution changes nothing physical.
    """
    import torch

    import isaaclab.utils.math as math_utils

    count = len(asset_names)
    half = torch.tensor(xy_half_m).unsqueeze(1)  # per-bowl half-extent, applied to both axes
    for cur_env in env_ids.tolist():
        xy = torch.tensor(nominal_xy)
        for _ in range(200):
            candidate = torch.tensor(nominal_xy) + (torch.rand(count, 2) * 2.0 - 1.0) * half
            candidate[:, 0] = candidate[:, 0].clamp(*_BOWL_X_BAND_M)
            separations = torch.cdist(candidate, candidate) + torch.eye(count)
            if float(separations.min()) >= _BOWL_MIN_SEPARATION_M:
                xy = candidate
                break
        for name, position_xy in zip(asset_names, xy):
            yaw = torch.rand(1) * 2.0 * math.pi - math.pi
            quat = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), yaw).to(env.device)
            position = torch.tensor([[float(position_xy[0]), float(position_xy[1]), z_m]], device=env.device)
            root_pose = torch.cat([position + env.scene.env_origins[cur_env : cur_env + 1], quat], dim=-1).float()
            asset = env.scene[name]
            asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=torch.tensor([cur_env], device=env.device))
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )


@dataclass
class AgibotStackBowlsEnvironmentCfg(AgibotTabletopEnvironmentCfg):
    """Configure the Agibot bowl-stacking environment.

    RoboDojo teleoperates stack_bowls with two arms, and some starting layouts are hard to solve
    with one, so the shared ``arm_mode`` default of ``"dual"`` stands.
    """

    num_bowls: int = 3
    """How many bowls to spawn. RoboDojo's stack_bowls uses three."""

    bowl_jitter_xy_m: float = 0.02
    """Half-extent of the per-reset random xy offset applied to each bowl.

    The sample is biased, not raw uniform: x is clamped into the arm's measured reach band
    (0.35-0.45) so large jitters do not waste resets on unreachable bowls, draws are rejected
    until every pair of bowls is 0.14 m apart, and the bowls stay upright (only their yaw
    spins). Data collection uses 0.04."""

    centre_bowl_jitter_xy_m: float = 0.02
    """Half-extent for the centre bowl alone, capped tighter than the outer two.

    The centre bowl is the one the base-selection scan almost always builds the pile on -- it is
    the only position both arms can release over -- so its jitter bounds where the *pile* goes.
    Keeping it tighter keeps the releases plannable while the outer bowls roam."""


@register_environment
class AgibotStackBowlsEnvironment(ArenaEnvironmentFactory[AgibotStackBowlsEnvironmentCfg]):
    """Stack three bowls into one pile with the Agibot."""

    name: str = "agibot_stack_bowls"
    _legacy_argparse_cfg_type = AgibotStackBowlsEnvironmentCfg

    def build(self, cfg: AgibotStackBowlsEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.stack_bowls_task import StackBowlsTask
        from isaaclab_arena.utils.pose import Pose

        background, surroundings, light = build_tabletop_stage(self, cfg)

        assert cfg.num_bowls <= len(
            _BOWL_POSITIONS_XY
        ), f"Only {len(_BOWL_POSITIONS_XY)} bowl positions are laid out, asked for {cfg.num_bowls}"
        # Nominal upright poses; the biased group jitter below re-places them on every reset.
        bowls = []
        for index in range(cfg.num_bowls):
            x, y = _BOWL_POSITIONS_XY[index]
            bowl = self.asset_registry.get_asset_by_name("bowl")(instance_name=f"bowl{index}")
            bowl.set_initial_pose(Pose(position_xyz=(x, y, _BOWL_Z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
            bowls.append(bowl)

        embodiment = build_agibot(self, cfg)
        teleop_device = build_teleop_device(self, cfg)
        scene = Scene(assets=[background, *bowls, surroundings, light])

        def env_cfg_callback(env_cfg):
            """Install the standard control stack, then attach the bowls' biased group jitter
            after their own reset events."""
            install_agibot_control_stack(env_cfg, cfg)
            if cfg.bowl_jitter_xy_m:
                from isaaclab.managers import EventTermCfg

                # The centre bowl -- index 2 in the layout, the (0.37, 0.00) one -- gets its own
                # tighter half-extent; see centre_bowl_jitter_xy_m.
                per_bowl = [
                    min(cfg.bowl_jitter_xy_m, cfg.centre_bowl_jitter_xy_m) if index == 2 else cfg.bowl_jitter_xy_m
                    for index in range(cfg.num_bowls)
                ]
                env_cfg.events.jitter_bowls = EventTermCfg(
                    func=_jitter_bowls,
                    mode="reset",
                    params={
                        "asset_names": [bowl.name for bowl in bowls],
                        "nominal_xy": [list(_BOWL_POSITIONS_XY[index]) for index in range(cfg.num_bowls)],
                        "z_m": _BOWL_Z,
                        "xy_half_m": per_bowl,
                    },
                )
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=StackBowlsTask(
                bowls=bowls,
                episode_length_s=120.0,
                viewer_cfg=embodiment.get_head_viewer_cfg() if cfg.head_view else None,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
