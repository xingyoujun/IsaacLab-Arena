# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn the stand mixer's speed knob to a target level with the Agibot.

The Agibot port of ``gr1_turn_stand_mixer_knob``: Arena's Lightwheel stand mixer stands on the
RoboDojo table with its knob face toward the robot, inside the 0.15-0.30 work band, and the task
is Arena's ``TurnKnobTask`` (seven 40-degree levels between 40 and 280 degrees of knob angle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.agibot_tabletop_common import (
    TABLE_TOP_Z,
    AgibotTabletopEnvironmentCfg,
    build_agibot,
    build_tabletop_stage,
    build_teleop_device,
    install_agibot_control_stack,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

MIXER_ORIGIN_ABOVE_BASE_M = 0.163
"""The StandMixer013 body's origin sits 163 mm above its base (its bbox spans z -0.163..+0.035),
so the origin goes at ``surface_z + 0.163`` for the mixer to stand on the surface."""

MIXER_KNOB_OFFSET_M = (0.068, 0.084, -0.039)
"""The speed knob's axis point in the mixer frame (the revolute joint's anchor). The knob is a
38 mm disc, 20 mm proud of the +x face, turning about the mixer's x axis."""

KNOB_FACE_TOWARD_ROBOT_ROTATION_XYZW = (0.0, 0.0, 1.0, 0.0)
"""Yaw 180 deg: the mixer's +x face (the knob face) turns to world -x, toward the robot."""

_MIXER_POSITION_XY = (0.29, 0.0)
"""Mixer origin. With the yaw above the knob lands at world x 0.222, y -0.084 (the robot's right,
so the right arm serves it), 124 mm above the table -- inside the work band. The body spans
x 0.22-0.38, y +/-0.16 on the table."""


def knob_world_position(table_top_z: float = TABLE_TOP_Z) -> tuple[float, float, float]:
    """World position of the knob's axis point for the default layout (yaw 180 deg)."""
    dx, dy, dz = MIXER_KNOB_OFFSET_M
    return (_MIXER_POSITION_XY[0] - dx, _MIXER_POSITION_XY[1] - dy, table_top_z + MIXER_ORIGIN_ABOVE_BASE_M + dz)


@dataclass
class AgibotTurnMixerKnobEnvironmentCfg(AgibotTabletopEnvironmentCfg):
    """Configure the Agibot stand-mixer-knob environment."""

    arm_mode: str = "dual"

    target_level: int = 4
    """Knob level the task asks for (0-6; level k is 40 + 40 k degrees of knob angle)."""

    reset_level: int = -1
    """Knob level set at every reset; -1 is the dead zone below level 0 (knob at 0 degrees)."""


@register_environment
class AgibotTurnMixerKnobEnvironment(ArenaEnvironmentFactory[AgibotTurnMixerKnobEnvironmentCfg]):
    """Turn the stand mixer's speed knob to the requested level with the Agibot."""

    name: str = "agibot_turn_mixer_knob"
    _legacy_argparse_cfg_type = AgibotTurnMixerKnobEnvironmentCfg

    def build(self, cfg: AgibotTurnMixerKnobEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.turn_knob_task import TurnKnobTask
        from isaaclab_arena.utils.pose import Pose

        background, surroundings, light = build_tabletop_stage(self, cfg, TABLE_TOP_Z)
        stand_mixer = self.asset_registry.get_asset_by_name("stand_mixer")()
        stand_mixer.set_initial_pose(
            Pose(
                position_xyz=(*_MIXER_POSITION_XY, TABLE_TOP_Z + MIXER_ORIGIN_ABOVE_BASE_M),
                rotation_xyzw=KNOB_FACE_TOWARD_ROBOT_ROTATION_XYZW,
            )
        )
        embodiment = build_agibot(self, cfg)
        teleop_device = build_teleop_device(self, cfg)
        scene = Scene(assets=[background, stand_mixer, surroundings, light])

        def env_cfg_callback(env_cfg):
            install_agibot_control_stack(env_cfg, cfg)
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=TurnKnobTask(
                turnable_object=stand_mixer,
                target_level=cfg.target_level,
                reset_level=cfg.reset_level,
                episode_length_s=120.0,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
