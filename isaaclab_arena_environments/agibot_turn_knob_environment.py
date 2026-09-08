# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn the detent knob on a fixed dial panel to a target level with the Agibot.

The USDCraft industrial rotary knob (a 75 mm ridged knob on a 130 mm dial panel, held by a
spring-loaded detent; used at scale 0.8) is pinned in mid-air above the RoboDojo table, in the near
part of the work band on the serving arm's side, its face pitched back 60 degrees to look up at the head, and the task is Arena's ``TurnKnobTask`` over the ``RotaryKnob``
asset's ten 27-degree levels. It replaces the stand-mixer knob attempt, whose knob was a
zero-friction free wheel with a 2.65 mm disc collider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.local_objects import RotaryKnob
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

PANEL_PITCH_TOWARD_HEAD_DEG = 60.0
"""The panel is tilted back this far from vertical so its face looks up toward the robot's head
(user layout decision 2026-09-07): a vertical panel low in the head camera's view rendered small
and edge-on. At 60 degrees the face normal is (-0.5, 0, 0.87), 9 degrees off the head camera's
gaze direction (0.63, 0, -0.78). 45 degrees was tried first and looked right in the head view, but
at this spot the arm has no IK solution for a tool pointing 15-45 degrees below horizontal (horizontal
and 60-75 degrees down are fine; 41 IK seeds, both arms' geometry), so the knob axis is put in the
reachable band instead of skewing the grasp off the axis."""

PANEL_POSITION_BY_SIDE = {
    "right": (0.16, -0.24, 0.82),
    "left": (0.16, 0.24, 0.82),
}
"""Knob axis point on the panel's mid-plane, per serving arm. In the near part of the work band
(the knob face ends up at world x ~0.14), off-centre so one arm serves it alone (the two arms'
workspaces do not overlap over the table), 197 mm above the table and below the resting hands
(z 0.93+). Pinned in mid-air: the panel is fixed to the world, nothing supports it."""


def knob_axis_world(side: str = "right") -> tuple[float, float, float]:
    """Unit vector along the knob's axis, pointing out of the face toward the robot's head (world)."""
    pitch = math.radians(PANEL_PITCH_TOWARD_HEAD_DEG)
    return (-math.cos(pitch), 0.0, math.sin(pitch))


def panel_rotation_xyzw(side: str = "right") -> tuple[float, float, float, float]:
    """Panel orientation: the asset's -y (knob face) turned toward the robot, then pitched back.

    Yaw -90 deg about z puts the face on -x; the pitch about world y then lifts the face normal to
    ``knob_axis_world``. Same for both sides.
    """
    yaw, pitch = math.radians(-90.0), math.radians(PANEL_PITCH_TOWARD_HEAD_DEG)
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    m = [[sum(ry[i][k] * rz[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    w = math.sqrt(max(0.0, 1.0 + m[0][0] + m[1][1] + m[2][2])) / 2.0
    x = (m[2][1] - m[1][2]) / (4.0 * w)
    y = (m[0][2] - m[2][0]) / (4.0 * w)
    z = (m[1][0] - m[0][1]) / (4.0 * w)
    return (x, y, z, w)


KNOB_SCALE = 0.8
"""Uniform scale of the knob asset (colliders are convex decompositions, so scaling is safe).

Two constraints, both measured 2026-09-07 with the scripted diameter pinch: at 1.0 the knob is
75 mm across, wider than the ~60 mm the pads can hold (closed on it, lost it on the first 15-degree
roll); at 0.7 it is only 21 mm deep, shorter than the 40 mm pad plate, which then either hangs off the
front or hits the panel behind it. 0.8 gives a 60 x 24 mm knob on a 104 mm panel."""


KNOB_GRASP_CLOSE_TARGET = 0.49
"""Gripper target for holding the knob: 57 mm free pad gap on the 60 mm knob (executor path, measured
0.994 -> 104.6, 0.6 -> 68.7, 0.49 -> 57.1, 0.3 -> 36.1, 0.0 -> 2.2 mm). A full close is what breaks
this grasp: the pads advance 18 mm along the fingers as they close and their 40 mm plates then reach
the panel 24 mm behind the knob face, which is pinned to the world and throws the arm off."""

KNOB_GRASP_PLATE_DEPTH_M = -0.006
"""Where the pad plates' centres sit along the knob axis at grasp: 6 mm in front of the face, so the
plates' rear ends stay 10 mm clear of the panel while 14 mm of plate covers the knob."""

KNOB_GRASP_PLUNGE_M = 0.011
"""Pad advance along the approach between the open hand and the partial close above (measured)."""

TOOL_PAST_OPEN_PLATES_M = 0.035
"""The tool frame (gripper_center) is 37.7 mm past the pad link origins with the hand open, and the
plate centre is 2.5 mm past the link origin: aim the tool this far beyond where the open plates go.
Result (2026-09-07, scripted right-arm diameter pinch, jaws horizontal, 15-degree rolls about the knob
axis): 0 -> 181 deg in twelve steps with no slip, gap constant at 60.3 mm, knob held after release."""


def knob_face_world_position(
    side: str = "right", panel_xyz: tuple[float, float, float] | None = None, scale: float = KNOB_SCALE
) -> tuple[float, float, float]:
    """World position of the centre of the knob's front face for the layout of ``side``."""
    px, py, pz = PANEL_POSITION_BY_SIDE[side] if panel_xyz is None else panel_xyz
    nx, ny, nz = knob_axis_world(side)
    depth = -scale * RotaryKnob.KNOB_FACE_Y_M  # 26 mm at scale 0.8
    return (px + depth * nx, py + depth * ny, pz + depth * nz)


@dataclass
class AgibotTurnKnobEnvironmentCfg(AgibotTabletopEnvironmentCfg):
    """Configure the Agibot rotary-knob environment."""

    arm_mode: str = "dual"

    side: str = "right"
    """Which arm the knob is laid out for: ``"right"`` or ``"left"``. Picks the panel position (the
    45-degree pitch toward the head is the same on both sides); both arms are still driven (``arm_mode`` dual)."""

    panel_x: float | None = None
    panel_y: float | None = None
    panel_z: float | None = None
    """Knob axis point; None takes ``PANEL_POSITION_BY_SIDE[side]``."""

    knob_scale: float = KNOB_SCALE
    """Uniform scale of the knob asset (see ``KNOB_SCALE``)."""

    target_level: int = 5
    """Knob level the task asks for (0-9; level k is 27 + 27 k .. 54 + 27 k degrees, i.e. the dial's
    10 k .. 10 (k + 1) reading)."""

    reset_level: int = -1
    """Knob level set at every reset; -1 is the dead zone below level 0 (knob at 0 degrees)."""


@register_environment
class AgibotTurnKnobEnvironment(ArenaEnvironmentFactory[AgibotTurnKnobEnvironmentCfg]):
    """Turn the dial-panel knob to the requested level with the Agibot."""

    name: str = "agibot_turn_knob"
    _legacy_argparse_cfg_type = AgibotTurnKnobEnvironmentCfg

    def build(self, cfg: AgibotTurnKnobEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.turn_knob_task import TurnKnobTask
        from isaaclab_arena.utils.pose import Pose

        background, surroundings, light = build_tabletop_stage(self, cfg, TABLE_TOP_Z)
        knob = self.asset_registry.get_asset_by_name("rotary_knob")(scale=(cfg.knob_scale,) * 3)
        assert cfg.side in PANEL_POSITION_BY_SIDE, f"side must be 'left' or 'right', got {cfg.side!r}"
        default_xyz = PANEL_POSITION_BY_SIDE[cfg.side]
        panel_xyz = tuple(d if v is None else v for v, d in zip((cfg.panel_x, cfg.panel_y, cfg.panel_z), default_xyz))
        knob.set_initial_pose(Pose(position_xyz=panel_xyz, rotation_xyzw=panel_rotation_xyzw(cfg.side)))
        embodiment = build_agibot(self, cfg)
        teleop_device = build_teleop_device(self, cfg)
        scene = Scene(assets=[background, knob, surroundings, light])

        def env_cfg_callback(env_cfg):
            install_agibot_control_stack(env_cfg, cfg)
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=TurnKnobTask(
                turnable_object=knob,
                target_level=cfg.target_level,
                reset_level=cfg.reset_level,
                episode_length_s=120.0,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
