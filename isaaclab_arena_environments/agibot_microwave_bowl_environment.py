# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Open the microwave, put the bowl inside, close the door -- with the Agibot.

Arena's Lightwheel microwave (``Microwave039``) stands on the RoboDojo table to the robot's left
with its door facing the table centre, so the left arm loads the cavity from the side and the door
swings across the front of the robot. The door is side-hinged (vertical axis at the far end of the
opening), its handle is a 30 mm bar only 6 mm proud and the door slab is 67 mm thick -- neither is
pinchable -- so, as in Arena's own door tasks, the episode starts with the door ajar and the robot
pushes it open and closed.
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

MICROWAVE_ORIGIN_ABOVE_BASE_M = 0.170
"""Microwave039's origin is at its centre; the body spans z -0.170..+0.170 (340 mm tall)."""

MICROWAVE_SCALE = 1.0
"""Must stay 1.0: Microwave039's colliders are SDF meshes and PhysX does not re-cook them for a
scaled prim -- at 0.7 the bowl rested 8 cm above the scaled top, fell through the scaled turntable and
the door left its joint limits (measured 2026-09-06). The full-size microwave is 586 wide (x), 475
deep (y) and 340 mm tall; on the 0.62 m table its top (0.96 m) runs through the Agibot's resting
hands (z 0.93-0.95 at y 0.03-0.46, x up to 0.31), so every placement that keeps the door within
reach collides with a resting hand. UNRESOLVED layout -- see docs/agibot/README.md."""

MICROWAVE_POSITION_XY = (0.20, 0.50)
"""Microwave centre (door face at world y 0.26, facing the table centre; body x -0.09..0.49,
y 0.26..0.74, 4 cm over the table's left edge). The resting left hand (pads at x 0.25-0.31,
y 0.19-0.30, z 0.94) is inside this body -- kept only as the starting point for the layout review."""

MICROWAVE_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)

DISC_OFFSET_M = (-0.060, 0.021, -0.132)
"""Turntable disc centre in the microwave frame (its joint anchor, top face at z -0.132)."""

_BOWL_POSITION_XY = (0.15, -0.08)
"""The bowl starts in the work band just right of centre, where both arms reach it (left 20/12/14
leaned poses, right 24/10/16) and outside the door's swing: the 330 mm door pivots at (0.01, 0.32)
and sweeps the quarter-disc toward -y, so anything farther than 0.33 m from the hinge is safe (this
spot is 0.42 m away). Only the left arm reaches into the cavity, so the bowl must be where the left
arm can take it."""


def disc_world_position(
    table_top_z: float = TABLE_TOP_Z, xy: tuple[float, float] = MICROWAVE_POSITION_XY, scale: float = MICROWAVE_SCALE
) -> tuple[float, float, float]:
    """World position of the turntable's centre (top face) for a microwave at ``xy`` and ``scale``."""
    return (
        xy[0] + scale * DISC_OFFSET_M[0],
        xy[1] + scale * DISC_OFFSET_M[1],
        table_top_z + scale * (MICROWAVE_ORIGIN_ABOVE_BASE_M + DISC_OFFSET_M[2]),
    )


@dataclass
class AgibotMicrowaveBowlEnvironmentCfg(AgibotTabletopEnvironmentCfg):
    """Configure the Agibot microwave environment."""

    arm_mode: str = "dual"

    microwave_scale: float = MICROWAVE_SCALE
    """Uniform scale of the microwave (see ``MICROWAVE_SCALE`` for why not 1.0)."""

    microwave_x: float = MICROWAVE_POSITION_XY[0]
    microwave_y: float = MICROWAVE_POSITION_XY[1]
    """Microwave centre on the table (its door faces -y)."""

    object: str = "bowl"
    """Asset that goes into the microwave. The RoboDojo bowl (110 x 60 mm, rim pinch) by default."""

    reset_openness: float = 0.2
    """Door openness at reset (0 closed .. 1 open). Ajar, as in Arena's open-door task: a parallel
    gripper cannot pull this handle, it pushes the door."""

    open_threshold: float = 0.8
    """Openness above which the door counts as open (stage 1 done)."""

    closed_threshold: float = 0.05
    """Openness below which the door counts as closed (stage 3 done)."""


@register_environment
class AgibotMicrowaveBowlEnvironment(ArenaEnvironmentFactory[AgibotMicrowaveBowlEnvironmentCfg]):
    """Open the microwave, put the bowl in, close the door."""

    name: str = "agibot_microwave_bowl"
    _legacy_argparse_cfg_type = AgibotMicrowaveBowlEnvironmentCfg

    def build(self, cfg: AgibotMicrowaveBowlEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_composite_tasks.agibot_open_put_close_door_task import (
            AgibotOpenPutCloseDoorTask,
        )
        from isaaclab_arena.utils.pose import Pose

        background, surroundings, light = build_tabletop_stage(self, cfg, TABLE_TOP_Z)
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        microwave.scale = (cfg.microwave_scale,) * 3  # Microwave.__init__ takes no scale argument
        microwave.set_initial_pose(
            Pose(
                position_xyz=(
                    cfg.microwave_x,
                    cfg.microwave_y,
                    TABLE_TOP_Z + cfg.microwave_scale * MICROWAVE_ORIGIN_ABOVE_BASE_M,
                ),
                rotation_xyzw=MICROWAVE_ROTATION_XYZW,
            )
        )
        pick_object = self.asset_registry.get_asset_by_name(cfg.object)()
        object_half_height = getattr(type(pick_object), "HALF_HEIGHT_M", 0.0)
        pick_object.set_initial_pose(
            Pose(
                position_xyz=(*_BOWL_POSITION_XY, TABLE_TOP_Z + object_half_height), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)
            )
        )
        # The microwave articulation owns the turntable body; this read-only reference gives the
        # place task its pose and geometry.
        disc = ObjectReference(
            name="microwave_disc",
            parent_asset=microwave,
            prim_path="{ENV_REGEX_NS}/microwave/Microwave039_Disc001",
        )
        embodiment = build_agibot(self, cfg)
        teleop_device = build_teleop_device(self, cfg)
        scene = Scene(assets=[background, microwave, pick_object, disc, surroundings, light])

        open_task = OpenDoorTask(
            openable_object=microwave,
            openness_threshold=cfg.open_threshold,
            reset_openness=cfg.reset_openness,
            task_description="Open the microwave door.",
        )
        put_task = PickAndPlaceTask(
            pick_up_object=pick_object,
            destination_object=microwave,
            destination_location=disc,
            background_scene=background,
            task_description=f"Put the {cfg.object} into the microwave.",
        )
        close_task = CloseDoorTask(
            openable_object=microwave,
            closedness_threshold=cfg.closed_threshold,
            reset_openness=cfg.reset_openness,
            task_description="Close the microwave door.",
        )

        def env_cfg_callback(env_cfg):
            install_agibot_control_stack(env_cfg, cfg)
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=AgibotOpenPutCloseDoorTask(
                openable_object=microwave,
                subtasks=[open_task, put_task, close_task],
                episode_length_s=180.0,
                viewer_cfg=embodiment.get_head_viewer_cfg() if cfg.head_view else None,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
