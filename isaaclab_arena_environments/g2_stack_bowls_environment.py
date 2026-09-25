# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Genie Sim's ``stack_bowls`` tabletop task (``table_task_2_g2_op``), rebuilt in Arena for the AgiBot G2.

The scene adapts Genie Sim's table and three ``benchmark_bowl_025`` bowls to a taller G2 work surface in a
simple furnished room. The task is a two-step sequence: place the first bowl into the second one,
then the third bowl on top of the first. Each step succeeds on contact with the destination bowl plus an
axis-aligned proximity check, mirroring Genie Sim's ``Stack`` predicate tolerance of 5 cm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class G2StackBowlsEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the G2 stack-bowls environment."""

    embodiment: str = "g2"
    bowl: str = "genie_benchmark_bowl"
    """Registered object used for all three bowls."""
    bowl_positions: list[float] = field(default_factory=lambda: [-0.12, -0.24, -0.10, 0.0, -0.12, 0.24])
    """Flattened table-frame x/y pairs of bowl_1 (first picked), bowl_2 (first destination) and bowl_3 (stacked
    last), arranged along the near edge within the two arms' workspace."""
    table_height_m: float = 0.75
    """Floor-to-tabletop height; tabletop stays at world z=0."""
    robot_position: list[float] | None = None
    """Optional base x/y/z override; by default the chassis follows the floor at (-0.65, 0, -table_height_m)."""
    arm_mode: str = "dual_arm"
    """Use dual_arm for Tab-switched keyboard control, or left/right for a single arm."""
    stack_tolerance_m: float = 0.05
    """Max x/y separation between stacked bowl centres for a stacking step to count as done."""
    hdr: str | None = "brown_photostudio_robolab"
    """HDR dome texture; the default matches Genie Sim's ``room_3`` background lighting."""
    light_intensity: float = 1000.0
    episode_length_s: float = 70.0
    teleop_device: str | None = None

    def __post_init__(self) -> None:
        assert len(self.bowl_positions) == 6, "bowl_positions must hold three x/y pairs"
        assert self.robot_position is None or len(self.robot_position) == 3, "robot_position must be x/y/z"
        assert self.table_height_m > 0.0, "table_height_m must be positive"
        assert self.arm_mode in ("dual_arm", "left", "right"), "Unsupported arm_mode"
        assert self.episode_length_s > 0.0, "episode_length_s must be greater than zero"


@register_environment
class G2StackBowlsEnvironment(ArenaEnvironmentFactory[G2StackBowlsEnvironmentCfg]):
    """Registered provider for the G2 stack-bowls environment."""

    name: str = "g2_stack_bowls"
    _legacy_argparse_cfg_type = G2StackBowlsEnvironmentCfg

    def build(self, cfg: G2StackBowlsEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.background import Background
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import AtPosition, IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
        from isaaclab_arena.tasks.g2_placement_task import G2PlacementTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("genie_benchmark_table")(height_m=cfg.table_height_m)
        room = Background(
            name="g2_workroom",
            usd_path=str(
                Path(__file__).resolve().parents[1] / "isaaclab_arena/embodiments/g2/assets/stack_bowls_room.usda"
            ),
            object_min_z=-cfg.table_height_m,
            initial_pose=Pose(position_xyz=(0.0, 0.0, -cfg.table_height_m), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/genie_benchmark_table/entity",
            parent_asset=background,
        )
        table_reference.add_relation(IsAnchor())

        bowl_cls = self.asset_registry.get_asset_by_name(cfg.bowl)
        bowls = []
        xy_pairs = list(zip(cfg.bowl_positions[0::2], cfg.bowl_positions[1::2]))
        for index, (x, y) in enumerate(xy_pairs, start=1):
            bowl = bowl_cls(instance_name=f"bowl_{index}")
            bowl.add_relation(On(table_reference))
            bowl.add_relation(AtPosition(x=x, y=y))
            bowls.append(bowl)
        bowl_1, bowl_2, bowl_3 = bowls

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")(
            initial_pose=Pose(position_xyz=(0.0, 0.0, -cfg.table_height_m - 0.08), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras, arm_mode=ArmMode(cfg.arm_mode)
        )
        robot_position = cfg.robot_position or [-0.65, 0.0, -cfg.table_height_m]
        embodiment.set_initial_pose(Pose(position_xyz=tuple(robot_position), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()

        scene = Scene(assets=[background, room, ground_plane, light, table_reference, *bowls])

        tolerance = cfg.stack_tolerance_m
        max_separation = (tolerance, tolerance, 0.1)
        stack_first = G2PlacementTask(
            pick_up_object=bowl_1,
            destination_location=bowl_2,
            background_scene=background,
            max_separation=max_separation,
            task_description="Stack bowl_1 into bowl_2",
        )
        stack_second = G2PlacementTask(
            pick_up_object=bowl_3,
            destination_location=bowl_1,
            background_scene=background,
            max_separation=max_separation,
            task_description="Stack bowl_3 on top of bowl_1",
        )
        task = CompositeTaskBase(
            subtasks_are_sequential=True,
            subtasks=[stack_first, stack_second],
            episode_length_s=cfg.episode_length_s,
            task_description="stack bowls",
            desired_subtask_success_state=[True, True],
        )

        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(-2.6, -3.0, 1.5), lookat=(0.0, 0.0, 0.1))
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_set_viewer_cfg,
        )
