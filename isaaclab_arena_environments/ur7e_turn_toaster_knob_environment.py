# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn the USDcraft toaster dial while it stands on two full-size boxx supports."""

import math
from dataclasses import dataclass, replace
from pathlib import Path

from isaaclab.managers import EventTermCfg, SceneEntityCfg

from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset, register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.ur7e_press_toaster_baselines_environment import (
    ToasterArticraft,
    ToasterMiniworkflowAstra,
    ToasterMiniworkflowGptsol,
)
from isaaclab_arena_environments.ur7e_press_toaster_environment import (
    TOASTER_FAR_XY,
    TOASTER_NEAR_XY,
    TOASTER_YAW_JITTER_RAD,
    ToasterRR,
    Ur7ePressToasterEnvironment,
    Ur7ePressToasterEnvironmentCfg,
    randomize_stacked_object_poses,
)

BOX_HEIGHT_M = 0.1178
BOX_SIZE_M = (0.3225, 0.2581330148, BOX_HEIGHT_M)


@register_asset
class BoxxSupport(LibraryObject):
    name = "boxx_support"
    tags = ["support"]
    object_type = ObjectType.RIGID
    usd_path = str(Path(__file__).resolve().parents[1] / "tools/rr_sim2real/asset_overlays/boxx_support.usda")


@register_asset
class UsdcraftToasterKnob(ToasterRR):
    name = "usdcraft_toast_knob"
    openable_joint_name = "browning_rotation"
    knob_body = "browning_dial"
    knob_offset_local = (0.0, -0.0051, 0.0)
    carriage_joint = "carriage_slide"
    carriage_rest = 0.0

    def rotate_revolute_joint(self, env, env_ids, asset_cfg=None, percentage=0.0):
        """Reset the dial and the unused carriage so accidental lever contacts cannot survive a reset."""
        import torch

        assert percentage == 0.0, "Knob episodes reset at zero radians, not a normalized joint limit"
        obj = env.unwrapped.scene.articulations[self.name]
        ids = env_ids.to(env.unwrapped.device) if env_ids is not None else None
        n = env.unwrapped.num_envs if ids is None else len(ids)
        joint_ids = torch.tensor(
            [
                obj.data.joint_names.index(self.openable_joint_name),
                obj.data.joint_names.index(self.carriage_joint),
            ],
            dtype=torch.int32,
            device=env.unwrapped.device,
        )
        zeros = torch.zeros((n, 2), device=env.unwrapped.device)
        positions = zeros.clone()
        positions[:, 1] = self.carriage_rest
        obj.write_joint_position_to_sim_index(position=positions, joint_ids=joint_ids, env_ids=ids)
        obj.write_joint_velocity_to_sim_index(velocity=zeros, joint_ids=joint_ids, env_ids=ids)

    def is_open(self, env, asset_cfg=None, threshold=None):
        """Succeed on either-direction dial motion beyond the configured radian threshold."""
        import warp as wp

        obj = env.unwrapped.scene.articulations[self.name]
        index = obj.data.joint_names.index(self.openable_joint_name)
        used_threshold = math.radians(2.0) if threshold is None else float(threshold)
        return wp.to_torch(obj.data.joint_pos)[:, index].abs() > used_threshold


@register_asset
class GptsolToasterKnob(UsdcraftToasterKnob):
    name = "miniworkflow_gptsol_toast_knob"
    usd_path = ToasterMiniworkflowGptsol.usd_path
    openable_joint_name = "DialRevolute"
    knob_body = "BrowningDial"
    knob_offset_local = (0.0, -0.00325, 0.0)
    carriage_joint = "LeverPrismatic"
    carriage_rest = 0.004


@register_asset
class AstraToasterKnob(UsdcraftToasterKnob):
    name = "miniworkflow_astra_toast_knob"
    usd_path = ToasterMiniworkflowAstra.usd_path
    openable_joint_name = "BrowningDial"
    knob_body = "Dial"
    knob_offset_local = (0.0, -0.005925, 0.0)
    carriage_joint = "BreadLift"


@register_asset
class ArticraftToasterKnob(UsdcraftToasterKnob):
    name = "articraft_toast_knob"
    usd_path = ToasterArticraft.usd_path
    spawn_cfg_addon = ToasterArticraft.spawn_cfg_addon
    openable_joint_name = "housing_to_browning_dial"
    knob_body = "browning_dial"
    knob_offset_local = (0.0, 0.0, 0.00875078)
    carriage_joint = "housing_to_carriage_lever"


@dataclass
class Ur7eTurnToasterKnobEnvironmentCfg(Ur7ePressToasterEnvironmentCfg):
    embodiment: str = "ur7e_robotiq_joint_pos"
    toaster_asset: str = "usdcraft_toast_knob"
    pedestal: bool = False
    randomize_toaster_pose: bool = False
    toaster_x: float = -0.16
    toaster_y: float = 0.12
    turn_threshold_deg: float = 2.0


@register_environment
class Ur7eTurnToasterKnobEnvironment(
    Ur7ePressToasterEnvironment,
    ArenaEnvironmentFactory[Ur7eTurnToasterKnobEnvironmentCfg],
):
    name = "ur7e_usdcraft_turn_toaster_knob"
    _legacy_argparse_cfg_type = Ur7eTurnToasterKnobEnvironmentCfg

    def build(self, cfg: Ur7eTurnToasterKnobEnvironmentCfg):
        """Reuse the calibrated pressing scene with two static box supports instead of the drawer pedestal."""
        assert not cfg.pedestal, "This task uses two boxx supports, not the drawer pedestal"
        assert cfg.turn_threshold_deg > 0.0, "Use a nonzero threshold to reject numerical noise"
        raised_cfg = replace(
            cfg,
            toaster_z=cfg.toaster_z + 2 * BOX_HEIGHT_M,
            pressed_threshold=math.radians(cfg.turn_threshold_deg),
            randomize_toaster_pose=False,
        )
        arena_env = super().build(raised_cfg)
        yaw = math.radians(cfg.toaster_yaw_deg)
        quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        for i in range(2):
            support = BoxxSupport(instance_name=f"box_support_{i}")
            support.set_initial_pose(
                Pose(
                    position_xyz=(
                        cfg.toaster_x,
                        cfg.toaster_y,
                        cfg.toaster_z + i * BOX_HEIGHT_M,
                    ),
                    rotation_xyzw=quat,
                )
            )
            arena_env.scene.add_asset(support)
        if cfg.randomize_toaster_pose:
            previous_callback = arena_env.env_cfg_callback

            def randomized_stack(env_cfg):
                env_cfg = previous_callback(env_cfg) if previous_callback else env_cfg
                env_cfg.events.reset_knob_stack = EventTermCfg(
                    func=randomize_stacked_object_poses,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": (TOASTER_NEAR_XY[0], TOASTER_FAR_XY[0]),
                            "y": (TOASTER_NEAR_XY[1], TOASTER_FAR_XY[1]),
                            "z": (cfg.toaster_z, cfg.toaster_z),
                            "yaw": (-TOASTER_YAW_JITTER_RAD, TOASTER_YAW_JITTER_RAD),
                        },
                        "asset_cfgs": [
                            SceneEntityCfg("box_support_0"),
                            SceneEntityCfg("box_support_1"),
                            SceneEntityCfg(cfg.toaster_asset),
                        ],
                        "z_offsets": [0.0, BOX_HEIGHT_M, 2 * BOX_HEIGHT_M],
                        "yaw_offsets": [0.0, 0.0, 0.0],
                    },
                )
                return env_cfg

            arena_env.env_cfg_callback = randomized_stack
        arena_env.task.task_description = "Grasp the toaster dial and rotate it slightly."
        return arena_env


@dataclass
class Ur7eGptsolTurnToasterKnobCfg(Ur7eTurnToasterKnobEnvironmentCfg):
    toaster_asset: str = "miniworkflow_gptsol_toast_knob"


@register_environment
class Ur7eGptsolTurnToasterKnob(
    Ur7eTurnToasterKnobEnvironment,
    ArenaEnvironmentFactory[Ur7eGptsolTurnToasterKnobCfg],
):
    name = "ur7e_miniworkflow_gptsol_turn_toaster_knob"
    _legacy_argparse_cfg_type = Ur7eGptsolTurnToasterKnobCfg


@dataclass
class Ur7eAstraTurnToasterKnobCfg(Ur7eTurnToasterKnobEnvironmentCfg):
    toaster_asset: str = "miniworkflow_astra_toast_knob"


@register_environment
class Ur7eAstraTurnToasterKnob(Ur7eTurnToasterKnobEnvironment, ArenaEnvironmentFactory[Ur7eAstraTurnToasterKnobCfg]):
    name = "ur7e_miniworkflow_astra_turn_toaster_knob"
    _legacy_argparse_cfg_type = Ur7eAstraTurnToasterKnobCfg


@dataclass
class Ur7eArticraftTurnToasterKnobCfg(Ur7eTurnToasterKnobEnvironmentCfg):
    toaster_asset: str = "articraft_toast_knob"


@register_environment
class Ur7eArticraftTurnToasterKnob(
    Ur7eTurnToasterKnobEnvironment,
    ArenaEnvironmentFactory[Ur7eArticraftTurnToasterKnobCfg],
):
    name = "ur7e_articraft_turn_toaster_knob"
    _legacy_argparse_cfg_type = Ur7eArticraftTurnToasterKnobCfg
