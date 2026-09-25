# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Comparison toasters in the unchanged USDcraft pressing workcell."""

import os
from dataclasses import dataclass

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.register import register_asset, register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.ur7e_press_toaster_environment import (
    ToasterRR,
    Ur7ePressToasterEnvironment,
    Ur7ePressToasterEnvironmentCfg,
)


@register_asset
class ToasterMiniworkflowGptsol(ToasterRR):
    name = "miniworkflow_gptsol_toast"
    usd_path = os.environ.get(
        "ARENA_TOASTER_GPTSOL_USD", "/home/ubuntu/playground/rr_ur/miniworkflow_gptsol_toast_arena.usda"
    )
    openable_joint_name = "LeverPrismatic"
    lever_body = "Lever"
    paddle_offset_local = (0.0, 0.0, 0.0)
    paddle_half_height = 0.007
    planning_box_size = (0.1434, 0.1952, 0.25545)
    planning_box_center = (0.0, -0.0135, 0.127725)


@register_asset
class ToasterMiniworkflowAstra(ToasterRR):
    name = "miniworkflow_astra_toast"
    usd_path = os.environ.get(
        "ARENA_TOASTER_ASTRA_USD", "/home/ubuntu/playground/rr_ur/miniworkflow_astra_toast_arena.usda"
    )
    openable_joint_name = "BreadLift"
    lever_body = "Lift"
    paddle_offset_local = (0.0, -0.014, -0.0005)
    paddle_half_height = 0.0035
    planning_box_size = (0.1434, 0.2645, 0.168)
    planning_box_center = (0.0, -0.01425, 0.084)


@dataclass
class Ur7eGptsolPressToasterCfg(Ur7ePressToasterEnvironmentCfg):
    toaster_asset: str = "miniworkflow_gptsol_toast"


@register_environment
class Ur7eGptsolPressToaster(Ur7ePressToasterEnvironment, ArenaEnvironmentFactory[Ur7eGptsolPressToasterCfg]):
    name = "ur7e_miniworkflow_gptsol_press_toaster"
    _legacy_argparse_cfg_type = Ur7eGptsolPressToasterCfg


@dataclass
class Ur7eAstraPressToasterCfg(Ur7ePressToasterEnvironmentCfg):
    toaster_asset: str = "miniworkflow_astra_toast"


@register_environment
class Ur7eAstraPressToaster(Ur7ePressToasterEnvironment, ArenaEnvironmentFactory[Ur7eAstraPressToasterCfg]):
    name = "ur7e_miniworkflow_astra_press_toaster"
    _legacy_argparse_cfg_type = Ur7eAstraPressToasterCfg


@register_asset
class ToasterArticraft(ToasterRR):
    name = "articraft_toast"
    usd_path = os.environ.get(
        "ARENA_TOASTER_ARTICRAFT_USD", "/home/ubuntu/playground/rr_ur/articraft_toast/articraft_toast.usd"
    )
    # The URDF import already contains a fixed world joint.
    spawn_cfg_addon = {"rigid_props": sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)}
    openable_joint_name = "housing_to_carriage_lever"
    lever_body = "carriage_lever"
    paddle_offset_local = (0.0000005 * 0.795522222222, -0.015 * 0.795522222222, 0.0000025 * 0.795522222222)
    paddle_half_height = 0.0119975 * 0.795522222222
    planning_box_size = (0.143194, 0.25456711, 0.16785518)
    planning_box_center = (0.0, -0.01193284, 0.08392759)


@dataclass
class Ur7eArticraftPressToasterCfg(Ur7ePressToasterEnvironmentCfg):
    toaster_asset: str = "articraft_toast"


@register_environment
class Ur7eArticraftPressToaster(Ur7ePressToasterEnvironment, ArenaEnvironmentFactory[Ur7eArticraftPressToasterCfg]):
    name = "ur7e_articraft_press_toaster"
    _legacy_argparse_cfg_type = Ur7eArticraftPressToasterCfg
