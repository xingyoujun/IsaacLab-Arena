# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unscaled miniworkflow Astra drawer in the shared randomized UR7e scene."""

import os
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_asset, register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.ur7e_open_drawer_environment import (
    DrawerRR,
    Ur7eOpenDrawerEnvironment,
    Ur7eOpenDrawerEnvironmentCfg,
)


@register_asset
class DrawerMiniworkflowAstra(DrawerRR):
    name = "miniworkflow_astra_drawer"
    usd_path = os.environ.get(
        "ARENA_DRAWER_ASTRA_USD", "/home/ubuntu/playground/rr_ur/miniworkflow_astra_drawer_arena.usda"
    )
    openable_joint_name = "DrawerSlide"
    knob_body = "Drawer"
    knob_offset_local = (0.0, -0.103, 0.043)
    planning_box_size = (0.199, 0.229, 0.078)
    planning_box_center = (0.0, -0.0005, 0.039)


@dataclass
class Ur7eOpenDrawerAstraEnvironmentCfg(Ur7eOpenDrawerEnvironmentCfg):
    drawer_asset: str = "miniworkflow_astra_drawer"


@register_environment
class Ur7eOpenDrawerAstraEnvironment(
    Ur7eOpenDrawerEnvironment, ArenaEnvironmentFactory[Ur7eOpenDrawerAstraEnvironmentCfg]
):
    name = "ur7e_miniworkflow_astra_open_drawer"
    _legacy_argparse_cfg_type = Ur7eOpenDrawerAstraEnvironmentCfg
