# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Width-normalized Articraft drawer in the shared UR7e workcell."""

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
class DrawerArticraft(DrawerRR):
    """Articraft drawer uniformly scaled by 0.495 to the reference drawer's 198 mm width."""

    name = "drawer_articraft"
    usd_path = os.environ.get(
        "ARENA_DRAWER_ARTICRAFT_USD", "/home/ubuntu/playground/rr_ur/drawer_articraft/drawer_articraft.usd"
    )
    spawn_cfg_addon = {}
    """The imported USD already fixes the enclosure to the world."""
    openable_joint_name = "drawer_slide"
    knob_body = "drawer"
    knob_offset_local = (0.0, -0.315 * 0.495, 0.065 * 0.495)
    planning_box_size = (0.4 * 0.495, 0.599 * 0.495, 0.15 * 0.495)
    """Closed asset bounding box, including the knob, in metres."""
    planning_box_center = (0.0, -0.0495 * 0.495, 0.075 * 0.495)
    """Centre of the planning box expressed in the enclosure frame."""


@dataclass
class Ur7eOpenDrawerArticraftEnvironmentCfg(Ur7eOpenDrawerEnvironmentCfg):
    """Preserve the original drawer pose distribution with the Articraft asset."""

    drawer_asset: str = "drawer_articraft"


@register_environment
class Ur7eOpenDrawerArticraftEnvironment(
    Ur7eOpenDrawerEnvironment, ArenaEnvironmentFactory[Ur7eOpenDrawerArticraftEnvironmentCfg]
):
    """Open the width-normalized Articraft drawer in the calibrated UR7e scene."""

    name = "ur7e_open_drawer_articraft"
    _legacy_argparse_cfg_type = Ur7eOpenDrawerArticraftEnvironmentCfg
