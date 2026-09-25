# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Method-qualified task names; legacy names and recorded articulation keys remain valid."""

from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.ur7e_open_drawer_articraft_environment import (
    Ur7eOpenDrawerArticraftEnvironment,
    Ur7eOpenDrawerArticraftEnvironmentCfg,
)
from isaaclab_arena_environments.ur7e_open_drawer_environment import (
    Ur7eOpenDrawerEnvironment,
    Ur7eOpenDrawerEnvironmentCfg,
    Ur7eOpenDrawerGpt56Environment,
    Ur7eOpenDrawerGpt56EnvironmentCfg,
)
from isaaclab_arena_environments.ur7e_press_toaster_environment import (
    Ur7ePressToasterEnvironment,
    Ur7ePressToasterEnvironmentCfg,
)


@dataclass
class Ur7eUsdcraftOpenDrawerCfg(Ur7eOpenDrawerEnvironmentCfg):
    pass


@register_environment
class Ur7eUsdcraftOpenDrawer(Ur7eOpenDrawerEnvironment, ArenaEnvironmentFactory[Ur7eUsdcraftOpenDrawerCfg]):
    name = "ur7e_usdcraft_open_drawer"
    _legacy_argparse_cfg_type = Ur7eUsdcraftOpenDrawerCfg


@dataclass
class Ur7eUsdcraftPressToasterCfg(Ur7ePressToasterEnvironmentCfg):
    pass


@register_environment
class Ur7eUsdcraftPressToaster(Ur7ePressToasterEnvironment, ArenaEnvironmentFactory[Ur7eUsdcraftPressToasterCfg]):
    name = "ur7e_usdcraft_press_toaster"
    _legacy_argparse_cfg_type = Ur7eUsdcraftPressToasterCfg


@dataclass
class Ur7eArticraftOpenDrawerCfg(Ur7eOpenDrawerArticraftEnvironmentCfg):
    pass


@register_environment
class Ur7eArticraftOpenDrawer(Ur7eOpenDrawerArticraftEnvironment, ArenaEnvironmentFactory[Ur7eArticraftOpenDrawerCfg]):
    name = "ur7e_articraft_open_drawer"
    _legacy_argparse_cfg_type = Ur7eArticraftOpenDrawerCfg


@dataclass
class Ur7eMiniworkflowGptsolOpenDrawerCfg(Ur7eOpenDrawerGpt56EnvironmentCfg):
    pass


@register_environment
class Ur7eMiniworkflowGptsolOpenDrawer(
    Ur7eOpenDrawerGpt56Environment, ArenaEnvironmentFactory[Ur7eMiniworkflowGptsolOpenDrawerCfg]
):
    name = "ur7e_miniworkflow_gptsol_open_drawer"
    _legacy_argparse_cfg_type = Ur7eMiniworkflowGptsolOpenDrawerCfg
