# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Tab-switching two-arm keyboard.

Kept apart from the device class on purpose: the device imports ``carb`` through Isaac Lab's
``Se3Keyboard``, which only exists once Kit is running, while this configuration is imported by
the device registry in plain Python processes (asset validation, CLI parsing, tests). The
``class_type`` string is resolved by Isaac Lab when the device is actually created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_arena.devices.dual_arm_keyboard import DualArmSe3Keyboard


@configclass
class DualArmSe3KeyboardCfg(Se3KeyboardCfg):
    """Configuration for the Tab-switching two-arm keyboard."""

    class_type: type[DualArmSe3Keyboard] | str = "{DIR}.dual_arm_keyboard:DualArmSe3Keyboard"
