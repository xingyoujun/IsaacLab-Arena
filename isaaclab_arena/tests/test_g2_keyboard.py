# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G2 Tab-switched dual-arm keyboard control regression check."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_g2_dual_keyboard(simulation_app):
    import torch

    import carb.input
    import omni
    from isaaclab.devices.keyboard import Se3KeyboardCfg

    from isaaclab_arena.embodiments.g2.keyboard import G2DualArmKeyboard

    # Headless Kit has no window; mock only the OS event subscription, exercising the real device logic.
    with (
        patch.object(omni, "appwindow", MagicMock(), create=True),
        patch.object(carb.input, "acquire_input_interface", return_value=MagicMock()),
    ):
        keyboard = G2DualArmKeyboard(Se3KeyboardCfg(sim_device="cpu", pos_sensitivity=0.05))

    def event(key, pressed=True):
        keyboard._on_keyboard_event(
            SimpleNamespace(
                input=SimpleNamespace(name=key),
                type=carb.input.KeyboardEventType.KEY_PRESS if pressed else carb.input.KeyboardEventType.KEY_RELEASE,
            )
        )

    event("W")
    event("K")
    command = keyboard.advance()
    assert command.shape == (14,) and command[0] > 0 and command[6] == -1
    assert torch.count_nonzero(command[7:13]) == 0 and command[13] == 1
    event("K", False)
    event("TAB")
    event("TAB", False)
    event("W", False)
    command = keyboard.advance()
    assert torch.count_nonzero(command[:6]) == 0 and torch.count_nonzero(command[7:13]) == 0
    assert command[6] == -1 and command[13] == 1
    event("A")
    event("K")
    command = keyboard.advance()
    assert command[8] > 0 and command[6] == -1 and command[13] == -1
    event("L")
    event("A", False)
    command = keyboard.advance()
    assert torch.count_nonzero(command[:6]) == 0 and torch.count_nonzero(command[7:13]) == 0
    assert command[6] == 1 and command[13] == 1 and keyboard.active_arm == 0
    return True


def test_g2_dual_keyboard():
    assert run_function_with_persistent_simulation_app(_test_g2_dual_keyboard, headless=True)
