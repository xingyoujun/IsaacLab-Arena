# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tab-switched keyboard control for the G2's two independent arms."""

import torch

import carb.input
from isaaclab.devices.keyboard import Se3Keyboard


class G2DualArmKeyboard(Se3Keyboard):
    """Send right pose/gripper followed by left pose/gripper, preserving each gripper across switches."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.reset()

    def reset(self):
        super().reset()
        self.active_arm = 0
        self._grippers = [False, False]
        self._pressed_keys = set()

    def __str__(self):
        return super().__str__() + "\n\tTab: switch right/left arm (starts on right); K: active gripper"

    def _on_keyboard_event(self, event, *args, **kwargs):
        key = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key not in self._pressed_keys:
                return True
            self._pressed_keys.remove(key)
        elif event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key in self._pressed_keys:
                return True
            self._pressed_keys.add(key)
            if key == "TAB":
                self._grippers[self.active_arm] = self._close_gripper
                self.active_arm = 1 - self.active_arm
                self._close_gripper = self._grippers[self.active_arm]
                # Require a fresh press after switching, including when a motion key was held.
                self._delta_pos.fill(0)
                self._delta_rot.fill(0)
                self._pressed_keys.intersection_update({"TAB"})
                print(f"[G2 teleop] Active arm: {'right' if self.active_arm == 0 else 'left'}")
                return True
        return super()._on_keyboard_event(event, *args, **kwargs)

    def advance(self) -> torch.Tensor:
        """Return 14 action values, with zero pose increments for the inactive arm."""
        active_command = super().advance()
        self._grippers[self.active_arm] = self._close_gripper
        command = active_command.new_zeros(14)
        command[6] = -1.0 if self._grippers[0] else 1.0
        command[13] = -1.0 if self._grippers[1] else 1.0
        offset = self.active_arm * 7
        command[offset : offset + 7] = active_command
        return command
