# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton runtime for the USB-C task, imported only when the task is configured."""

from isaaclab_newton.physics import NewtonMJWarpManager


class NewtonUsbcManager(NewtonMJWarpManager):
    """Clean up the task's procedural cable hooks during manager teardown."""

    @classmethod
    def _solver_specific_clear(cls) -> None:
        from .cables import _remove_connector_cable_builder_hooks

        _remove_connector_cable_builder_hooks()
        super()._solver_specific_clear()
