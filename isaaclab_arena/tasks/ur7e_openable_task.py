# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve RR success predicates when using the new Arena termination interface."""

from functools import partial

from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.open_door_task import OpenDoorTask
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg


class Ur7eOpenableTask(OpenDoorTask):
    """Terminate on the asset's original threshold, without a normalized-motion precondition.

    Drawer and press assets use normalized travel; knob assets override is_open
    with an absolute, either-direction radian threshold. Adding the upstream
    OpenDoorTask's 5%-of-range sequence would change the recorded benchmark.
    """

    def get_termination_cfg(self) -> TaskTerminationCfg:
        params = {}
        if self.target_joint_percentage_threshold is not None:
            params["threshold"] = self.target_joint_percentage_threshold
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="rr_asset_threshold",
                    predicate_sequence=[partial(self.openable_object.is_open, **params)],
                )
            ],
        )
