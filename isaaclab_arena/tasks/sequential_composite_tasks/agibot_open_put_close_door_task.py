# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Open a door, put an object inside, close the door: three subtasks in sequence."""

import numpy as np

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class AgibotOpenPutCloseDoorTask(SequentialTaskBase):
    """Sequential composite: open the container, place the object in it, close it again.

    The three subtasks are Arena's ``OpenDoorTask``, ``PickAndPlaceTask`` and ``CloseDoorTask``.
    The composite succeeds once all three have succeeded in order and, at the end, the object is
    still inside and the door is still closed; the open-door stage is naturally false by then, so
    its final state is a don't-care.
    """

    def __init__(
        self,
        openable_object,
        subtasks: list[TaskBase],
        episode_length_s: float | None = None,
        viewer_cfg: ViewerCfg | None = None,
    ):
        assert len(subtasks) == 3, "expects [open, put, close]"
        super().__init__(
            subtasks=subtasks,
            episode_length_s=episode_length_s,
            desired_subtask_success_state=[None, True, True],
        )
        self.openable_object = openable_object
        self._viewer_cfg = viewer_cfg

    def get_viewer_cfg(self) -> ViewerCfg:
        if self._viewer_cfg is not None:
            return self._viewer_cfg
        return get_viewer_cfg_look_at_object(lookat_object=self.openable_object, offset=np.array([-1.3, -1.3, 1.3]))

    def get_prompt(self) -> str:
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function not implemented yet.")
