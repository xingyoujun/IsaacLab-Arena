# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Preserve G2's recorded bowl placement contract across Arena task migrations."""

import torch

from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask


class G2PlacementTask(PickAndPlaceTask):
    """Use the original filtered contact, speed, and root-distance placement checks.

    Args:
        max_separation: Maximum absolute root separation along world X, Y, and Z.
        **kwargs: Remaining PickAndPlaceTask constructor arguments.
    """

    def __init__(self, *, max_separation: tuple[float, float, float], **kwargs):
        kwargs.setdefault("velocity_threshold", 0.1)
        super().__init__(**kwargs)
        assert len(max_separation) == 3 and all(value > 0 for value in max_separation)
        self.max_separation = max_separation

    def is_placed(self, env) -> torch.Tensor:
        """Return current placement validity for every environment, without latching success."""
        body = env.scene[self.pick_up_object.name].data
        target = env.scene[self.destination_location.name].data
        force = env.scene[self.contact_sensor_name].data.force_matrix_w.torch
        assert force.shape == (env.num_envs, 1, 1, 3)
        touching = torch.linalg.vector_norm(force, dim=-1).reshape(-1) > self.force_threshold
        slow = torch.linalg.vector_norm(body.root_lin_vel_w.torch, dim=-1) < self.velocity_threshold
        separation = (body.root_pos_w.torch - target.root_pos_w.torch).abs()
        limits = separation.new_tensor(self.max_separation)
        return touching & slow & (separation < limits).all(dim=-1)

    def get_termination_cfg(self):
        """Keep the original success condition with Arena's managed progress/reset lifecycle."""
        cfg = super().get_termination_cfg()
        cfg.success = [ProgressObjective(name="g2_placement", predicate_sequence=[self.is_placed])]
        return cfg
