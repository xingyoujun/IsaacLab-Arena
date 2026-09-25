# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Insert an upright Factory peg into a fixed sleeve."""

import math
import torch

import warp as wp
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.utils.configclass import make_configclass


def peg_is_inserted(
    env,
    sleeve_cfg: SceneEntityCfg,
    peg_cfg: SceneEntityCfg,
    radial_tolerance_m: float,
    seating_tolerance_m: float,
    axis_tolerance_deg: float,
) -> torch.Tensor:
    """Check that the peg bottom reaches the bore floor with aligned local Z axes.

    Args:
        env: Simulation environment with sleeve and peg root poses in XYZW convention.
        sleeve_cfg: Sleeve whose origin lies at the inside of its closed end.
        peg_cfg: Upright peg whose origin lies at its bottom centre.
        radial_tolerance_m: Maximum lateral peg-bottom error in the sleeve frame.
        seating_tolerance_m: Maximum gap or penetration at the bore floor.
        axis_tolerance_deg: Maximum angle between the upright peg and sleeve axes.

    Returns:
        Per-environment success flags; proximity without seating is unsuccessful.
    """
    sleeve = env.scene[sleeve_cfg.name].data
    peg = env.scene[peg_cfg.name].data
    sleeve_quat = wp.to_torch(sleeve.root_quat_w)
    offset = quat_apply_inverse(sleeve_quat, wp.to_torch(peg.root_pos_w) - wp.to_torch(sleeve.root_pos_w))
    axis = torch.zeros_like(offset)
    axis[:, 2] = 1.0
    peg_axis = quat_apply_inverse(sleeve_quat, quat_apply(wp.to_torch(peg.root_quat_w), axis))
    return (
        (torch.linalg.vector_norm(offset[:, :2], dim=-1) <= radial_tolerance_m)
        & (offset[:, 2].abs() <= seating_tolerance_m)
        & (peg_axis[:, 2] >= math.cos(math.radians(axis_tolerance_deg)))
    )


class SleeveTask(NoTask):
    """Finish an episode when the movable peg is fully inserted into the fixed sleeve."""

    def __init__(
        self,
        sleeve,
        peg,
        radial_tolerance_m: float = 0.001,
        seating_tolerance_m: float = 0.003,
        axis_tolerance_deg: float = 1.0,
        episode_length_s: float = 120.0,
    ):
        super().__init__()
        self.peg = peg
        self.episode_length_s = episode_length_s
        self.task_description = "Pick up the peg and fully insert it into the fixed sleeve"
        success = TerminationTermCfg(
            func=peg_is_inserted,
            params={
                "sleeve_cfg": SceneEntityCfg(sleeve.name),
                "peg_cfg": SceneEntityCfg(peg.name),
                "radial_tolerance_m": radial_tolerance_m,
                "seating_tolerance_m": seating_tolerance_m,
                "axis_tolerance_deg": axis_tolerance_deg,
            },
        )
        self.termination_cfg = make_configclass(
            "SleeveTerminationsCfg",
            [
                ("success", TerminationTermCfg, success),
                ("time_out", TerminationTermCfg, TerminationTermCfg(func=mdp.time_out, time_out=True)),
                (
                    "object_dropped",
                    TerminationTermCfg,
                    TerminationTermCfg(
                        func=mdp.root_height_below_minimum,
                        params={"minimum_height": -0.10, "asset_cfg": SceneEntityCfg(peg.name)},
                    ),
                ),
            ],
        )()

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_metrics(self):
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.peg)]
