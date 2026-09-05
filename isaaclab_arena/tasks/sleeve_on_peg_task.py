# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.spatial import any_object_in_frame_box, objects_at_rest, objects_upright
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

DEFAULT_SEAT_XY_TOLERANCE_M = 0.008
"""How far off the peg axis the seated sleeve's origin may sit.

A sleeve actually over the peg is within a millimetre or two of the axis. One standing on the
deck beside the peg is at least 32 mm off it (peg radius plus sleeve radius), so anything under
that separates the two cases."""

DEFAULT_SEAT_Z_TOLERANCE_M = (0.005, 0.010)
"""How far below and above the authored seated height the sleeve's origin may be.

The peg stands 48 mm proud of the deck, so a sleeve perched on its tip reads 48 mm high and is
rejected; the 10 mm upper margin admits an insert that stopped a little short of the deck."""

DEFAULT_UPRIGHT_THRESHOLD_RAD = math.radians(10.0)
"""How far the sleeve's axis may tilt. Tight, since a seated sleeve is held square by the peg."""

DEFAULT_REST_VELOCITY_M_S = 0.05
"""Speed below which the sleeve counts as settled.

Measured seated on the peg with the sleeve's SDF collider at Arena's 1/120 physics step: 0.0014
to 0.004 m/s, so this is a comfortable ten times the resting-contact noise while staying far
below a sleeve being carried, which at the teleop rate reads 0.2 m/s and up. (With the shipped
convexDecomposition collider the interfering hulls produced 0.076 m/s of standing velocity; that
collider is not used any more.)"""


@register_task
class SleeveOnPegTask(TaskBase):
    """Slide the removable sleeve back onto the platform's peg.

    Success needs the sleeve's origin inside a box fixed to the platform around the authored
    seated pose -- within ``seat_xy_tolerance_m`` of the peg axis and within
    ``seat_z_tolerance_m`` (below, above) of the seated height -- with the sleeve upright to
    ``upright_threshold_rad`` and settled below ``rest_velocity_m_s``. The box rides on the
    platform, so the check holds whether or not the platform itself has been moved.
    """

    def __init__(
        self,
        sleeve: ObjectBase,
        platform: ObjectBase,
        seated_offset_m: tuple[float, float, float],
        seat_xy_tolerance_m: float = DEFAULT_SEAT_XY_TOLERANCE_M,
        seat_z_tolerance_m: tuple[float, float] = DEFAULT_SEAT_Z_TOLERANCE_M,
        upright_threshold_rad: float = DEFAULT_UPRIGHT_THRESHOLD_RAD,
        rest_velocity_m_s: float = DEFAULT_REST_VELOCITY_M_S,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        viewer_cfg: ViewerCfg | None = None,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Put the sleeve onto the peg." if task_description is None else task_description,
        )
        self.sleeve = sleeve
        self.platform = platform
        self.seated_offset_m = seated_offset_m
        """The sleeve body's position in the platform body's frame when assembled."""
        self.seat_xy_tolerance_m = seat_xy_tolerance_m
        self.seat_z_tolerance_m = seat_z_tolerance_m
        self.upright_threshold_rad = upright_threshold_rad
        self.rest_velocity_m_s = rest_velocity_m_s
        self.viewer_cfg = viewer_cfg
        """Overrides the default over-the-shoulder view, e.g. with a robot head-mounted one."""

    def is_success(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Returns whether the sleeve is seated on the peg and at rest."""
        return (
            self._seated(env)
            & objects_upright(env, [self.sleeve.name], self.upright_threshold_rad)
            & objects_at_rest(env, [self.sleeve.name], self.rest_velocity_m_s)
        )

    def _seated(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Whether the sleeve's origin lies in the seat box, expressed in the platform's frame."""
        x, y, z = self.seated_offset_m
        below, above = self.seat_z_tolerance_m
        return any_object_in_frame_box(
            env,
            [self.sleeve.name],
            self.platform.name,
            x_range=(x - self.seat_xy_tolerance_m, x + self.seat_xy_tolerance_m),
            y_range=(y - self.seat_xy_tolerance_m, y + self.seat_xy_tolerance_m),
            z_range=(z - below, z + above),
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return TerminationsCfg(success=TerminationTermCfg(func=self.is_success, params={}))

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Mimic is not set up for SleeveOnPegTask yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.viewer_cfg is not None:
            return self.viewer_cfg
        return get_viewer_cfg_look_at_object(lookat_object=self.platform, offset=np.array([-0.9, -0.9, 0.85]))

    def apply_reachability_constraints(self) -> None:
        """The robot has to reach the sleeve where it lands and the peg it goes onto."""
        self._apply_reachability_constraints([self.sleeve, self.platform])


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    # Depends on the scene objects, so the task passes it in at construction time.
    success: TerminationTermCfg = MISSING
