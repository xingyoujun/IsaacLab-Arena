# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
from isaaclab_arena.tasks.predicates.spatial import any_object_in_frame_box, objects_at_rest
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

DEFAULT_REST_VELOCITY_M_S = 0.05
"""Speed below which a sorted part counts as settled in its container."""

DEFAULT_ABOVE_RIM_MARGIN_M = 0.03
"""How far above a container's rim a part's origin may still count as inside it."""


@register_task
class TidyWorkbenchTask(TaskBase):
    """Put every loose part into the container that belongs to it.

    ``assignments`` pairs each part with its container: tools into the tool tray, raw stock into
    the parts tray, finished parts into the storage bin. Success needs every part's origin inside
    a box fixed to its own container -- the footprint shrunk by the container's ``WALL_MARGIN_M``
    on each side, from just below the floor to ``above_rim_margin_m`` over the rim -- and every
    part settled below ``rest_velocity_m_s``. Parts are located by their root, so an articulated
    part is judged by its base link.
    """

    def __init__(
        self,
        assignments: list[tuple[ObjectBase, ObjectBase]],
        rest_velocity_m_s: float = DEFAULT_REST_VELOCITY_M_S,
        above_rim_margin_m: float = DEFAULT_ABOVE_RIM_MARGIN_M,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        viewer_cfg: ViewerCfg | None = None,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=(
                "Tidy the workbench: put each part into its container."
                if task_description is None
                else task_description
            ),
        )
        assert assignments, "The task needs at least one (part, container) pair"
        self.assignments = assignments
        """``(part, container)`` pairs; the container classes carry the footprint constants."""
        self.parts = [part for part, _ in assignments]
        self.rest_velocity_m_s = rest_velocity_m_s
        self.above_rim_margin_m = above_rim_margin_m
        self.viewer_cfg = viewer_cfg
        """Overrides the default over-the-shoulder view, e.g. with a robot head-mounted one."""

    def is_success(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Returns whether every part sits inside its own container, at rest."""
        result = objects_at_rest(env, [part.name for part in self.parts], self.rest_velocity_m_s)
        for part, container in self.assignments:
            result &= self._part_in_container(env, part, container)
        return result

    def _part_in_container(self, env: ManagerBasedEnv, part: ObjectBase, container: ObjectBase) -> torch.Tensor:
        """Whether a part's origin lies within its container's interior box, in the container's frame."""
        kind = type(container)
        scale = float(container.scale[0])  # the footprint constants describe the unscaled asset
        half_x, half_y = (extent * scale for extent in kind.HALF_EXTENTS_XY_M)
        margin = kind.WALL_MARGIN_M * scale
        return any_object_in_frame_box(
            env,
            [part.name],
            container.name,
            x_range=(-(half_x - margin), half_x - margin),
            y_range=(-(half_y - margin), half_y - margin),
            z_range=(-0.01, kind.RIM_HEIGHT_M * scale + self.above_rim_margin_m),
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return TerminationsCfg(success=TerminationTermCfg(func=self.is_success, params={}))

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Mimic is not set up for TidyWorkbenchTask yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.viewer_cfg is not None:
            return self.viewer_cfg
        return get_viewer_cfg_look_at_object(lookat_object=self.parts[0], offset=np.array([-0.9, -0.9, 0.85]))

    def apply_reachability_constraints(self) -> None:
        """The robot has to reach every part where it lies and every container it goes into."""
        self._apply_reachability_constraints(self.parts + [container for _, container in self.assignments])


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    # Depends on the scene objects, so the task passes it in at construction time.
    success: TerminationTermCfg = MISSING
