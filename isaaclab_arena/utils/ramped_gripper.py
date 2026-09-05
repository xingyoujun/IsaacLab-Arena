# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A binary gripper action whose open/close target is ramped over time, then held constant.

Isaac Lab's ``BinaryJointPositionAction`` steps the joint target straight to the open or closed
position. On the Agibot that drives the fingers at about 0.9 m/s into whatever they are about to
pinch, and a light, thin-walled part is spat out rather than held (measured on the SDF-collided
sleeve of ``agibot_sleeve_on_peg``: 0/5 controlled grasps held). The cuMotion recording pipeline
never had this problem because its executor ramps every gripper command over
``DEFAULT_GRIPPER_RAMP_SECONDS`` (1.67 s) and then holds the target constant; with that ramp the
same grasp held 5/5. This term gives the teleoperation path the same behaviour: the target moves
towards the commanded state at a constant rate and, once there, stays put.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import fields

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils.configclass import configclass

DEFAULT_GRIPPER_RAMP_SECONDS = 200 / 120
"""Time a full open-to-closed travel is spread over; the cuMotion executor's value."""


class RampedBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary gripper action with a rate-limited target.

    The commanded state (open or closed) is resolved exactly as by the parent; the applied joint
    target then approaches it by at most a fixed fraction of the full travel per control step, so
    a full close takes ``cfg.ramp_seconds``. Once reached the target is constant, as before.
    """

    cfg: RampedBinaryJointPositionActionCfg

    def __init__(self, cfg: RampedBinaryJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        travel = (self._open_command - self._close_command).abs()
        steps = max(1.0, cfg.ramp_seconds / env.step_dt)
        self._max_step = travel / steps
        """Per-joint change of the applied target allowed in one control step."""
        self._ramped = self._open_command.unsqueeze(0).repeat(self.num_envs, 1)
        """The applied target, which trails the commanded state."""

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        desired = self._processed_actions
        delta = (desired - self._ramped).clamp(-self._max_step, self._max_step)
        self._ramped = self._ramped + delta
        self._processed_actions = self._ramped.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._ramped[env_ids] = self._open_command


@configclass
class RampedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for ``RampedBinaryJointPositionAction``."""

    class_type: type = RampedBinaryJointPositionAction

    ramp_seconds: float = DEFAULT_GRIPPER_RAMP_SECONDS
    """How long a full open-to-closed travel of the target takes."""


def install_ramped_gripper(env_cfg, ramp_seconds: float = DEFAULT_GRIPPER_RAMP_SECONDS) -> None:
    """Swap every binary gripper term in ``env_cfg`` for the ramped subclass.

    Intended as an ``env_cfg_callback``. Joint-space gripper terms (the recording pipeline's) are
    left alone: their targets are ramped by the executor that drives them.

    Args:
        env_cfg: The compiled environment configuration, patched in place.
        ramp_seconds: Duration of a full open-to-closed travel of the target.
    """
    swapped = []
    for term_name, term_cfg in vars(env_cfg.actions).items():
        if not isinstance(term_cfg, BinaryJointPositionActionCfg) or isinstance(
            term_cfg, RampedBinaryJointPositionActionCfg
        ):
            continue
        ramped = RampedBinaryJointPositionActionCfg(**{f.name: getattr(term_cfg, f.name) for f in fields(term_cfg)})
        ramped.class_type = RampedBinaryJointPositionAction
        ramped.ramp_seconds = ramp_seconds
        setattr(env_cfg.actions, term_name, ramped)
        swapped.append(term_name)
    if swapped:
        print(f"[ramped gripper] {', '.join(swapped)}: full travel over {ramp_seconds:.2f} s")
