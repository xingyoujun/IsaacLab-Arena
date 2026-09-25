# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""An RMPFlow arm term that refuses to drive the gripper into the work surface.

The Agibot arms are position-servoed with very high stiffness and no damping, and RMPFlow runs
with ``ignore_robot_state_updates``, so an arm commanded below the table does not yield -- it
presses, saturates torque, and a gripper closing while pressed into the table throws whatever it
was closing on (measured: 1.04 m/s finger-body speed pressed against 0.57 m/s free, and the joint
velocity limit exceeded 2.4x). The fix belongs at the command level: this term clamps each control
step's descent so the lowest protected gripper body never goes below the surface plus a clearance.
Lateral motion and ascent keep full authority, so the guard is invisible until it is needed.
"""

from __future__ import annotations

import re
import torch
from dataclasses import fields

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.arm_target_hold import TargetHoldingRMPFlowAction, TargetHoldingRMPFlowActionCfg

DEFAULT_GRIPPER_BODY_PATTERN = r"(Pad|Support|_0[01]_|_2_)_?Link$"
"""Regex selecting the gripper bodies to protect; matches the Agibot's finger pads and linkage."""

DEFAULT_CONTACT_OFFSET_M = 0.018
"""How far a protected body's origin sits above the surface when the gripper rests on it.

Measured on the Agibot as 15.4 mm with the gripper open and 18.3 mm closed; the larger value so
the guard holds in both states."""


class SurfaceGuardedRMPFlowAction(TargetHoldingRMPFlowAction):
    """Target-holding RMPFlow arm term whose commanded descent is clamped at the work surface."""

    cfg: SurfaceGuardedRMPFlowActionCfg

    def __init__(self, cfg: SurfaceGuardedRMPFlowActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        side = "right" if cfg.body_name.startswith("right") else "left"
        matcher = re.compile(cfg.gripper_body_pattern)
        body_names = list(self._asset.data.body_names)
        self._guard_body_ids = [i for i, b in enumerate(body_names) if b.startswith(side) and matcher.search(b)]
        assert (
            self._guard_body_ids
        ), f"SurfaceGuard found no {side} gripper bodies matching {cfg.gripper_body_pattern!r}"
        self._floor_z = cfg.surface_z + cfg.contact_offset_m
        self._z_scale = float(self._scale[0, 2].item()) if self._scale.dim() == 2 else float(self._scale[2].item())
        print(
            f"[surface guard] {cfg.body_name}: floor z {self._floor_z:.4f} m, {len(self._guard_body_ids)} bodies"
            f" e.g. {body_names[self._guard_body_ids[0]]}"
        )

    def process_actions(self, actions: torch.Tensor):
        """Clamp the descent channel to the headroom above the floor, then process as usual.

        Args:
            actions: This term's slice of the action vector, a delta pose of shape (num_envs, 6).
        """
        body_pos_z = torch.as_tensor(self._asset.data.body_pos_w)[..., 2]
        lowest = body_pos_z[:, self._guard_body_ids].min(dim=1).values
        headroom = ((lowest - self._floor_z) / self._z_scale).clamp(min=0.0)
        guarded = actions.clone()
        guarded[:, 2] = torch.maximum(guarded[:, 2], -headroom)
        super().process_actions(guarded)


@configclass
class SurfaceGuardedRMPFlowActionCfg(TargetHoldingRMPFlowActionCfg):
    """Configuration for ``SurfaceGuardedRMPFlowAction``."""

    class_type: type = SurfaceGuardedRMPFlowAction

    surface_z: float = 0.0
    """World height of the work surface, in metres."""

    contact_offset_m: float = DEFAULT_CONTACT_OFFSET_M
    """Clearance kept between the lowest protected body's origin and the surface."""

    gripper_body_pattern: str = DEFAULT_GRIPPER_BODY_PATTERN
    """Regex selecting the gripper bodies whose height is guarded."""


def install_surface_guard(env_cfg, surface_z: float, contact_offset_m: float = DEFAULT_CONTACT_OFFSET_M) -> None:
    """Swap every relative-mode RMPFlow arm term in ``env_cfg`` for the surface-guarded subclass.

    Run it after ``install_arm_target_hold``; the guarded term keeps the target-hold behaviour.
    Joint-space arm terms (the recording pipeline's) are left alone.

    Args:
        env_cfg: The compiled environment configuration, patched in place.
        surface_z: World height of the work surface, in metres.
        contact_offset_m: Clearance kept between the lowest gripper body's origin and the surface.
    """
    from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg

    swapped = []
    for term_name, term_cfg in vars(env_cfg.actions).items():
        if not isinstance(term_cfg, RMPFlowActionCfg) or isinstance(term_cfg, SurfaceGuardedRMPFlowActionCfg):
            continue
        if not term_cfg.use_relative_mode:
            continue
        values = {f.name: getattr(term_cfg, f.name) for f in fields(term_cfg)}
        guarded = SurfaceGuardedRMPFlowActionCfg(**values)
        guarded.class_type = SurfaceGuardedRMPFlowAction
        guarded.surface_z = surface_z
        guarded.contact_offset_m = contact_offset_m
        setattr(env_cfg.actions, term_name, guarded)
        swapped.append(term_name)
    if swapped:
        print(f"[surface guard] {', '.join(swapped)}: floor at {surface_z + contact_offset_m:.4f} m")
