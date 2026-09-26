# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that carry auditing distinguishes wrist motion, slip, and deliberate release."""

import numpy as np
from scipy.spatial.transform import Rotation

import pytest

from isaaclab_arena_cumotion.g2_collection.workcell.carry_audit import audit_carry


def moving_hand(side):
    arm = 0 if side == "right" else 1
    tool = "finished_gear" if arm == 0 else "drill"
    rotation = Rotation.from_euler("y", np.array([0, 40, 80, 80, 80])[:, None], degrees=True) * Rotation.from_euler(
        "x", 180, degrees=True
    )
    positions = np.c_[np.arange(5) * 0.1, np.zeros(5), np.full(5, 0.3)]
    tcp = np.zeros((5, 6))
    tcp[:, arm * 3 : arm * 3 + 3] = positions
    quaternions = np.tile([0.0, 0, 0, 1], (5, 2, 1))
    quaternions[:, arm] = rotation.as_quat()
    objects = positions + rotation.apply([0.08, 0, 0])
    objects[3:] += 10  # Post-release movement is deliberately outside the carry window.
    actions = np.zeros((5, 16))
    actions[:3, 7 if arm == 0 else 15] = -1
    actions[3:, 7 if arm == 0 else 15] = 1
    return tool, dict(
        tool=np.array([tool] * 5),
        phase=np.array(["hold", "carry_to_bin", "lower_into_bin", "release", "verify_placement"]),
        actions=actions,
        object_names=np.array([tool]),
        object_positions=objects[:, None],
        object_quaternions=rotation.as_quat()[:, None],
        tcp_positions=tcp,
        tcp_quaternions=quaternions,
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_stable_rotating_hand_and_deliberate_release(side):
    tool, trace = moving_hand(side)
    result = audit_carry(trace, tool, side)
    assert result["passed"] and result["relative_drift_m"] < 1e-12
    assert result["maximum_tcp_tilt_deg"] == pytest.approx(80)


@pytest.mark.parametrize("fault", ["translation", "rotation", "opening"])
def test_reject_slip_or_premature_opening(fault):
    tool, trace = moving_hand("right")
    if fault == "translation":
        trace["object_positions"][1, 0, 0] += 0.006
    elif fault == "rotation":
        rotation = Rotation.from_quat(trace["object_quaternions"][1, 0])
        trace["object_quaternions"][1, 0] = (rotation * Rotation.from_euler("x", 6, degrees=True)).as_quat()
    else:
        trace["actions"][1, 7] = 1
    with pytest.raises(AssertionError):
        audit_carry(trace, tool, "right")
