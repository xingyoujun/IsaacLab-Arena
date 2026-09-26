# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independently measure held-object stability from the last hold frame through release."""

import numpy as np
from scipy.spatial.transform import Rotation


def audit_carry(trace, tool, side):
    """Return verified closed-hand carry metrics through the start of release.

    Args:
        trace: Recorded state arrays, including phases, actions, objects, and both TCPs.
        tool: Name of the carried object in the trace.
        side: Active arm, either "left" or "right".
    """
    selected = trace["tool"] == tool if "tool" in trace else np.ones(len(trace["phase"]), dtype=bool)
    hold = np.flatnonzero(selected & (trace["phase"] == "hold"))
    release = np.flatnonzero(selected & (trace["phase"] == "release"))
    assert len(hold) and len(release) and release[0] > hold[-1], "Missing ordered hold/release phases"
    frames = np.arange(hold[-1], release[0])
    assert selected[frames].all(), "Tool changed during carry"
    arm = 0 if side == "right" else 1
    assert np.all(trace["actions"][frames, 7 if arm == 0 else 15] == -1), "Gripper opened during carry"
    index = list(trace["object_names"]).index(tool)
    positions = trace["object_positions"][frames, index]
    rotations = Rotation.from_quat(trace["object_quaternions"][frames, index])
    tcp = trace["tcp_positions"][frames, arm * 3 : arm * 3 + 3]
    tcp_rotation = Rotation.from_quat(trace["tcp_quaternions"][frames, arm])
    relative = tcp_rotation.inv().apply(positions - tcp)
    relative_rotation = tcp_rotation.inv() * rotations
    drift = float(np.linalg.norm(relative - relative[0], axis=1).max())
    angle = float(np.degrees((relative_rotation[0].inv() * relative_rotation).magnitude()).max())
    assert drift < 0.005 and angle < 5, f"Held-object slip: {drift} m / {angle} deg"
    tilt = np.degrees(np.arccos(np.clip(-tcp_rotation.apply([0, 0, 1])[:, 2], -1, 1)))
    return dict(
        audited_frames=len(frames),
        relative_drift_m=drift,
        relative_angle_deg=angle,
        maximum_tcp_tilt_deg=float(tilt.max()),
        passed=True,
    )
