# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate measured object clearance and grasp stability."""

import numpy as np
from scipy.spatial.transform import Rotation


def evaluate_hold(initial, samples):
    """Require actual object lift and a stable object-to-TCP transform throughout the hold."""
    assert samples, "Missing hold samples"
    positions = np.array([x["relative_position"] for x in samples])
    rotations = Rotation.from_quat([x["relative_quaternion_xyzw"] for x in samples])
    drift = float(np.linalg.norm(positions - positions[0], axis=1).max())
    angle = float(np.degrees((rotations[0].inv() * rotations).magnitude()).max())
    clearance = min(x["minimum_z_m"] for x in samples)
    height_gain = samples[-1]["position"][2] - initial["position"][2]
    metrics = dict(
        relative_drift_m=drift,
        relative_rotation_drift_deg=angle,
        minimum_clearance_m=clearance,
        height_gain_m=height_gain,
    )
    metrics["passed"] = bool(height_gain > 0.07 and clearance > 0.05 and drift < 0.005 and angle < 5)
    return metrics
