# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject drawer layouts that hide task direction or clip the full opening envelope."""

import itertools
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def check_drawer_layout(item, pose, bounds, target_opening):
    """Check fixed-camera geometry; physical reachability and occlusion need runtime review."""
    contract = item.get("layout_contract")
    if contract is None:
        return {"status": "unknown", "reason": "No explicit layout contract"}
    assert contract["training_cameras"] == ["realsense_d435"], "Unsupported calibrated camera contract"
    calibration_path = (
        Path(__file__).resolve().parents[3] / "isaaclab_arena/embodiments/ur7e/calibration/pine_wm/d435.json"
    )
    calibration = json.loads(calibration_path.read_text())
    T_W_C = np.array(calibration["T_base_cam"])
    T_W_C[:3, 3] += [0, -0.425, 0.75]
    rotation = Rotation.from_quat(pose[3:])
    position = np.array(pose[:3])
    direction = T_W_C[:3, 3] - position
    direction[2] = 0
    direction /= np.linalg.norm(direction)
    front = rotation.apply(contract["front_local"])
    front[2] = 0
    front /= np.linalg.norm(front)
    angle = float(np.rad2deg(np.arccos(np.clip(front @ direction, -1, 1))))
    sweep = np.array(bounds, dtype=float).copy()
    sweep[0, 1] -= item["drawer_stroke_m"]
    corners = np.array(list(itertools.product(*zip(*sweep))))
    world = rotation.apply(corners) + position
    camera = (world - T_W_C[:3, 3]) @ T_W_C[:3, :3]
    intr = calibration["intrinsics"]
    pixels = camera[:, :2] / camera[:, 2:] * [intr["fx"], intr["fy"]] + [intr["ppx"], intr["ppy"]]
    margin = contract["image_margin_px"]
    projected = bool(np.all(camera[:, 2] > 0) and np.all(pixels >= margin))
    projected = projected and bool(np.all(pixels <= np.array([640, 480]) - margin))
    local_handle = np.array(contract["handle_local_m"])
    handles = rotation.apply([local_handle, local_handle + np.array([0, -target_opening, 0])]) + position
    xmin, xmax, ymin, ymax = contract["operation_core_xy_m"]
    central = bool(np.all(handles[:, :2] >= [xmin, ymin]) and np.all(handles[:, :2] <= [xmax, ymax]))
    failures = []
    if angle > contract["max_front_angle_deg"]:
        failures.append("drawer_front_not_facing_training_camera")
    if not projected:
        failures.append("drawer_full_sweep_outside_training_image")
    if not central:
        failures.append("handle_operation_outside_core_region")
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "front_angle_deg": angle,
        "sweep_pixel_min": pixels.min(axis=0).tolist(),
        "sweep_pixel_max": pixels.max(axis=0).tolist(),
        "operation_handle_world_m": handles.tolist(),
        "scope": "Static projection/direction/core only; no occlusion, reachability or collision proof",
    }
