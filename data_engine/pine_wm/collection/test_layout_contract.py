# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regress the camera-backside layout and full opening sweep clipping."""

import json
from pathlib import Path
from scipy.spatial.transform import Rotation

from data_engine.pine_wm.collection.layout_contract import check_drawer_layout


def test_drawer_training_view_contract():
    root = Path(__file__).resolve().parents[3]
    item = json.loads((root / "data_engine/pine_wm/review_layouts.json").read_text())["tasks"]["T041"]
    bounds = [[-0.14, -0.186, 0], [0.14, 0.13, 0.18]]

    def check(x, y, yaw):
        pose = [x, y, 0.74, *Rotation.from_euler("z", yaw, degrees=True).as_quat()]
        return check_drawer_layout(item, pose, bounds, 0.10)

    old = check(0.082177, -0.020179, 49.53067)
    assert "drawer_front_not_facing_training_camera" in old["failures"]
    rotated_only = check(0.082177, -0.020179, -80.834)
    assert "drawer_full_sweep_outside_training_image" in rotated_only["failures"]
    assert check(*item["positions_xy_m"]["drawer"], item["drawer_yaw_deg"])["status"] == "pass"
