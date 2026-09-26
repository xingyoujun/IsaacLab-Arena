# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed central layout drafts for one-success user review; no random broad-table sampling."""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def sample_review_layout(task, names, bounds, stack_count):
    """Compose a draft without claiming camera occlusion or reachability validation."""
    cfg = json.loads(Path(__file__).with_name("review_layouts.json").read_text())
    item = cfg["tasks"][task]
    assert not item["asset_revision_required"], item["notes"]
    poses = {}
    yaw = np.deg2rad(item.get("drawer_yaw_deg", 0))
    goals = {
        "target": item["target_xy_m"],
        "yaw": float(yaw),
        "layout_version": cfg["version"],
        "layout_review_required": True,
        "allow_target_relocation": False,
    }
    for name in names:
        if task == "T044" and name == "cube_25":
            continue
        angle = np.deg2rad(item.get("yaw_deg", {}).get(name, 0))
        if name == "drawer":
            angle = yaw
        rotation = Rotation.from_euler("z", angle)
        if name == "bottle":
            rotation = Rotation.from_euler("y", np.pi / 2)
        corners = np.array(
            [[x, y, z] for x in bounds[name][:, 0] for y in bounds[name][:, 1] for z in bounds[name][:, 2]]
        )
        bottom = rotation.apply(corners)[:, 2].min()
        fixed = name in {
            "drawer",
            "button",
            "wall",
            "v_support",
            "coaster",
            "push_mat",
            "landmark_a",
            "landmark_b",
            "box_a",
            "box_b",
        }
        z = 0.74 - bottom + (0 if fixed else 0.002)
        if task == "T004" and name == "cylinder":
            z = 0.773
        if name == "mug":
            z = 0.746
        poses[name] = [*item["positions_xy_m"][name], float(z), *rotation.as_quat()]
    if task in {"T031", "T038", "T143"}:
        cubes = sorted([n for n in names if n.startswith("cube_")], key=lambda n: int(n.split("_")[1]))
        active = cubes[-stack_count:] if task != "T143" else cubes
        goals.update(active_cubes=active, stack_count=len(active))
        if task == "T038":
            height = 0.742
            for n in reversed(active):
                side = int(n.split("_")[1]) / 1000
                poses[n][:3] = [*item["tower_xy_m"], height + side / 2]
                height += side
    if "drawer" in names:
        dimensions = np.asarray(bounds["drawer"][1]) - bounds["drawer"][0]
        assert np.all(
            dimensions <= np.array(item["drawer_size_cap_mm"]) / 1000 + 1e-5
        ), "Drawer exceeds reviewed size cap"
        goals["initial_opening"] = 0.10 if task == "T042" else (0.15 if task in {"T044", "T045"} else 0)
        goals["target_opening"] = 0.06 if task == "T043" else (0.10 if task == "T041" else 0)
        if task == "T044":
            local = [0, -0.055 - goals["initial_opening"], 0]
            point = np.array(poses["drawer"][:3]) + Rotation.from_euler("z", yaw).apply(local)
            poses["cube_25"] = [*point[:2], 0.7895, *Rotation.from_euler("z", yaw).as_quat()]
    if task == "T017":
        goals["box_marker_offset_y_m"] = -float(bounds["box_a"][1, 1]) - 0.0006
        goals["box_marker_height_m"] = float(bounds["box_a"][1, 2]) * 0.55
    if task == "T143":
        goals["row_pitch_m"] = item["row_pitch_m"]
    if task == "T014":
        goals["target_yaw"] = float(np.deg2rad(item["target_yaw_deg"]))
    if "container_offsets_m" in item:
        goals["container_offsets_m"] = item["container_offsets_m"]
    if task == "T145":
        goals["count"] = item["count"]
    assert set(poses) == set(names), "Draft has missing asset placements"
    return poses, goals
