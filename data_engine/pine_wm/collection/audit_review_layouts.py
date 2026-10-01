# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Audit layout drafts against fixed-camera frusta without starting a simulator."""

import csv
import itertools
import json
import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation

from data_engine.pine_wm.collection.review_layout import sample_review_layout

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("ARENA_PINE_WM_EXPERIMENT_ROOT", ROOT / "outputs/pine_wm/first20"))


def project(points, position, rotation, intrinsics):
    """Project world points into an OpenCV camera; return pixels and front-facing status."""
    camera = (np.asarray(points) - position) @ rotation
    fx, fy, cx, cy = intrinsics
    return camera[:, :2] / camera[:, 2:] * [fx, fy] + [cx, cy], bool(np.all(camera[:, 2] > 0))


def main():
    config = json.loads((Path(__file__).resolve().parents[3] / "data_engine/pine_wm/review_layouts.json").read_text())
    assets = {
        a["asset_id"]: np.array([a["bounds_min_m"], a["bounds_max_m"]])
        for a in json.loads((BASE / "asset_audit.json").read_text())
    }
    specs = json.loads((Path(__file__).resolve().parents[3] / "data_engine/pine_wm/tasks.json").read_text())["tasks"]
    calibration = json.loads((ROOT / "isaaclab_arena/embodiments/ur7e/calibration/pine_wm/d435.json").read_text())
    T_W_C = np.array(calibration["T_base_cam"])
    T_W_C[:3, 3] += [0, -0.425, 0.75]
    intr = calibration["intrinsics"]
    overview_pos = np.array([1.7, -1.5, 1.7])
    forward = np.array([0, 0, 0.8]) - overview_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    cameras = {
        "realsense_d435": (T_W_C[:3, 3], T_W_C[:3, :3], [intr[k] for k in ["fx", "fy", "ppx", "ppy"]], [640, 480]),
        "scene_cam": (
            overview_pos,
            np.column_stack([right, down, forward]),
            [1280 * 18 / 20.955] * 2 + [640, 360],
            [1280, 720],
        ),
    }
    audit = []
    rows = []
    for spec in specs:
        tid = spec["task_id"]
        item = config["tasks"][tid]
        bindings = {a.removeprefix("P20_"): a for a in spec["asset_bindings"]}
        if tid == "T016":
            bindings = {"cube_40": "P20_cube_40", "landmark_a": "P20_cylinder", "landmark_b": "P20_cylinder"}
        if tid == "T017":
            bindings = {"cube_40": "P20_cube_40", "box_a": "P20_box", "box_b": "P20_box"}
        if tid == "T142":
            bindings = {
                "cube_40": "P20_cube_40",
                "cylinder": "P20_cylinder",
                "sphere": "P20_sphere",
                **{f"tray_{i}": "P20_tray" for i in range(3)},
            }
        if tid == "T145":
            bindings = {**{f"cube_{i}": "P20_cube_40" for i in range(6)}, "box": "P20_box"}
        bounds = {n: assets[a] for n, a in bindings.items()}
        for name, size in item.get("asset_sizes_mm", {}).items():
            bounds[name] = bounds[name] * (np.array(size) / 1000 / (bounds[name][1] - bounds[name][0]))
        entry = {
            "task": tid,
            "name": spec["name"],
            "status": "static_projection_only",
            "notes": item["notes"],
            "projection": {},
        }
        clipped = []
        if not item["asset_revision_required"]:
            poses, goals = sample_review_layout(tid, list(bindings), bounds, 5)
            points = {}
            for name, pose in poses.items():
                corners = np.array(list(itertools.product(*zip(*bounds[name]))))
                points[name] = Rotation.from_quat(pose[3:]).apply(corners) + pose[:3]
            if tid == "T031":
                points["completed_tower"] = np.array([
                    [goals["target"][0] + x, goals["target"][1] + y, z]
                    for x, y, z in itertools.product([-0.03, 0.03], [-0.03, 0.03], [0.74, 0.947])
                ])
            if tid == "T143":
                points["completed_row"] = np.array([
                    [goals["target"][0] + (i - 2) * goals["row_pitch_m"] + x, goals["target"][1] + y, z]
                    for i in range(5)
                    for x, y, z in itertools.product([-0.03, 0.03], [-0.03, 0.03], [0.74, 0.80])
                ])
            if "drawer" in poses:
                swept = bounds["drawer"].copy()
                swept[0, 1] -= 0.15
                points["drawer_full_sweep"] = (
                    Rotation.from_quat(poses["drawer"][3:]).apply(np.array(list(itertools.product(*zip(*swept)))))
                    + poses["drawer"][:3]
                )
            for label, (pos, rot, ks, resolution) in cameras.items():
                entry["projection"][label] = {}
                for name, corners in points.items():
                    pixels, front = project(corners, pos, rot, ks)
                    inside = front and bool(np.all(pixels >= 12) and np.all(pixels <= np.array(resolution) - 12))
                    entry["projection"][label][name] = {
                        "inside_12px_margin": inside,
                        "pixel_min": pixels.min(axis=0).tolist(),
                        "pixel_max": pixels.max(axis=0).tolist(),
                    }
                    if not inside:
                        clipped.append(f"{label}:{name}")
        else:
            entry["status"] = "asset_revision_required_before_preview"
        entry["limitation"] = (
            "Frustum only; no occlusion, robot reachability, collision or wrist camera visibility claim."
        )
        audit.append(entry)
        rows.append({
            "task_id": tid,
            "任务": spec["name"],
            "状态": (
                "单次成功，待用户审核" if item.get("status") == "preview_success_awaiting_user_review" else "布局待审核"
            ),
            "目标XY米": str(item["target_xy_m"]),
            "源物体和夹具XY米": json.dumps(item["positions_xy_m"], ensure_ascii=False),
            "固定相机边界风险": "; ".join(clipped) or (
                "未投影：先调整资产" if item["asset_revision_required"] else "角点通过，仍需遮挡审核"
            ),
            "调整说明": item["notes"],
        })
    folder = ROOT / "docs/pine_wm_review"
    folder.mkdir(exist_ok=True)
    (folder / "projection_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    with (folder / "tasks.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Pine WM 全部 20 项任务布局审核草案",
        "",
        (
            "2026-09-26：全部 20 项新布局均已有一次成功预览，80 段四视角视频已上线。执行已停止，"
            "等待用户审核；稳定性测试和采集仍关闭。"
        ),
        (
            "世界坐标 XY 单位为米；桌面 Z=0.74 m。核心目标区 X∈[-0.10,0.10]、Y∈[-0.10,0.06]。外围夹具需按完整包络审核，"
            "不能仅以中心落在桌上作为合格依据。"
        ),
        "原有相机标定保持不变。固定相机投影检查含 12 像素边距；不包含机械臂遮挡、可达性、碰撞或动态腕部相机可见性。",
        "T031 新塔顶也参与检查；抽屉检查闭合包络和完整 150 mm 抽拉包络。投影风险意味着仍需调整，不能直接批准。",
        "",
        "|任务|目标 XY|调整|投影/资产风险|",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"|{row['task_id']} {row['任务']}|{row['目标XY米']}|{row['调整说明']}|{row['固定相机边界风险']}|")
    lines += [
        "",
        "## 审核顺序",
        "",
        "先审核 T031 的中心叠塔布局，再审核 T041/T044 抽屉与托盘；之后逐任务处理其他布局。",
        (
            "每次仅启动一个明确授权的预览任务；一次成功后立即停止。用户确认布局、尺寸、视角及随机范围后，"
            "才能开放该任务的稳定性测试；稳定性通过后再采集。"
        ),
        "当前仅开放预览。容器按逐任务配置在场景实例上缩放，源 USD 不改写；新布局仍待用户视角和任务效果审核。",
    ]
    (folder / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
