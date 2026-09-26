# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report same-version qualification independently from cumulative development progress."""

import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("--min-trials", type=int, default=20)
parser.add_argument("--min-success-rate", type=float, default=0.95)
args = parser.parse_args()
tasks = json.loads(Path(__file__).with_name("tasks.json").read_text())["tasks"]
rows = []
for task in tasks:
    folder = args.root / task["task_id"]
    result_path = folder / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else []
    integrity = folder / "source_integrity.json"
    frozen = integrity.exists() and json.loads(integrity.read_text())["unchanged"]
    successes = sum(bool(r["success"]) for r in results)
    rate = successes / len(results) if results else 0.0
    forbidden = sum(bool(r.get("forbidden_contacts")) for r in results)
    execution_guard_failures = sum(
        r.get("validation_stage") == "execution"
        and any(token in r.get("error", "") for token in ("clearance", "self_collision", "forbidden_contact"))
        for r in results
    )
    positions = []
    for result in results:
        positions.extend(p[:2] for name, p in result.get("initial", {}).items() if not name.startswith("marker"))
    span_x = max((p[0] for p in positions), default=0) - min((p[0] for p in positions), default=0)
    span_y = max((p[1] for p in positions), default=0) - min((p[1] for p in positions), default=0)
    # Layout-dependent coverage and actual image/state agreement require review.
    review_path = folder / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    numerical = (
        frozen
        and len(results) >= args.min_trials
        and rate >= args.min_success_rate
        and forbidden == 0
        and execution_guard_failures == 0
    )
    reviewed = review.get("coverage_pass", False) and review.get("camera_replay_pass", False)
    rows.append({
        "task_id": task["task_id"],
        "任务": task["name"],
        "已记录回合": len(results),
        "成功回合": successes,
        "成功率": round(rate, 4),
        "代码保持一致": frozen,
        "发生禁止接触的回合": forbidden,
        "触发执行安全检查的回合": execution_guard_failures,
        "全部实例中心X跨度m": round(span_x, 4),
        "全部实例中心Y跨度m": round(span_y, 4),
        "随机覆盖复核": review.get("coverage_pass", False),
        "三相机重放复核": review.get("camera_replay_pass", False),
        "状态": "通过" if numerical and reviewed else "待复核" if numerical else "未通过/未完成",
        "证据": str(result_path),
    })
args.root.mkdir(parents=True, exist_ok=True)
with (args.root / "qualification.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# pine_wm 同版本稳定性验收",
    "",
    f"门槛：每任务至少 {args.min_trials} 回合、成功率 ≥ {args.min_success_rate:.0%}、无禁止执行接触。",
    "布局覆盖和三路相机同步必须另行复核；物体总体跨度不能代表每个角色覆盖充分。",
    "源码一致性来自运行前后校验；未完成任务、失败和缺少复核均不标为通过。",
    "",
    "| ID | 任务 | 成功/尝试 | 同版本 | 禁止接触回合 | 验收 |",
    "|---|---|---|---|---|---|",
]
for row in rows:
    lines.append(
        f"| {row['task_id']} | {row['任务']} | {row['成功回合']}/{row['已记录回合']} |"
        f" {row['代码保持一致']} | {row['发生禁止接触的回合']} | {row['状态']} |"
    )
(args.root / "qualification.md").write_text("\n".join(lines) + "\n")
print(args.root / "qualification.md")
