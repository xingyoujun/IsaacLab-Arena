# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Aggregate development rounds without presenting mixed-version successes as qualification."""

import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
args = parser.parse_args()
tasks = json.loads((Path(__file__).resolve().parents[3] / "data_engine/pine_wm/tasks.json").read_text())["tasks"]
rows = []
for task in tasks:
    attempts = []
    latest_path = None
    latest = []
    best = None
    for directory in sorted(args.root.glob("round*")):
        path = directory / task["task_id"] / "results.json"
        if not path.exists():
            continue
        results = json.loads(path.read_text())
        attempts.extend(results)
        latest_path, latest = path, results
        if any(r["success"] for r in results):
            best = path
    errors = sorted({r.get("error") or "success_predicate_false" for r in latest if not r["success"]})
    rows.append({
        "task_id": task["task_id"],
        "任务": task["name"],
        "已有开发回合通过": bool(best),
        "累计调试成功": sum(r["success"] for r in attempts),
        "累计已记录尝试": len(attempts),
        "最近一轮成功": sum(r["success"] for r in latest),
        "最近一轮已记录": len(latest),
        "最近失败": "; ".join(errors),
        "稳定性验收": "未完成",
        "成功证据": str(best) if best else "",
        "最近证据": str(latest_path) if latest_path else "",
    })
with (args.root / "progress.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
passed = sum(r["已有开发回合通过"] for r in rows)
lines = [
    "# pine_wm 首批 20 项任务开发测试进度",
    "",
    f"{passed}/20 项曾取得开发回合成功记录；这不表示 {passed} 项已经通过稳定性验收。",
    "",
    "各轮可能使用不同代码。累计数只用于追踪调试，不用于计算最终同版本成功率。",
    "表中只计已写出的结果；正在运行或报告写入失败的进程需查看对应 runtime.log。",
    "",
    "| ID | 任务 | 曾通过 | 累计成功/尝试 | 最近成功/已记录 | 最近失败 | 稳定性验收 |",
    "|---|---|---|---|---|---|---|",
]
for row in rows:
    evidence = f"[是]({row['成功证据']})" if row["已有开发回合通过"] else "否"
    lines.append(
        f"| {row['task_id']} | {row['任务']} | {evidence} | {row['累计调试成功']}/{row['累计已记录尝试']} |"
        f" {row['最近一轮成功']}/{row['最近一轮已记录']} | {row['最近失败']} | 未完成 |"
    )
(args.root / "progress.md").write_text("\n".join(lines) + "\n")
print(args.root / "progress.md")
