# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write a reviewable CSV/Markdown qualification ledger from one immutable test round."""

import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("round", type=Path)
args = parser.parse_args()
tasks = json.loads((Path(__file__).resolve().parents[3] / "data_engine/pine_wm/tasks.json").read_text())["tasks"]
batch_path = args.round / "summary.json"
batch = {r["task_id"]: r for r in json.loads(batch_path.read_text())} if batch_path.exists() else {}
rows = []
for t in tasks:
    path = args.round / t["task_id"] / "results.json"
    results = json.loads(path.read_text()) if path.exists() else []
    successes = sum(r["success"] for r in results)
    startup_failure = not results and (path.parent / "runtime.log").exists()
    failures = [r.get("error") or "success_predicate_false" for r in results if not r["success"]]
    status = "未测试" if not results else ("单轮通过，待稳定性验收" if not failures else "需调试")
    if startup_failure:
        status = "运行失败，见 runtime.log" if t["task_id"] in batch else "待完成，尚无结果"
    rows.append({
        "task_id": t["task_id"],
        "任务": t["name"],
        "状态": status,
        "成功": successes,
        "已完成回合": len(results),
        "失败原因": "; ".join(sorted(set(failures))),
        "证据": str(path) if results else "",
    })
with (args.round / "status.csv").open("w", encoding="utf-8-sig", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# pine_wm 首批 20 项任务测试记录",
    "",
    "本表只汇总指定测试轮次。单轮成功不能证明稳定采集；调试数据不可直接并入训练集。",
    "",
    "| ID | 任务 | 状态 | 成功/完成 | 失败原因 |",
    "|---|---|---|---|---|",
]
for r in rows:
    lines.append(f"| {r['task_id']} | {r['任务']} | {r['状态']} | {r['成功']}/{r['已完成回合']} | {r['失败原因']} |")
(args.round / "status.md").write_text("\n".join(lines) + "\n")
print(args.round / "status.md")
