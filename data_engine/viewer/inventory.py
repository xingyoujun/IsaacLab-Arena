# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Current-asset Pine collection coverage without promoting historical previews."""

import hashlib
import html
from pathlib import Path

from data_engine.harness.catalog import scenarios

BUNDLE = "USDCraft-Scene-v2-20260930"
STATES = {
    "pending": "尚未运行",
    "planned": "等待执行",
    "collecting": "正在执行",
    "raw_sealed": "正在审计",
    "preview_validated": "预览通过",
    "collection_failed": "执行失败",
    "audit_failed": "审计未通过",
    "cancelled": "已中断",
}


def inventory(records):
    """Match attempts to the current manifest hash, independently of older successes."""
    manifest = Path(__file__).resolve().parents[2] / "local_assets" / BUNDLE / "manifest.json"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None
    rows = []
    for scenario in scenarios().values():
        if scenario["scene"] != "pine_wm":
            continue
        attempts = [r for r in records if r["scenario"] == scenario["id"]]
        matching = [r for r in attempts if manifest_hash and r.get("asset_manifest_sha256") == manifest_hash]
        latest = max(matching, key=lambda r: (r["created_at"], r["name"]), default=None)
        rows.append(
            dict(
                id=scenario["id"],
                title=scenario["instruction"],
                family=scenario["adapter"],
                preview_supported=scenario["preview_supported"],
                annotation_supported=scenario["annotation_supported"],
                state=latest["state"] if latest else "pending",
                run=latest["name"] if latest else None,
                failure=(latest.get("failure") or (latest.get("events") or [{}])[-1].get("error")) if latest else None,
                cameras=len(latest["videos"]) if latest else 0,
                attempts=len(matching),
                historical_attempts=len(attempts) - len(matching),
            )
        )
    return dict(bundle=BUNDLE, asset_manifest_sha256=manifest_hash, tasks=rows)


def page(records):
    """Render task coverage, including pending or failed tasks, with truthful asset provenance."""
    data = inventory(records)
    rows = data["tasks"]
    passed = sum(row["state"] == "preview_validated" for row in rows)
    body = []
    for row in rows:
        task = html.escape(row["id"].split("/")[-1])
        link = f'<a href="/#{html.escape(row["run"], quote=True)}">查看本次结果 →</a>' if row["run"] else "等待新预览"
        failure = f'<br><small>{html.escape(str(row["failure"]))}</small>' if row["failure"] else ""
        state = html.escape(STATES.get(row["state"], row["state"]))
        body.append(
            f'<tr><td>{task}</td><td>{html.escape(row["title"])}</td><td>{state}{failure}</td>'
            f'<td>{row["cameras"]} 路</td><td>{"已接入" if row["annotation_supported"] else "尚未接入"}</td>'
            f"<td>{link}</td></tr>"
        )
    content = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="15">
<title>Pine WM · 新资产任务同步</title><link rel="stylesheet" href="/style.css"><link rel="stylesheet" href="/review.css">
</head><body><header><div><span class="eyebrow">DATA ENGINE · PINE WM</span><h1>新资产任务同步</h1></div>
<a href="/">返回任务审核台 →</a></header><main class="review"><section class="review-intro">
<h2>20 个 Pine 任务 + 2 个联动任务</h2>
<p>已有 cuMotion 执行与录制流程的任务共 22 个。批量采集资格仍未授予；这里逐项验证新版资产的单次预览。</p>
<p>当前资产：<strong>__BUNDLE__</strong> · 最新尝试通过 <strong>__PASSED__ / 22</strong>。</p>
<p>每 15 秒更新。按资产 manifest hash 匹配，只显示新版资产的最新尝试；失败与待运行不由旧版成功替代。
18 个旧任务尚无完整 skill 标注，已有视频和状态检查不自动补全该能力。</p></section>
<div class="table-scroll"><table><thead><tr><th>任务</th><th>内容</th><th>新版资产状态</th><th>有效视频</th>
<th>Skill 标注</th><th>证据</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p><a href="/reviews/usdcraft-v2-20260930">资产审计、联动曲线与问题归因</a> · <a href="/api/pine-tasks">机器可读同步状态</a></p>
<p><a href="/review-assets/pine_sync_report.md">本轮完整报告</a> · <a href="/review-assets/pine_sync_summary.json">20 项运行结果</a> · <a href="/review-assets/pine_sync_verification.json">封存核验</a> · <a href="/review-assets/pine_sync_source.tar.gz">执行源码快照</a> · <a href="/review-assets/pine_sync_source_manifest.json">源码哈希</a></p>
<p class="muted">所有预览仍是 post-step 视频。随机布局稳定性、训练图像对齐与批量数据资格需要单独验证。</p>
</main></body></html>"""
    return (
        content.replace("__BUNDLE__", BUNDLE)
        .replace("__PASSED__", str(passed))
        .replace("__ROWS__", "".join(body))
        .encode()
    )
