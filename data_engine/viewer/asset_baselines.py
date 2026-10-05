# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Review page for the cross-source asset baseline set (Lightwheel, Infinigen, USDCraft).

Reads the manifest written by ``tools/asset_baselines/build_test_assets.py`` (and the runtime fields
added by ``render_assets.py``). The page only displays recorded evidence: static estimates stay
labelled as estimates and a missing runtime result is shown as not run, never as a pass.
"""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "outputs" / "asset_baselines" / "test_assets_v0"
SOURCES = {"lightwheel": "Lightwheel", "infinigen": "Infinigen", "usdcraft": "USDCraft v2"}
CATEGORY_ORDER = ["microwave", "toaster_oven", "toaster", "drawer_cabinet", "cabinet_door"]
DERIVATION = {
    "declared": ("资产声明", "ok"),
    "name_rule": ("名称规则推断", "warn"),
    "unresolved": ("无法确定", "bad"),
}
STEP_LABEL = {
    "scale": "缩放",
    "bind_joint": "绑定关节",
    "open_direction": "确认开合方向",
    "handle": "标注把手",
    "colliders": "补碰撞体",
    "mass": "补质量",
    "fix_topology": "修复网格拓扑",
}


def load(root=ROOT):
    path = root / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None


def files(root=ROOT):
    """Expose only render images listed in the manifest and located under the baseline folder."""
    manifest = load(root) or {}
    allowed = {}
    for entry in manifest.get("entries", []):
        for relative in (entry.get("runtime") or {}).get("renders", []):
            candidate = (root / relative).resolve()
            if candidate.is_relative_to(root.resolve()) and candidate.is_file():
                allowed[f"/asset-baselines/{relative}"] = candidate
    return allowed


def _summary(entries):
    rows = []
    for source, label in SOURCES.items():
        items = [e for e in entries if e["source"] == source]
        if not items:
            continue
        count = len(items)

        def share(predicate, items=items, count=count):
            hits = sum(1 for e in items if predicate(e))
            return f"{hits}/{count}"

        loaded = [e for e in items if (e.get("runtime") or {}).get("isaac_load") is not None]
        rows.append(
            dict(
                source=label,
                count=count,
                steps=sum(len(e["manual_steps_estimate"]) for e in items) / count,
                declared=share(lambda e: e["binding"]["derivation"] == "declared"),
                direction=share(lambda e: e["binding"]["open_direction"] != "unknown"),
                handle=share(lambda e: e["binding"]["handle"] != "none"),
                colliders=share(lambda e: e["static"]["colliders"] > 0),
                mass=share(
                    lambda e: e["static"]["mass_explicit"]
                    + e["static"]["density_only"]
                    + e["static"]["material_density"]
                    > 0
                ),
                isaac=(
                    f"{sum(1 for e in loaded if e['runtime']['isaac_load'] == 'ok')}/{len(loaded)}"
                    if loaded
                    else "未运行"
                ),
            )
        )
    return rows


def _joint(entry):
    binding = entry["binding"]
    if binding["derivation"] == "declared":
        unit = binding.get("unit") or ""
        return (
            f"{html.escape(str(binding['joint']))} · 交互 <code>{html.escape(str(binding.get('interaction')))}</code>"
            f"（{html.escape(str(binding.get('kind')))}）<br>限位 {binding.get('limits')} {unit}，"
            f"起点 {binding.get('start')} → 目标 {binding.get('target')}"
        )
    if binding["joint"] is None:
        return "—"
    note = "（命中多个）" if binding.get("ambiguous") else ""
    return (
        f"{html.escape(str(binding['joint']))}{note}<br>限位 {binding.get('lower')} … {binding.get('upper')}（USD"
        " 单位，开合方向未声明）"
    )


def _physics(entry):
    static = entry["static"]
    target = next((j for j in static["joints"] if j["name"] == entry["binding"].get("joint")), None)
    parts = [
        f"刚体 {static['rigid_bodies']}",
        f"碰撞体 {static['colliders']} {html.escape(json.dumps(static['collider_approximation'], ensure_ascii=False))}",
        "质量 "
        + (
            "显式"
            if static["mass_explicit"]
            else ("密度" if static["density_only"] or static["material_density"] else "<b class='bad'>缺失</b>")
        ),
    ]
    if target:
        drive = [f"{k}={v}" for k, v in target.items() if k.startswith("drive_") or "riction" in k]
        if drive:
            parts.append("目标关节 " + html.escape(", ".join(drive)))
    return "<br>".join(parts)


def _card(entry):
    runtime = entry.get("runtime") or {}
    renders = runtime.get("renders", [])
    images = (
        "".join(f"<img loading='lazy' src='/asset-baselines/{html.escape(r)}' alt=''>" for r in renders)
        or "<div class='placeholder'>尚未渲染</div>"
    )
    label, tone = DERIVATION[entry["binding"]["derivation"]]
    steps = entry["manual_steps_estimate"]
    step_html = (
        "".join(f"<li><b>{STEP_LABEL.get(s['step'], s['step'])}</b>：{html.escape(s['reason'])}</li>" for s in steps)
        or "<li class='ok'>静态检查未发现需要人工补齐的项</li>"
    )
    origin = entry.get("origin", {})
    if entry["source"] == "usdcraft":
        origin_html = (
            f"文本输入：{html.escape(origin.get('prompt', ''))}<br><code>{html.escape(origin.get('record_id', ''))}</code>"
        )
    elif entry["source"] == "infinigen":
        proxy = "（落地烤箱作为台面烤箱代理）" if origin.get("proxy") else ""
        origin_html = f"生成器 {html.escape(str(origin.get('generator')))}，seed {origin.get('seed')}{proxy}"
    else:
        origin_html = html.escape(origin.get("pick", ""))
    load_state = runtime.get("isaac_load")
    load_html = {
        None: "<span class='muted'>Isaac 加载：未运行</span>",
        "ok": "<span class='ok'>Isaac 加载：成功</span>",
    }.get(load_state, f"<span class='bad'>Isaac 加载：{html.escape(str(load_state))}</span>")
    ingest = entry.get("ingest") or {}
    rewrite_html = (
        "<br><span class='warn'>接入改写："
        + html.escape("；".join(ingest.get("rewrites", [])))
        + "（渲染使用改写后的副本）</span>"
        if ingest
        else ""
    )
    note = {
        "geometry_not_visible": "<br><span class='bad'>渲染：几何不可见</span>",
        "authored_materials_not_rendered_in_isaac_rtx": (
            "<br><span class='warn'>渲染：原材质在 Isaac RTX 中不可见，图为统一灰色材质</span>"
        ),
    }.get(runtime.get("render_note"), "")
    size = entry["static"]["size_m"]
    return f"""
<article class="card">
  <div class="thumbs">{images}</div>
  <header><span class="tag {entry['source']}">{SOURCES[entry['source']]}</span>
    <h3>{html.escape(entry['asset_id'])}</h3></header>
  <dl>
    <dt>尺寸 (m)</dt><dd>{size}</dd>
    <dt>目标关节</dt><dd><span class="pill {tone}">{label}</span><br>{_joint(entry)}</dd>
    <dt>把手</dt><dd>{'已声明 ' + html.escape(str(entry['binding'].get('handle_point'))) if entry['binding']['handle'] == 'declared' else '<span class="warn">未声明</span>'}</dd>
    <dt>物理</dt><dd>{_physics(entry)}</dd>
    <dt>来源</dt><dd>{origin_html}<br>{load_html}{note}{rewrite_html}</dd>
  </dl>
  <h4>静态估计的人工步骤（{len(steps)}）</h4><ul>{step_html}</ul>
  <details><summary>文件与身份</summary><code>{html.escape(entry['usd'])}</code><br>依赖闭包 {entry['static']['closure_files']} 个文件，
    sha256 {entry['static']['closure_sha256'][:16]}…，未解析引用 {len(entry['static']['unresolved'])}；许可 {html.escape(entry['license'])}</details>
</article>"""


def page(root=ROOT):
    manifest = load(root)
    if manifest is None:
        return "<!doctype html><meta charset='utf-8'><p>尚未生成 test_assets_v0 清单。</p>".encode()
    entries = manifest["entries"]
    summary = "".join(
        f"<tr><td>{r['source']}</td><td>{r['count']}</td><td>{r['steps']:.2f}</td><td>{r['declared']}</td><td>{r['direction']}</td>"
        f"<td>{r['handle']}</td><td>{r['colliders']}</td><td>{r['mass']}</td><td>{r['isaac']}</td></tr>"
        for r in _summary(entries)
    )
    sections = []
    for category in CATEGORY_ORDER:
        items = [e for e in entries if e["category"] == category]
        task = manifest["tasks"][category]
        columns = []
        for source, label in SOURCES.items():
            cards = (
                "".join(_card(e) for e in items if e["source"] == source)
                or "<div class='placeholder'>该来源没有此品类</div>"
            )
            columns.append(f"<div class='column'><h4 class='colhead'>{label}</h4>{cards}</div>")
        sections.append(
            f"<section><h2>{html.escape(task['label'])} <small>任务 <code>{task['task']}</code> · 目标关节类型"
            f" {task['joint']} · 成功判据：{html.escape(task['success'])}</small></h2><div"
            f" class='grid'>{''.join(columns)}</div></section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>资产基线对比 · test_assets_v0</title><link rel="stylesheet" href="/style.css">
<style>
body{{overflow:auto}} main.baseline{{padding:20px 28px;max-width:1800px}}
.baseline h2{{margin:28px 0 10px;font-size:20px}} .baseline h2 small{{display:block;color:var(--muted);font-weight:400;font-size:13px;margin-top:4px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}} .colhead{{margin:0 0 8px;color:var(--cyan)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px;font-size:13px}}
.card h3{{display:inline;font-size:14px;margin-left:6px}} .card dl{{display:grid;grid-template-columns:72px 1fr;gap:4px 8px;margin:10px 0}}
.card dt{{color:var(--muted)}} .card dd{{margin:0;word-break:break-word}} .card h4{{margin:8px 0 4px;font-size:13px}} .card ul{{margin:0;padding-left:18px}}
.thumbs{{display:grid;grid-template-columns:1fr 1fr;gap:4px}} .thumbs img{{width:100%;border-radius:6px;background:#000}}
.placeholder{{color:var(--muted);border:1px dashed var(--line);border-radius:6px;padding:16px;text-align:center}}
.tag{{font-size:11px;padding:2px 6px;border-radius:4px;background:#24364b}} .tag.usdcraft{{background:#1d4a44}}
.pill{{font-size:11px;padding:1px 6px;border-radius:999px;border:1px solid}} .ok{{color:#6ee7a8}} .warn{{color:#f5c26b}} .bad{{color:#ff7b7b}} .muted{{color:var(--muted)}}
table.sum{{border-collapse:collapse;margin:10px 0}} table.sum td,table.sum th{{border:1px solid var(--line);padding:6px 10px;text-align:center}}
.note{{color:var(--muted);max-width:1100px;line-height:1.6}} code{{font-size:12px}}
</style></head><body>
<header><div><span class="eyebrow">DATA ENGINE · ASSET BASELINES</span><h1>资产基线对比 · test_assets_v0</h1></div>
<a href="/" style="color:var(--cyan)">← 任务审核台</a> <a href="/api/asset-baselines" style="color:var(--cyan)">清单 JSON</a></header>
<main class="baseline">
<p class="note">对比 Lightwheel、Infinigen 与 USDCraft v2 在同一组品类和任务上的资产。每个资产列出：任务的目标关节以及它是怎么确定的（资产声明 / 名称规则 / 无法确定）、开合方向和把手是否由资产声明、物理配置，以及<b>静态估计</b>的人工步骤。
人工步骤是“零人工接入率”这一指标的静态起点；最终数字以 data engine 在 Isaac Sim 中的运行结果为准。仿真器统一为 Isaac Sim。</p>
<table class="sum"><tr><th>来源</th><th>资产数</th><th>平均人工步骤（静态估计）</th><th>目标关节由声明确定</th><th>开合方向已知</th><th>把手已声明</th><th>有碰撞体</th><th>有质量/密度</th><th>Isaac 加载成功</th></tr>{summary}</table>
{''.join(sections)}
</main></body></html>""".encode()


def api(root=ROOT):
    return json.dumps(load(root) or {}, ensure_ascii=False).encode()
