# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Review page for sim2sim cases: what the system may see, what it produced, how the target scores it.

Reads outputs/sim2sim/<case>/manifest.json written by tools/sim2sim/build_make_toast_case.py and only
displays recorded evidence; unfinished stages are shown as not done, never as results.
"""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "outputs" / "sim2sim" / "make_toast_v0"


def load(root=ROOT):
    path = root / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None


def files(root=ROOT):
    """Serve only files under the case folder that the manifest references."""
    manifest = load(root) or {}
    refs = []

    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, str) and value.endswith((".png", ".mp4")):
            refs.append(value)

    walk(manifest)
    allowed = {}
    for relative in refs:
        candidate = (root / relative).resolve()
        if candidate.is_relative_to(root.resolve()) and candidate.is_file():
            allowed[f"/sim2sim/{relative}"] = candidate
    return allowed


def _img(rel, cls="thumb"):
    return f"<img class='{cls}' loading='lazy' src='/sim2sim/{html.escape(rel)}' alt=''>"


def _video(rel):
    return f"<video controls muted preload='metadata' src='/sim2sim/{html.escape(rel)}'></video>"


def _pct(value):
    return "—" if value is None else f"{value:g}"


def page(root=ROOT):
    m = load(root)
    if m is None:
        return "<!doctype html><meta charset='utf-8'><p>尚未生成 sim2sim 案例。</p>".encode()
    t = m["target"]
    inputs = "".join(
        f"<figure>{_img(c['frame'], 'wide')}<figcaption>训练集 episode"
        f" {c['episode']} 第一帧（系统输入）</figcaption>{_video(c['video'])}</figure>"
        for c in m["train_inputs"]
    )
    examples = "".join(
        f"<figure>{_img(c['frame'], 'wide')}<figcaption>训练集 episode"
        f" {c['episode']}</figcaption>{_video(c['video'])}</figure>"
        for c in m["train_examples"]
    )
    outputs = []
    for u in m["usdcraft"]:
        if not u.get("record_id"):
            outputs.append(
                f"<article class='card'><h3>{html.escape(u['row_id'])}</h3><p"
                f" class='muted'>生成状态：{html.escape(u['status'])}</p></article>"
            )
            continue
        rows = (
            "".join(
                f"<tr><td>{html.escape(i['name'])}</td><td>{html.escape(i['kind'])}</td><td>{html.escape(str(i.get('joint')))}</td><td>{html.escape(str(i.get('type')))}</td><td>{i.get('limits')}</td><td>{i.get('start')} →"
                f" {i.get('target')}</td></tr>"
                for i in u["interactions"]
            )
            or "<tr><td colspan=6 class='bad'>没有声明交互</td></tr>"
        )
        latches = (
            "；".join(f"{latch['name']}：{latch['joint']} 由 {latch['release']} 释放" for latch in u["latches"]) or "无"
        )
        renders = "".join(_img(r) for r in u["renders"]) or "<div class='placeholder'>尚未渲染</div>"
        ref = next((r for r in m["reference_toasters"] if r["name"] == u.get("visible_instance")), None)
        size_note = ""
        if ref and u.get("render_size_m"):
            ratio = max(u["render_size_m"]) / max(ref["size_m"])
            size_note = (
                f"<p>与图中实际实例 {html.escape(ref['name'])} 对比：最长边 {max(u['render_size_m']):.3f} m vs"
                f" {max(ref['size_m']):.3f} m（{ratio:.2f}×）。单张图像没有尺度参照，尺寸是估计值。</p>"
            )
        outputs.append(
            f"<article class='card'><h3>{html.escape(u['row_id'])} <small>输入：episode"
            f" {u['input_episode']} 第一帧</small></h3><div class='thumbs'>{renders}</div><p>尺寸"
            f" {u.get('render_size_m')} m ·"
            f" <code>{html.escape(u['record_id'])}</code></p>{size_note}<table><tr><th>声明的操作</th><th>类型</th><th>关节</th><th>关节类型</th><th>限位</th><th>起点"
            f" → 目标</th></tr>{rows}</table><p>Latch：{html.escape(latches)}</p></article>"
        )
    reference = "".join(
        f"<article class='card'><h3>{html.escape(r['name'])} <small>{html.escape(r['split'])}</small></h3><div"
        f" class='thumbs'>{''.join(_img(x) for x in r['renders'])}</div><p>尺寸 {r['size_m']} m</p></article>"
        for r in m["reference_toasters"]
    )
    evals = []
    for run in m["local_eval"]:
        if run["status"] == "not_started":
            evals.append(f"<p>{html.escape(run['task'])}：尚未开始</p>")
            continue
        clips = "".join(
            f"<figure>{_video(c['video'])}<figcaption>episode {c['episode']} · layout {c['layout']} · 得分"
            f" {c['score']} · {'成功' if c['success'] else '失败'}</figcaption></figure>"
            for c in run["clips"]
        )
        evals.append(
            f"<h3>{html.escape(run['task'])}：{run['successes']}/{run['episodes']} 成功，平均得分"
            f" {_pct(run['score'])}（{'已完成' if run['status'] == 'complete' else '进行中'}）</h3><div"
            f" class='grid2'>{clips}</div>"
        )
    official = "".join(
        f"<tr><td>{html.escape(r['model'])}</td><td>{_pct(r['make_toast'][0])}% /"
        f" {_pct(r['make_toast'][1])}</td><td>{_pct(r['make_toast_random'][0])}% /"
        f" {_pct(r['make_toast_random'][1])}</td></tr>"
        for r in m["official"]["rows"]
    )
    local_row = "".join(
        (
            f"<td>{(run.get('successes', 0) / run['episodes'] * 100 if run.get('episodes') else 0):.1f}% /"
            f" {_pct(run.get('score'))}（{run.get('episodes', 0)} 回合）</td>"
            if run["status"] != "not_started"
            else "<td>未开始</td>"
        )
        for run in m["local_eval"]
    )
    act_row = "".join(
        (
            f"<td>{(run.get('successes', 0) / run['episodes'] * 100 if run.get('episodes') else 0):.1f}% /"
            f" {_pct(run.get('score'))}（{run.get('episodes', 0)} 回合）</td>"
            if run["status"] != "not_started"
            else "<td>未运行</td>"
        )
        for run in m.get("local_eval_act", [])
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sim2sim · RoboDojo make_toast</title><link rel="stylesheet" href="/style.css">
<style>
body{{overflow:auto}} main.s2s{{padding:20px 28px;max-width:1700px;line-height:1.6}}
.s2s h2{{margin:30px 0 8px;font-size:20px}} .s2s h3 small,.s2s h2 small{{color:var(--muted);font-weight:400;font-size:13px;margin-left:8px}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}} .flow div{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px;font-size:13px}}
.flow b{{color:var(--cyan)}} .done{{border-color:#2f6f5a!important}} .todo{{opacity:.6}}
.grid2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}} .grid4{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px}} figcaption{{color:var(--muted);font-size:12px;margin:4px 0}}
video,img.wide{{width:100%;border-radius:6px;background:#000}} .thumbs{{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}} img.thumb{{width:100%;border-radius:6px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;font-size:13px}}
table{{border-collapse:collapse;margin:8px 0;font-size:13px}} td,th{{border:1px solid var(--line);padding:5px 9px}} .muted{{color:var(--muted)}} .bad{{color:#ff7b7b}}
.rules td:first-child{{color:var(--cyan);white-space:nowrap}} .placeholder{{color:var(--muted);border:1px dashed var(--line);border-radius:6px;padding:16px;text-align:center}}
</style></head><body>
<header><div><span class="eyebrow">DATA ENGINE · SIM2SIM</span><h1>sim2sim 测试 · RoboDojo make_toast</h1></div>
<a href="/" style="color:var(--cyan)">← 任务审核台</a> <a href="/api/sim2sim" style="color:var(--cyan)">案例 JSON</a></header>
<main class="s2s">
<h2>1. 测试协议：系统能看到什么、要产出什么、怎么打分</h2>
<table class="rules">
<tr><td>目标环境 G</td><td>{html.escape(t['benchmark'])} · {html.escape(t['task'])} · {html.escape(t['robot'])} · {html.escape(t['cameras'])}</td></tr>
<tr><td>成功判据</td><td>{html.escape(t['success'])}</td></tr>
<tr><td>官方训练数据</td><td>{html.escape(t['official_train_data'])}</td></tr>
<tr><td><b>系统输入（允许）</b></td><td>训练集里的一张头部相机图像 + 任务描述（“make toast”）；机器人、相机、桌面规格属于 benchmark 公开规格，可以使用</td></tr>
<tr><td><b>禁止输入</b></td><td>RoboDojo 的物体资产文件（含训练用 0/1 号烤面包机）、评估布局 Eval_Layout、任务配置与物理参数、<code>_random</code> 用的 2/4 号烤面包机</td></tr>
<tr><td><b>系统输出</b></td><td>① 由图像生成的带关节和交互声明的资产（烤面包机 twin + cousins，面包架、面包）→ ② 重建的场景 → ③ data engine 采集的 demo（与官方同格式：14 维关节动作、3 路 640×480、25 fps、LeRobot）→ ④ 训练出的策略（在训练机上）</td></tr>
<tr><td><b>打分</b></td><td>策略回到 G 中评估（官方协议，seed 0/1/2）：只用官方 100 条 / 官方 + 我们生成的 / 只用我们生成的，三组对比；重点看 <code>make_toast_random</code>（未见过的烤面包机）上的提升，并同时报告部分得分</td></tr>
</table>
<div class="flow">
<div class="done"><b>输入</b><br>训练集第一帧 + 任务描述</div>
<div class="done"><b>① 资产</b><br>usdcraft 从图像生成烤面包机（本页第 3 节）</div>
<div class="todo"><b>② 场景</b><br>按图重建桌面布局（待做，第 3 步）</div>
<div class="todo"><b>③ 数据</b><br>双臂 X5 采集 demo（待做，第 3 步）</div>
<div class="todo"><b>④ 训练 + 评估</b><br>训练机训练，回到 G 评估（待做）</div>
</div>

<h2>2. 系统输入：训练集第一帧 <small>只用这张图，不读 RoboDojo 的资产和配置</small></h2>
<div class="grid2">{inputs}</div>
<h3>其他训练样本 <small>官方训练数据只覆盖两种烤面包机（灰、白）</small></h3>
<div class="grid2">{examples}</div>

<h2>3. 系统输出 ①：usdcraft 由图像生成的烤面包机</h2>
<div class="grid2">{''.join(outputs)}</div>

<h2>参考（不作为输入）：RoboDojo 自带的烤面包机实例 <small>0/1 出现在训练数据中；2/4 只在 make_toast_random 评估中出现</small></h2>
<div class="grid2">{reference}</div>

<h2>4. 目标环境中的评估：官方 Pi_05 checkpoint 的本地复现</h2>
<table><tr><th>模型</th><th>make_toast 成功率 / 得分</th><th>make_toast_random 成功率 / 得分</th></tr>
<tr><td><b>Pi_05（本地复现，seed 0）</b></td>{local_row}</tr><tr><td>ACT（本地复现，seed 0，已停止）</td>{act_row}</tr>{official}</table>
<p class="muted">官方数据来源：{html.escape(m['official']['source'])}。ACT 的官方成功率为 0（本地 22 回合也是 0），用它做增广实验会有地板效应，所以改用官方 Pi_05 作为基线，并同时报告得分。</p>
{''.join(evals)}
</main></body></html>""".encode()


def api(root=ROOT):
    return json.dumps(load(root) or {}, ensure_ascii=False).encode()
