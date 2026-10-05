# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assemble the sim2sim review case for RoboDojo make_toast (8091 /reviews/sim2sim).

Copies the evidence the page shows into outputs/sim2sim/make_toast_v0/ and writes manifest.json:
the allowed system input (training-set first frames and their videos), what our system produced
from it (USDCraft assets), the target environment's own toaster instances (reference only, never
an input), and local reproduction rollouts of the official ACT checkpoint next to the official
leaderboard numbers.

    .venv/bin/python tools/sim2sim/build_make_toast_case.py
"""

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "outputs/sim2sim/make_toast_v0"
LOCAL = Path("/home/ubuntu/playground/robodojo_local")
TRAIN_VIDEO = LOCAL / "train_preview/data/RoboDojo/make_toast/arx_x5/preview_video"
EVAL_ROOT = Path("/home/ubuntu/code/RoboDojo/eval_result/RoboDojo")
USDCRAFT = Path("/home/ubuntu/code/usdcraft_v2")
GEN_LOG = LOCAL / "sim2sim/usdcraft_toaster_v0.log"
INPUT_EPISODES = {"toaster_from_ep0": 0, "toaster_from_ep12": 12}
# The instance visible in each input frame, matched by eye (grey = 1, white = 0); used only for review.
VISIBLE_INSTANCE = {"toaster_from_ep0": "rd_toaster_1", "toaster_from_ep12": "rd_toaster_0"}
EXTRA_TRAIN_EPISODES = [4, 24]

# Official sim leaderboard snapshot (robodojo-benchmark.com front-end data, dated 2026-09-28).
OFFICIAL = dict(
    source="robodojo-benchmark.com leaderboard data (snapshot 2026-09-28), 3 seeds averaged",
    rows=[
        dict(model="Pi_05（官方）", make_toast=(1.33, 9.33), make_toast_random=(0, 2.67)),
        dict(model="ACT (Single Task)", make_toast=(0, 0), make_toast_random=(0, 0)),
        dict(model="InternW0-Δ (rank 1 on make_toast)", make_toast=(24, 39.67), make_toast_random=(8, 22.67)),
        dict(model="VPP2-Preview", make_toast=(18.67, 26.67), make_toast_random=(4, 13.67)),
        dict(model="X_VLA", make_toast=(2.67, 7.67), make_toast_random=(0, 1.67)),
    ],
)


def copy(src: Path, rel: str) -> str:
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
        shutil.copy2(src, dst)
    return rel


def train_case(episode: int) -> dict:
    name = f"episode_{episode:07d}_cam_head"
    return dict(
        episode=episode,
        frame=copy(LOCAL / "first_frames" / f"{name}.png", f"train/{name}.png"),
        video=copy(TRAIN_VIDEO / f"{name}.mp4", f"train/{name}.mp4"),
    )


def usdcraft_outputs() -> list[dict]:
    records = {}
    if GEN_LOG.is_file():
        for line in GEN_LOG.read_text().splitlines():
            m = re.search(r'row finished.*row_id="([^"]+)" \| status=(\w+)(?: \| record_id="([^"]+)")?', line)
            if m:
                records[m[1]] = dict(status=m[2], record_id=m[3])
    out = []
    renders = (
        json.loads((ROOT / "usdcraft/renders.json").read_text()) if (ROOT / "usdcraft/renders.json").is_file() else {}
    )
    for row_id, episode in INPUT_EPISODES.items():
        info = records.get(row_id, dict(status="running", record_id=None))
        entry = dict(
            row_id=row_id,
            input_episode=episode,
            visible_instance=VISIBLE_INSTANCE[row_id],
            **info,
            interactions=[],
            latches=[],
            renders=[],
        )
        rid = info.get("record_id")
        if rid:
            entry["usd"] = str(USDCRAFT / "data/cache/record_materialization" / rid / "isaac/model.usdc")
            sidecar = USDCRAFT / "data/cache/record_training" / rid / "interaction_annotations.json"
            if sidecar.is_file():
                ann = json.loads(sidecar.read_text())
                for item in ann.get("interactions", []):
                    joint = item.get("joint") or {}
                    entry["interactions"].append(
                        dict(
                            name=item["name"],
                            kind=item["kind"],
                            part=item.get("part"),
                            joint=joint.get("name"),
                            type=joint.get("type"),
                            limits=joint.get("limits"),
                            start=joint.get("start"),
                            target=joint.get("target"),
                        )
                    )
                entry["latches"] = [
                    dict(name=latch["name"], joint=latch["joint"]["name"], release=latch["release_joint"]["name"])
                    for latch in ann.get("latches", [])
                ]
                entry["size_m"] = (ann.get("asset") or {}).get("bounds", {}).get("size_m")
            entry["renders"] = [f"usdcraft/{r}" for r in renders.get(row_id, {}).get("renders", [])]
            entry["render_size_m"] = renders.get(row_id, {}).get("size_m")
        out.append(entry)
    return out


def eval_runs(policy: str) -> list[dict]:
    runs = []
    for task in ("make_toast", "make_toast_random"):
        dirs = sorted((EVAL_ROOT / task / policy / "arx_x5").glob("0_*/2026-*"))
        dirs = [d for d in dirs if (d / "_result.json").is_file()]
        if not dirs:
            runs.append(dict(policy=policy, task=task, status="not_started"))
            continue
        run = max(dirs, key=lambda d: len(json.loads((d / "_result.json").read_text()).get("details", {})))
        result = json.loads((run / "_result.json").read_text())
        details = result.get("details", {})
        picked = sorted(details, key=lambda k: (-details[k]["score"], int(k)))[:2] + sorted(details, key=int)[:1]
        clips = []
        for key in dict.fromkeys(picked):
            video = next(run.glob(f"episode_{int(key):07d}_cam_head_*.mp4"), None)
            if video:
                clips.append(
                    dict(
                        episode=int(key),
                        layout=details[key]["layout_id"],
                        score=details[key]["score"],
                        success=details[key]["success"],
                        video=copy(video, f"eval/{task}/{video.name}"),
                    )
                )
        runs.append(
            dict(
                policy=policy,
                task=task,
                status="complete" if len(details) >= 25 else "running",
                episodes=len(details),
                successes=sum(d["success"] for d in details.values()),
                success_rate=result.get("success_rate"),
                score=result.get("score"),
                clips=clips,
            )
        )
    return runs


def main() -> None:
    reference = json.loads((ROOT / "robodojo_toasters/renders.json").read_text())
    manifest = dict(
        schema="sim2sim.case.v0",
        target=dict(
            benchmark="RoboDojo (Isaac Sim 5.1, local reproduction)",
            task="make_toast / make_toast_random",
            robot="dual ARX X5, 25 Hz, 14-D joint actions",
            cameras="cam_head + 2 wrist d435, RGB 640x480",
            success="two slices standing in both slots, 2 left on the shelf, lever ≥85% travel, arms back home",
            official_train_data=(
                "100 teleoperated episodes per task (arx_x5), toaster instances 0/1 only; make_toast_random (unseen"
                " toasters 2/4 + clutter) has no training data"
            ),
        ),
        train_inputs=[train_case(e) for e in INPUT_EPISODES.values()],
        train_examples=[train_case(e) for e in EXTRA_TRAIN_EPISODES],
        usdcraft=usdcraft_outputs(),
        reference_toasters=[
            dict(
                name=k,
                split="train" if k[-1] in "01" else "eval (_random only)",
                size_m=v["size_m"],
                renders=[f"robodojo_toasters/{r}" for r in v["renders"]],
            )
            for k, v in reference.items()
        ],
        local_eval=eval_runs("Pi_05"),
        local_eval_act=eval_runs("ACT"),
        official=OFFICIAL,
    )
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(ROOT / "manifest.json")


if __name__ == "__main__":
    main()
