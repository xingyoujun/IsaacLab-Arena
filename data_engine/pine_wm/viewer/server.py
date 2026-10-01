# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only task dashboard; serves only catalogued reports and validated videos."""

import argparse
import csv
import json
import mimetypes
import os
import re
from contextlib import suppress
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("ARENA_PINE_WM_EXPERIMENT_ROOT", ROOT / "outputs/pine_wm/first20"))


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def rows(path):
    try:
        with path.open(encoding="utf-8-sig") as stream:
            return {r["task_id"]: r for r in csv.DictReader(stream)}
    except OSError:
        return {}


def snapshot():
    progress = rows(BASE / "progress.csv")
    formal = rows(BASE / "qualification_v1/qualification.csv")
    tasks = read_json(ROOT / "data_engine/pine_wm/tasks.json", {})["tasks"]
    allowed = {}
    for name in (
        "progress.csv",
        "progress.md",
        "qualification_v1/qualification.csv",
        "qualification_v1/qualification.md",
    ):
        allowed["/files/" + name] = BASE / name
    review = read_json(ROOT / "data_engine/pine_wm/review_layouts.json", {})
    allowed["/files/layout_review.csv"] = ROOT / "docs/pine_wm_review/tasks.csv"
    allowed["/files/layout_review.md"] = ROOT / "docs/pine_wm_review/README.md"
    output = []
    for spec in tasks:
        tid = spec["task_id"]
        videos = []
        for manifest in sorted(BASE.glob(f"*/{tid}/demos.hdf5.cameras/demo_*_state_replay.json")):
            if not read_json(manifest, {}).get("validation_passed"):
                continue
            demo = manifest.name.removesuffix("_state_replay.json")
            for video in sorted(manifest.parent.glob(demo + "_*.mp4")):
                if ".part." in video.name:
                    continue
                url = "/files/" + video.relative_to(BASE).as_posix()
                allowed[url] = video
                videos.append(
                    {"url": url, "label": video.stem.removeprefix(demo + "_"), "run": manifest.parts[-4], "demo": demo}
                )
        historical_videos = videos
        live_videos = []
        annotation = None
        annotation_url = None
        for manifest in sorted(BASE.glob(f"review*/{tid}/live_preview/trial_*.json"), reverse=True):
            record = read_json(manifest, {})
            if not record.get("validation_passed"):
                continue
            snapshot_config = read_json(manifest.parent.parent / "layout_snapshot.json", {})
            current_config = review.get("tasks", {}).get(tid, {})
            relevant = (
                "positions_xy_m",
                "target_xy_m",
                "asset_sizes_mm",
                "drawer_yaw_deg",
                "row_pitch_m",
                "container_offsets_m",
                "tower_xy_m",
                "count",
                "target_yaw_deg",
            )
            if any(snapshot_config.get(k) != current_config.get(k) for k in relevant):
                continue
            for camera, filename in record.get("files", {}).items():
                video = manifest.parent / filename
                if video.parent != manifest.parent or not video.is_file():
                    continue
                url = "/files/" + video.relative_to(BASE).as_posix()
                allowed[url] = video
                live_videos.append({"url": url, "label": camera + "_rgb", "run": "新布局实录", "demo": manifest.stem})
            if live_videos:
                trial = int(manifest.stem.removeprefix("trial_"))
                results = read_json(manifest.parent.parent / "results.json", [])
                annotated = next(
                    (r for r in results if r.get("trial") == trial and r.get("annotation_alignment_checked")), None
                )
                if annotated is not None:
                    filename = annotated.get("annotations", "")
                    path = manifest.parent.parent / filename
                    if path.parent == manifest.parent.parent and path.is_file():
                        document = read_json(path, {})
                        if document.get("num_steps") == record.get("frames"):
                            annotation = {
                                k: v for k, v in document.items() if k not in {"step_skill_ids", "step_stage_ids"}
                            }
                            annotation_url = "/files/" + path.relative_to(BASE).as_posix()
                            allowed[annotation_url] = path
                break
        videos = live_videos
        recent = []
        for path in sorted(BASE.glob(f"review*/{tid}/results.json"), key=lambda p: p.stat().st_mtime):
            recent = read_json(path, [])
        p = progress.get(tid, {})
        q = formal.get(tid, {})
        output.append({
            "id": tid,
            "name": spec["name"],
            "sequence": spec.get("sequence", ""),
            "progress": p,
            "formal": q,
            "layout_review": review.get("tasks", {}).get(tid, {}),
            "videos": videos,
            "annotation": annotation,
            "annotation_url": annotation_url,
            "historical_video_count": len(historical_videos),
            "preview_result": recent[-1] if recent else {},
            "complete": False,
            "developed": p.get("已有开发回合通过") == "True",
        })
    return {
        "updated": datetime.now(timezone.utc).isoformat(),
        "tasks": output,
        "workflow": review.get("phase", "layout_review"),
        "monitor": read_json(BASE / "monitor.json", {}),
    }, allowed


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        path = unquote(self.path.split("?")[0])
        data, allowed = snapshot()
        if path == "/api/tasks":
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head:
                self.wfile.write(body)
            return
        file = Path(__file__).with_name("index.html") if path == "/" else allowed.get(path)
        if file is None or not file.is_file():
            self.send_error(404)
            return
        size = file.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            a, b = match.groups()
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if a and b else size - 1
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("X-Content-Type-Options", "nosniff")
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head:
            with suppress(BrokenPipeError, ConnectionResetError):
                with file.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
