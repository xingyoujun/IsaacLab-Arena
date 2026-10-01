# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task review dashboard with isolated display names and immutable recording evidence."""

import argparse
import html
import json
import mimetypes
import re
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from data_engine.viewer.presentation import TITLES, default_name, group_tasks, timeline


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def snapshot(root):
    """Read explicit run manifests; an MP4 by itself never becomes a successful run."""
    records, allowed = [], {}
    for path in sorted(root.glob("*/run.json"), reverse=True):
        run = read(path)
        if not run or run.get("schema") != "arena.collection.run.v1":
            continue
        directory = path.parent
        spec = run["spec"]
        quality = read(directory / "quality.json", {})
        diagnostic_review = read(directory / "diagnostic_review.json", {})
        if diagnostic_review.get("spec_id") != spec["spec_id"]:
            diagnostic_review = {}
        diagnostic = read(directory / "payload/raw/report.json", {})
        pine_results = read(directory / "payload/results.json", [])
        if pine_results:
            diagnostic = pine_results[0] if isinstance(pine_results, list) else pine_results
        record = dict(
            name=directory.name,
            scenario=spec["scenario"]["id"],
            task_title=TITLES.get(spec["scenario"]["id"], spec["scenario"].get("instruction", spec["scenario"]["id"])),
            state=run["state"],
            created_at=run["created_at"],
            spec_id=spec["spec_id"],
            asset_manifest_sha256=spec.get("asset_manifest_sha256"),
            asset_root=spec.get("asset_root"),
            events=run["events"],
            quality=quality,
            diagnostic_review=diagnostic_review,
            task_plan=spec["scenario"].get("task_plan", {}),
            decisions=read(directory / "payload/raw/decisions.json", diagnostic.get("decisions", [])),
            phases=diagnostic.get("phases", []),
            videos=[],
            stages=[],
            failure=diagnostic.get("error"),
        )
        for name in ("run.json", "quality.json", "diagnostic_review.json"):
            candidate = directory / name
            if candidate.is_file():
                allowed[f"/{directory.name}/{name}"] = candidate
        if run["state"] == "preview_validated" and quality.get("spec_id") == spec["spec_id"]:
            for camera, video in quality.get("videos", {}).items():
                relative = "payload/" + video["path"]
                candidate = directory / relative
                if video["path"] not in run["artifacts"] or not candidate.resolve().is_relative_to(directory.resolve()):
                    continue
                if candidate.is_file():
                    url = f"/{directory.name}/{relative}"
                    allowed[url] = candidate
                    record["videos"].append(dict(name=camera, url=url))
            trace = read(directory / "payload/trial_000_annotations.json", {})
            if not trace:
                trace = read(directory / "payload/annotations.json", {})
            if trace.get("num_steps") == quality.get("raw", {}).get("steps"):
                record["stages"] = trace.get("stages", [])
                record["skills"] = trace.get("skills", [])
                record["stage_source"] = "recorded_skill_trace"
                record["step_dt"] = trace.get("step_dt_s", 0)
            elif diagnostic.get("frames") == quality.get("raw", {}).get("steps") and record["phases"]:
                phases = record["phases"]
                frames = [phase["frame"] for phase in phases] + [diagnostic["frames"]]
                if frames == sorted(frames) and frames[0] >= 0:
                    record["stages"] = [
                        dict(name=phase["name"], start_step=phase["frame"], end_step=end)
                        for phase, end in zip(phases, frames[1:])
                    ]
                    record["step_dt"] = 1 / spec["scenario"]["capture_profile"]["fps"]
                    record["stage_source"] = "collector_phases_preview_only"
        metadata = read(directory / "viewer_metadata.json", {})
        record["display_name"] = default_name(record, spec["scenario"])
        if metadata.get("spec_id") == spec["spec_id"] and metadata.get("display_name"):
            record["display_name"] = metadata["display_name"]
        record["timeline"] = timeline(record)
        record["planning_events"] = [s for s in record["stages"] if s.get("start_step") == s.get("end_step")]
        records.append(record)
    group_tasks(records)  # Assign chronological attempt numbers for both API views.
    return records, allowed


def review_files(root, records):
    """Expose curated review evidence without opening arbitrary output paths."""
    repository = Path(__file__).resolve().parents[2]
    manifest = read(Path(__file__).parent / "review_assets.json", {})
    allowed = {}
    for route, relative in manifest.items():
        candidate = (repository / relative).resolve()
        if candidate.is_relative_to(repository) and candidate.is_file():
            allowed[route] = candidate
    # Raw evidence is downloadable only for the four explicitly reviewed, sealed runs.
    reviewed = {
        "kettle_release_v2_20260930_12",
        "toaster_cancel_v2_20260930_08",
        "pine_t001_v2_20260930_02",
        "pine_t041_v2_20260930_01",
    }
    for record in records:
        if record["name"] not in reviewed or record["state"] != "preview_validated":
            continue
        if record["quality"].get("spec_id") != record["spec_id"]:
            continue
        directory = root / record["name"]
        run = read(directory / "run.json", {})
        for name in ("results.json", "demos.hdf5", "annotations.json", "trial_000_annotations.json"):
            candidate = (directory / "payload" / name).resolve()
            if (
                name in run.get("artifacts", {})
                and candidate.is_relative_to(directory.resolve())
                and candidate.is_file()
            ):
                allowed[f"/{record['name']}/payload/{name}"] = candidate
    return allowed


def page(records):
    """Bootstrap the task dashboard; escape embedded JSON independently of HTML text."""
    data = json.dumps(group_tasks(records), ensure_ascii=False)
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    fallback = "".join(html.escape(r.get("diagnostic_review", {}).get("summary", "")) for r in records)
    template = (Path(__file__).parent / "index.html").read_text()
    values = {"__BOOTSTRAP__": data, "__FALLBACK__": fallback}
    return re.sub(r"__(BOOTSTRAP|FALLBACK)__", lambda match: values[match[0]], template).encode()


def rename_attempt(root, value):
    """Persist only a spec-bound display label; never change directories or sealed artifacts."""
    from data_engine.harness.storage import write

    records, _ = snapshot(root)
    matches = [r for r in records if r["name"] == value["run"]]
    assert len(matches) == 1, "Unknown attempt"
    record = matches[0]
    assert value["spec_id"] == record["spec_id"], "Stale attempt identity"
    label = value["display_name"]
    assert isinstance(label, str), "Name must be text"
    label = label.strip()
    assert 1 <= len(label) <= 80 and not any(ord(c) < 32 for c in label), "Use 1-80 printable characters"
    directory = root / record["name"]
    assert directory.resolve().parent == root.resolve(), "Invalid attempt directory"
    write(directory / "viewer_metadata.json", dict(spec_id=record["spec_id"], display_name=label))


def serve(root, host, port):
    """Serve indexed files with byte ranges for video seeking."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            route = unquote(self.path.split("?", 1)[0])
            if route != "/api/attempt-name":
                self.send_error(404)
                return
            if self.headers.get("Origin") != "http://" + self.headers.get("Host", ""):
                self.send_error(403, "Same-origin edits only")
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_error(415)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                assert 0 < size <= 4096, "Invalid request size"
                value = json.loads(self.rfile.read(size))
                rename_attempt(root, value)
            except (AssertionError, ValueError, TypeError, KeyError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            records, allowed = snapshot(root)
            route = unquote(self.path.split("?", 1)[0])
            allowed.update(review_files(root, records))
            if route == "/reviews/usdcraft-v2-20260930":
                allowed[route] = Path(__file__).parent / "review.html"
            if route in ("/app.js", "/style.css", "/review.css"):
                asset = Path(__file__).parent / route[1:]
                body = asset.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript" if route.endswith(".js") else "text/css")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
                return
            if route in ("/reviews/pine-wm", "/api/pine-tasks"):
                from data_engine.viewer.inventory import inventory
                from data_engine.viewer.inventory import page as inventory_page

                body = (
                    inventory_page(records) if route == "/reviews/pine-wm" else json.dumps(inventory(records)).encode()
                )
                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/html; charset=utf-8" if route.startswith("/reviews/") else "application/json"
                )
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if route in ("/", "/api/runs", "/api/tasks"):
                data = group_tasks(records) if route == "/api/tasks" else records
                body = page(records) if route == "/" else json.dumps(data, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8" if route == "/" else "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            path = allowed.get(route)
            if path is None:
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                if not match:
                    self.send_error(416)
                    return
                start = int(match[1])
                end = min(int(match[2]), end) if match[2] else end
                if start > end or start >= size:
                    self.send_error(416)
                    return
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError), path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/harness")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    serve(args.root.resolve(), args.host, args.port)
