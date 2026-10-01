# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Continuously record first20 progress and process health without modifying simulations."""

import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("ARENA_PINE_WM_EXPERIMENT_ROOT", ROOT / "outputs/pine_wm/first20"))


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    lock = (BASE / "monitor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    last_key = None
    last_change = time.monotonic()
    while True:
        try:
            for name, folder in [("progress.py", BASE), ("qualification_report.py", BASE / "qualification_v1")]:
                subprocess.run(
                    ["python3", str(ROOT / "data_engine/pine_wm/collection" / name), str(folder)],
                    cwd=ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=True,
                )
            tasks = {}
            for path in sorted((BASE / "qualification_v1").glob("T*/results.json")):
                rows = read(path, [])
                tasks[path.parent.name] = {
                    "attempts": len(rows),
                    "successes": sum(bool(r.get("success")) for r in rows),
                }
            videos = set()
            for path in BASE.glob("*/T*/demos.hdf5.cameras/demo_*_state_replay.json"):
                if read(path, {}).get("validation_passed"):
                    videos.add(path.parts[-3])
            processes = []
            for path in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    args = path.read_bytes().split(b"\0")
                    names = [Path(a.decode(errors="replace")).name for a in args[:4]]
                    if any(
                        n in ("qualify.py", "rerender_embodiment_cameras.py", "run_batch.py", "render_batch.py")
                        for n in names
                    ):
                        processes.append(
                            {"pid": int(path.parent.name), "command": b" ".join(args).decode(errors="replace")}
                        )
                except (OSError, ValueError):
                    continue
            count = sum(t["attempts"] for t in tasks.values())
            key = (count, len(videos), tuple(p["pid"] for p in processes))
            if key != last_key:
                last_change = time.monotonic()
            warning = ""
            if not processes and count < 400:
                warning = "采集未完成，但没有采集或渲染进程"
            elif processes and time.monotonic() - last_change > 1800:
                warning = "超过 30 分钟无新增回合或视频，请检查日志；不自动终止进程"
            workflow = read(ROOT / "data_engine/pine_wm/review_layouts.json", {})
            if not workflow.get("launch_authorized", True):
                warning = "预览执行已关闭，等待用户审核布局；没有运行进程是预期状态"
                if processes:
                    warning = "异常：等待布局审核期间检测到采集或渲染进程"
            preview_ready = set()
            for manifest in BASE.glob("review*/T*/live_preview/trial_*.json"):
                saved = read(manifest.parent.parent / "layout_snapshot.json", {})
                current = workflow.get("tasks", {}).get(manifest.parts[-3], {})
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
                if read(manifest, {}).get("validation_passed") and all(
                    saved.get(k) == current.get(k) for k in relevant
                ):
                    preview_ready.add(manifest.parts[-3])
            if workflow.get("phase") == "single_success_preview" and not processes and len(preview_ready) < 20:
                warning = f"新布局预览仍有 {20 - len(preview_ready)} 项待处理，当前没有执行进程；并非全部完成"
            status = {
                "preview_tasks_with_video": sorted(preview_ready),
                "updated": datetime.now(timezone.utc).isoformat(),
                "attempts": count,
                "completed_task_batches": sum(t["attempts"] >= 20 for t in tasks.values()),
                "video_tasks": sorted(videos),
                "tasks": tasks,
                "processes": processes,
                "warning": warning,
            }
            temporary = BASE / "monitor.json.tmp"
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2))
            temporary.replace(BASE / "monitor.json")
            if key != last_key or warning:
                print(json.dumps(status, ensure_ascii=False), flush=True)
            last_key = key
        except Exception as error:
            print(datetime.now(timezone.utc).isoformat(), "MONITOR_ERROR", repr(error), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
