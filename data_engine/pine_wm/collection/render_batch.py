# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render representative successful first20 demonstrations sequentially on one GPU."""

import argparse
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("--tasks", nargs="+", required=True, help="One explicitly reviewed task at a time.")
parser.add_argument("--all-demos", action="store_true", help="Render every success rather than one per task.")
parser.add_argument("--timeout", type=float, default=3600)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
tasks = json.loads((Path(__file__).resolve().parents[3] / "data_engine/pine_wm/tasks.json").read_text())["tasks"]
from data_engine.pine_wm.collection.review_workflow import validate_run

validate_run("preview", args.tasks)
assert not args.all_demos, "Review only one representative success before user feedback"
tasks = [task for task in tasks if task["task_id"] in args.tasks]
summary = []
for task in tasks:
    folder = args.input / task["task_id"]
    dataset = folder / "demos.hdf5"
    results_path = folder / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    if not dataset.exists() or not any(result["success"] for result in results):
        summary.append({"task_id": task["task_id"], "status": "no_successful_recording"})
        continue
    cmd = [
        str(root / ".venv/bin/python"),
        "-u",
        str(root / "isaaclab_arena_cumotion/scripts/rerender_embodiment_cameras.py"),
        "--env",
        "pine_wm_first20",
        "--hdf5",
        str(dataset),
        "--env-config",
        str(folder / "environment.json"),
        "--headless",
        "--streams",
        "realsense_d435_rgb",
        "wrist_a_rgb",
        "wrist_b_rgb",
        "scene_cam_rgb",
    ]
    if not args.all_demos:
        cmd.extend(["--demo-range", "0", "1"])
    print("RENDER", task["task_id"], flush=True)
    with (folder / "camera_replay.log").open("w") as log:
        try:
            result = subprocess.run(
                cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), timeout=args.timeout
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
    summary.append({
        "task_id": task["task_id"],
        "returncode": returncode,
        "scope": "all_successes" if args.all_demos else "first_success_only",
        "visual_review_required": True,
    })
    (args.input / "camera_replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RENDER_RESULT", summary[-1], flush=True)
(args.input / "camera_replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
