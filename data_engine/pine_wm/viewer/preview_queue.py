# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sequential one-attempt previews; never automatically repeat a successful task."""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(os.environ.get("ARENA_PINE_WM_EXPERIMENT_ROOT", ROOT / "outputs/pine_wm/first20")) / "review",
)
parser.add_argument("--tasks", nargs="+")
args = parser.parse_args()
BASE = args.output
env = os.environ.copy()
env.update(
    OMNI_KIT_ACCEPT_EULA="YES",
    ACCEPT_EULA="Y",
    PYTHONPATH=os.pathsep.join(
        filter(
            None,
            [
                str(ROOT),
                os.environ.get("ARENA_ISAACLAB_SOURCE", str(ROOT / "submodules/IsaacLab/source/isaaclab")),
                os.environ.get("PYTHONPATH"),
            ],
        )
    ),
)
tasks = json.loads((ROOT / "data_engine/pine_wm/tasks.json").read_text())["tasks"]
if args.tasks:
    assert set(args.tasks) <= {task["task_id"] for task in tasks}
    tasks = [task for task in tasks if task["task_id"] in args.tasks]
for task in tasks:
    tid = task["task_id"]
    current = json.loads((ROOT / "data_engine/pine_wm/review_layouts.json").read_text())["tasks"][tid]
    prior_success = False
    for previous in BASE.parent.glob(f"review*/{tid}/results.json"):
        layout_file = previous.parent / "layout_snapshot.json"
        if not layout_file.exists():
            continue
        saved = json.loads(layout_file.read_text())
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
        if all(saved.get(k) == current.get(k) for k in relevant):
            prior_success |= any(r.get("success") for r in json.loads(previous.read_text()))
    if prior_success:
        print("STOP_AFTER_EXISTING_SUCCESS", tid, flush=True)
        continue
    folder = BASE / tid
    if (folder / "results.json").exists():
        print("RETAIN_EXISTING", tid, flush=True)
        continue
    assert not (folder / "runtime.log").exists(), f"Inspect interrupted attempt first: {folder}"
    folder.mkdir(parents=True, exist_ok=True)
    print("PREVIEW", tid, flush=True)
    with (folder / "runtime.log").open("w") as log:
        cmd = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / "data_engine/pine_wm/collection/qualify.py"),
            "--task",
            tid,
            "--trials",
            "1",
            "--enable_cameras",
            "--device",
            "cuda:0",
            "--output",
            str(folder),
        ]
        try:
            code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600).returncode
        except subprocess.TimeoutExpired:
            code = 124
    print("PREVIEW_FINISHED", tid, code, flush=True)
print("PREVIEW_PASS_COMPLETE: inspect failures, no stability or collection launched", flush=True)
