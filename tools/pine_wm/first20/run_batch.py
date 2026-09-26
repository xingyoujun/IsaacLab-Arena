# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run task qualifications sequentially; keep every failure and never share the GPU."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tasks", nargs="+")
parser.add_argument("--stage", choices=["preview", "stability", "collect"], default="preview")
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--timeout", type=float, default=14400)
parser.add_argument("--audit-only", action="store_true")
parser.add_argument("--workspace", type=float, nargs=4, default=(-0.30, 0.30, -0.12, 0.25))
parser.add_argument("--require-frozen", action="store_true", help="Abort if qualification source files change.")
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
specs = json.loads((Path(__file__).parent / "tasks.json").read_text())["tasks"]
if args.tasks:
    by_id = {task["task_id"]: task for task in specs}
    assert len(args.tasks) == len(set(args.tasks)), "Duplicate task IDs"
    assert set(args.tasks) <= set(by_id), "Unknown task ID"
    specs = [by_id[task_id] for task_id in args.tasks]
from review_workflow import validate_run

validate_run(args.stage, [task["task_id"] for task in specs])
args.output.mkdir(parents=True, exist_ok=True)
summary = []
batch_hashes = None
for task in specs:
    tid = task["task_id"]
    if args.tasks and tid not in args.tasks:
        continue
    out = args.output / tid
    assert (
        not (out / "results.json").exists() and not (out / "demos.hdf5").exists()
    ), f"Refusing to overwrite prior results: {out}"
    out.mkdir(parents=True, exist_ok=True)
    sources = (
        list(Path(__file__).parent.glob("*.py"))
        + list((root / "isaaclab_arena_environments").glob("pine_wm*.py"))
        + list((root / "isaaclab_arena/embodiments/ur7e").glob("pine_wm*.py"))
        + list((root / "isaaclab_arena/embodiments/ur7e/rmpflow").glob("pine_wm_ur7e.*"))
        + list((root / "isaaclab_arena/embodiments/ur7e/calibration/pine_wm").glob("*.json"))
        + [Path(__file__).parent / "tasks.json", Path(__file__).parent / "review_layouts.json"]
    )
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if batch_hashes is None:
        batch_hashes = hashes
        (args.output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if args.require_frozen:
        assert hashes == batch_hashes, "Source changed during the frozen qualification batch"
    (out / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    cmd = [
        str(root / ".venv/bin/python"),
        "-u",
        str(Path(__file__).with_name("qualify.py")),
        "--task",
        tid,
        "--trials",
        str(args.trials),
        "--seed",
        str(args.seed),
        "--output",
        str(out),
    ]
    cmd.extend(["--workspace", *map(str, args.workspace)])
    if args.audit_only:
        cmd.append("--audit-only")
    print("START", tid, task["name"], flush=True)
    with (out / "runtime.log").open("w") as log:
        try:
            run = subprocess.run(
                cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), timeout=args.timeout
            )
            returncode = run.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            log.write("\nQUALIFICATION_TIMEOUT\n")
    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else []
    end_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    source_unchanged = end_hashes == hashes == batch_hashes
    (out / "source_integrity.json").write_text(json.dumps({"unchanged": source_unchanged}, indent=2) + "\n")
    summary.append({
        "task_id": tid,
        "name": task["name"],
        "returncode": returncode,
        "trials_completed": len(results),
        "successes": sum(r["success"] for r in results),
        "errors": [r.get("error") or "success_predicate_false" for r in results if not r["success"]],
        "audit_only": args.audit_only,
        "source_unchanged": source_unchanged,
    })
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("FINISH", summary[-1], flush=True)

    if args.require_frozen:
        assert source_unchanged, "Source changed during task execution; do not qualify this batch"
