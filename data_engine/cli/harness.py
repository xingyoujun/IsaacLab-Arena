# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena data pipeline: resolve, preview and audit without duplicating environment registration."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from data_engine.harness.catalog import resolve, scenarios
    from data_engine.harness.runner import preview, resume_audit
    from data_engine.harness.storage import read, write

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    explain = sub.add_parser("explain")
    explain.add_argument("--scenario", required=True)
    for command in ("plan", "preflight", "preview"):
        child = sub.add_parser(command)
        child.add_argument("--scenario", required=True)
        child.add_argument("--assets", type=Path, default=ROOT / "local_assets/USDCraft-Scene")
        child.add_argument("--device", default="cuda:0")
        child.add_argument("--seed", type=int, default=0)
        if command == "preview":
            child.add_argument("--run-dir", type=Path, required=True)
            child.add_argument("--timeout", type=float, default=1800)
        if command == "plan":
            child.add_argument("--output", type=Path)
    for command in ("audit", "status"):
        child = sub.add_parser(command)
        child.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "list":
        result = scenarios()
    elif args.command == "explain":
        from data_engine.planning.task_plan import load_plan

        scenario = scenarios()[args.scenario]
        result = dict(scenario=scenario, task_plan=load_plan(scenario["adapter"], scenario["task_id"]))
    elif args.command == "status":
        result = read(args.run_dir / "run.json")
    elif args.command == "audit":
        result = resume_audit(args.run_dir)
    else:
        result = resolve(args.scenario, args.assets, args.seed, args.device)
        if args.command == "preview":
            result = preview(result, args.run_dir, args.timeout)
        elif args.command == "plan" and args.output:
            assert not args.output.exists(), "Use a new plan destination"
            write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
