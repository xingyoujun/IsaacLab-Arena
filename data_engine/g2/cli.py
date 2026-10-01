# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single local G2 entrypoint: pinned assets, native cuMotion, raw audit, replay and export."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TASKS_PATH = ROOT / "data_engine/g2/tasks.json"
TASKS = json.loads(TASKS_PATH.read_text())["tasks"]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def preflight(asset_root, task):
    """Resolve every required ID from a verified bundle; never fall back to old mounts."""
    from data_engine.assets.manage import digest, verify
    from isaaclab_arena.assets.usdcraft_scene import bundle_root, resolve_asset

    root = bundle_root(asset_root).resolve()
    os.environ["ARENA_USDCRAFT_SCENE_ROOT"] = str(root)
    manifest = verify(root)
    required = [
        "g2",
        "g2_room",
        "g2_table",
        "g2_robot_yaml",
        "g2_tcp_urdf",
        "g2_task_registry",
        "g2_camera_calibration",
        "g2_workcell_motion",
        *task["assets"],
    ]
    for name in required:
        resolve_asset(name, root)
    snapshots = {
        "g2_task_registry": TASKS_PATH,
        "g2_camera_calibration": ROOT / "isaaclab_arena/embodiments/g2/g2.py",
        "g2_workcell_motion": ROOT / "data_engine/g2/workcell_motion.yaml",
    }
    for name, source in snapshots.items():
        assert digest(resolve_asset(name, root)) == digest(source), f"Git/HF snapshot mismatch: {name}; restage assets"
    return dict(
        root=str(root),
        manifest_sha256=digest(root / "manifest.json"),
        required_ids=required,
        repository=manifest["repository"],
    )


def child(arguments, log):
    """Isolate each simulator lifetime and retain a separate process log."""
    env = dict(os.environ, ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE="1", PYTHONUNBUFFERED="1", ARENA_G2_FRAMEWORK="1")
    with log.open("w") as stream:
        result = subprocess.run(
            # AppLauncher checks sys.argv when routing Kit output. Keep PhysX warnings in stage logs.
            [sys.executable, *map(str, arguments), "--info"],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    assert result.returncode == 0, f"Stage failed ({result.returncode}); inspect {log}"


def worker(task_id, raw, seed, device):
    """Start an internal collector only after bundle verification."""
    task = TASKS[task_id]
    preflight(None, task)
    if task["collector"] == "workcell":
        from data_engine.g2.collection.workcell.collect import main as collect_workcell

        collect_workcell([
            "--output_dir",
            str(raw),
            "--planner_backend",
            "cumotion",
            "--stage",
            "sequence",
            "--device",
            device,
            "--seed",
            str(seed),
            "--num_envs",
            "1",
            "--headless",
        ])
    else:
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

        args = get_isaaclab_arena_cli_parser().parse_args(["--device", device, "--seed", str(seed), "--num_envs", "1"])
        args.task, args.output_dir, args.headless = task_id, raw, True
        with SimulationAppContext(args):
            from data_engine.g2.collection.session import collect

            collect(args, task)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["list", "preflight", "collect", "validate", "render", "export", "run", "_collect"]
    )
    parser.add_argument("--info", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--official-loader", action="store_true", help="Validate exported samples with an installed LeRobot v2.1 loader"
    )
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(TASKS, indent=2))
        return
    assert args.device.startswith("cuda"), "G2 collection and replay require CUDA"
    assert args.run_dir is not None or args.command == "preflight", "--run-dir required"
    if args.command in ("collect", "run", "preflight", "_collect"):
        assert args.task, "--task required"
        task = TASKS[args.task]
    else:
        saved = json.loads((args.run_dir / "run.json").read_text())
        args.task = saved["task_id"]
        task = TASKS[args.task]
        assert saved["task"] == task, "Task changed; use matching source revision for this run"
    assets = preflight(args.assets, task)
    if args.command == "preflight":
        print(json.dumps(assets, indent=2))
        return
    root = args.run_dir.resolve()
    if args.command == "_collect":
        worker(args.task, root, args.seed, args.device)
        return
    if args.command in ("collect", "run"):
        root.mkdir(parents=True, exist_ok=False)
        write(
            root / "run.json",
            dict(
                schema="arena.g2.run.v1",
                task_id=args.task,
                task=task,
                assets=assets,
                seed=args.seed,
                device=args.device,
                planner="native_cumotion",
                task_registry_sha256=hashlib.sha256(TASKS_PATH.read_bytes()).hexdigest(),
            ),
        )
        write(root / "environment.json", task["environment_config"])
        child(
            [
                __file__,
                "_collect",
                "--task",
                args.task,
                "--run-dir",
                root / "raw",
                "--seed",
                args.seed,
                "--device",
                args.device,
            ],
            root / "collect.log",
        )
    else:
        assert saved["assets"]["manifest_sha256"] == assets["manifest_sha256"], "Asset release differs from recording"
    if args.command in ("run", "collect", "validate", "render", "export"):
        from data_engine.g2.collection.validation import audit_run

        write(
            root / "validation.json",
            audit_run(root, official_loader=args.official_loader and args.command == "validate"),
        )
    if args.command in ("run", "render"):
        arguments = [
            "data_engine/g2/collection/workcell/rerender_states.py",
            "--env",
            task["environment"],
            "--output_dir",
            root / "render",
            "--device",
            args.device,
            "--num_envs",
            "1",
            "--enable_cameras",
            "--headless",
        ]
        if task["collector"] == "workcell":
            arguments += ["--raw_dir", root / "raw"]
        else:
            arguments += ["--hdf5", root / "raw/episodes.hdf5", "--env_config", root / "environment.json"]
        child(arguments, root / "render.log")
        write(root / "validation.json", audit_run(root))
    if args.command in ("run", "export"):
        from data_engine.g2.export import export

        export(root / "raw", root / "render", root / "lerobot", task_id=args.task)
        write(root / "validation.json", audit_run(root, official_loader=args.official_loader))
    print("G2_STAGE_COMPLETE", args.command, root, flush=True)


if __name__ == "__main__":
    main()
