# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resume-safe, bounded successful-demo collection followed by offline LeRobot rendering."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from g2_dataset_io import atomic_json, file_digest, read_core

HERE = Path(__file__).resolve().parent
PYTHON = "/isaac-sim/python.sh"


def run_child(command, log_path, timeout=None):
    """Run an isolated worker, forwarding interruption and bounding its process lifetime."""
    with log_path.open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def prepare(args, root, work):
    """Check the immutable run configuration and archive replay source dependencies."""
    config = {
        "robot": "g2",
        "task": "peg_into_sleeve",
        "seed": args.seed,
        "xy_noise_m": args.xy_noise_m,
        "table_height_m": args.table_height_m,
        "robot_yaml_sha256": file_digest(args.robot_yaml),
        "robot_urdf_sha256": file_digest(args.robot_urdf),
        "raw_schema": 1,
        "lerobot_version": "v2.1",
    }
    config_path = work / "collection_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Cannot resume with a different collection configuration"
    else:
        assert not (root / "meta").exists(), "Refusing to modify an unrelated existing dataset"
        atomic_json(config_path, config)
    provenance = work / "provenance"
    provenance.mkdir(exist_ok=True)
    snapshots = {
        "robot.yaml": args.robot_yaml,
        "robot.urdf": args.robot_urdf,
        **{
            name: HERE / name
            for name in (
                "g2_collect_peg_dataset.py",
                "g2_sleeve_curobo.py",
                "g2_dataset_io.py",
                "check_g2_sleeve_dataset.py",
                "g2_stack_bowls_curobo.py",
                "g2_raw_to_lerobot.py",
            )
        },
        "g2.py": HERE.parent / "isaaclab_arena/embodiments/g2/g2.py",
        "recorders.py": HERE.parent / "isaaclab_arena/embodiments/g2/recorders.py",
        "environment.py": HERE.parent / "isaaclab_arena_environments/g2_sleeve_environment.py",
        "sleeve_task.py": HERE.parent / "isaaclab_arena/tasks/sleeve_task.py",
        "workbench.py": HERE.parent / "isaaclab_arena_environments/g2_workbench_environments.py",
        "room.usda": HERE.parent / "isaaclab_arena/embodiments/g2/assets/stack_bowls_room.usda",
        "geniesim_library.py": HERE.parent / "isaaclab_arena/assets/geniesim_library.py",
    }
    for name, source in snapshots.items():
        target = provenance / name
        if target.exists():
            assert file_digest(target) == file_digest(source), f"Source changed since collection began: {name}"
        else:
            shutil.copyfile(source, target)
    atomic_json(provenance / "source_hashes.json", {name: file_digest(path) for name, path in snapshots.items()})
    return provenance


def scan_attempts(root, args):
    """Recover validated successes, including workers completed before a supervisor interruption."""
    results, successes = [], []
    for directory in sorted((root / "attempts").glob("attempt_*")):
        attempt = int(directory.name.split("_")[-1])
        result = {"attempt": attempt, "seed": args.seed + attempt, "raw_dir": str(directory), "success": False}
        try:
            arrays, report = read_core(directory / "episodes.hdf5")
            cfg = report["configuration"]
            assert cfg["seed"] == args.seed + attempt and cfg["xy_noise_m"] == args.xy_noise_m
            assert cfg["table_height_m"] == args.table_height_m
            result.update(success=True, frames=len(arrays["action"]), sha256=file_digest(directory / "episodes.hdf5"))
            successes.append(result)
        except (OSError, KeyError, AssertionError, ValueError) as exc:
            result["error"] = str(exc)
        results.append(result)
    atomic_json(root / "collection_manifest.json", {"attempts": results, "successes": successes})
    return results, successes


def collect(args, root, provenance):
    """Collect bounded independent seeds until the requested number of successful demos exists."""
    results, successes = scan_attempts(root, args)
    next_attempt = max((item["attempt"] for item in results), default=-1) + 1
    infrastructure_failures = 0
    while len(successes) < args.demos and next_attempt < args.max_attempts:
        directory = root / "attempts" / f"attempt_{next_attempt:06d}"
        log = root / "logs" / f"attempt_{next_attempt:06d}.log"
        command = [
            PYTHON,
            str(HERE / "g2_sleeve_curobo.py"),
            "--headless",
            "--device",
            "cpu",
            "--robot_yaml",
            str(provenance / "robot.yaml"),
            "--robot_urdf",
            str(provenance / "robot.urdf"),
            "--output_dir",
            str(directory),
            "--attempts",
            "1",
            "--seed",
            str(args.seed + next_attempt),
            "--table_height_m",
            str(args.table_height_m),
            "--xy_noise_m",
            str(args.xy_noise_m),
        ]
        print(f"COLLECT attempt={next_attempt} successes={len(successes)}/{args.demos} log={log}", flush=True)
        try:
            run_child(command, log, args.attempt_timeout_s)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT attempt={next_attempt}; raw retained", flush=True)
        # Even an initialization crash consumes the seed and leaves a durable diagnostic record.
        directory.mkdir(parents=True, exist_ok=True)
        infrastructure_failures = 0 if (directory / "report.json").exists() else infrastructure_failures + 1
        results, successes = scan_attempts(root, args)
        assert infrastructure_failures < 3, "Three consecutive worker initialization/crash failures; inspect logs"
        next_attempt += 1
    assert len(successes) >= args.demos, f"Attempt budget exhausted: {len(successes)}/{args.demos} successes"
    return successes[: args.demos]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/datasets/agibot_dataset_v1"))
    parser.add_argument("--raw_root", type=Path, help="Raw/debug root; defaults to the dataset root plus _raw")
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "pilot"),
        help="train exports at task root; pilot keeps all output under raw_root/pilot",
    )
    parser.add_argument("--demos", type=int, default=200)
    parser.add_argument("--max_attempts", type=int, default=600)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--xy_noise_m", type=float, default=0.02)
    parser.add_argument("--table_height_m", type=float, default=0.75)
    parser.add_argument("--attempt_timeout_s", type=float, default=900)
    parser.add_argument("--stage", choices=("all", "core", "render"), default="all")
    parser.add_argument("--robot_yaml", type=Path, required=True)
    parser.add_argument("--robot_urdf", type=Path, required=True)
    args = parser.parse_args()
    assert args.demos > 0 and args.max_attempts >= args.demos
    dataset_base = args.root.resolve()
    raw_base = args.raw_root.resolve() if args.raw_root else dataset_base.with_name(dataset_base.name + "_raw")
    assert raw_base != dataset_base and dataset_base not in raw_base.parents, "Raw must be outside the sync root"
    root = dataset_base / "peg_into_sleeve"
    work = raw_base / "peg_into_sleeve"
    if args.split == "pilot":
        root = raw_base / "pilot" / "peg_into_sleeve"
        work = raw_base / "pilot" / "peg_into_sleeve_work"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("attempts", "logs"):
        (work / name).mkdir(parents=True, exist_ok=True)
    with (work / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = prepare(args, root, work)
        if args.stage != "render":
            collect(args, work, provenance)
        if args.stage != "core":
            _, successes = scan_attempts(work, args)
            assert len(successes) >= args.demos, "Collect the requested successful core episodes before rendering"
            command = [
                PYTHON,
                str(HERE / "g2_raw_to_lerobot.py"),
                "--root",
                str(root),
                "--work_dir",
                str(work),
                "--demos",
                str(args.demos),
                "--enable_cameras",
                "--headless",
                "--device",
                "cpu",
            ]
            code = run_child(command, work / "logs" / "render.log")
            assert code == 0 and (root / "meta/info.json").exists(), f"Renderer failed; inspect {work}/logs/render.log"
            info = json.loads((root / "meta/info.json").read_text())
            assert info["total_episodes"] == args.demos, "Render incomplete; resume with --stage render"
        print(f"DONE stage={args.stage} dataset={root} time={time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
