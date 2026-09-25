# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Explicitly launch isolated black-gripper toaster camera replay; never replace existing training datasets."""

import argparse
import fcntl
import h5py
import hashlib
import json
import os
from pathlib import Path

from collect_ur7e_open_drawer_gpt56 import REPO, SCRIPTS, merge_exact, run_workers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/ubuntu/playground/datasets/rr_sim2real_raw/press_toaster/press_toaster_v2.hdf5"),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("/home/ubuntu/playground/datasets/rr_sim2real_raw/usdcraft_press_toaster_black_gripper"),
    )
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--skies", type=Path, default=Path("/home/ubuntu/playground/assets/skies"))
    parser.add_argument("--randomize-seed", type=int, default=0)
    args = parser.parse_args()
    args.source, args.raw, args.skies = args.source.resolve(), args.raw.resolve(), args.skies.resolve()
    assert args.target > 0 and args.source.is_file() and list(args.skies.glob("*.hdr"))
    os.chdir(REPO)
    lab = Path(
        os.environ.get(
            "ARENA_ISAACLAB_SOURCE", str((REPO / ".venv").resolve().parent / "submodules/IsaacLab/source/isaaclab")
        )
    )
    assert lab.is_dir()
    os.environ.update(OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PYTHONPATH=f"{REPO}:{lab}", PYTHONUNBUFFERED="1")
    args.raw.mkdir(parents=True, exist_ok=True)
    lock = (args.raw / "render.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with args.source.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(source_sha256=digest, gripper="black_fingertips_direct")
    manifest = args.raw / "render_plan.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == config, "Resume with original settings or use a new raw directory"
    else:
        assert sorted(p.name for p in args.raw.iterdir()) == ["render.lock"], "Use an empty raw directory"
        manifest.write_text(json.dumps(config, indent=2) + "\n")
    merged = args.raw / "usdcraft_press_toaster_black_gripper.hdf5"
    assert merged.resolve() != args.source.resolve()
    if not merged.exists():
        with h5py.File(args.source, "r") as handle:
            names = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
            names = [name for name in names if handle[f"data/{name}"].attrs.get("success", False)]
        merge_exact([(args.source, names)], merged, args.target)
    half = (args.target + 1) // 2
    commands = []
    for worker, (start, end) in enumerate(((0, half), (half, args.target))):
        if start == end:
            continue
        commands.append((
            [
                str(REPO / ".venv/bin/python"),
                str(SCRIPTS / "rerender_embodiment_cameras.py"),
                "--env",
                "ur7e_usdcraft_press_toaster",
                "--hdf5",
                str(merged),
                "--device",
                "cuda:0",
                "--streams",
                "realsense_d435_rgb",
                "--demo-range",
                str(start),
                str(end),
                "--randomize",
                "--randomize-seed",
                str(args.randomize_seed),
                "--skies-dir",
                str(args.skies),
            ],
            args.raw / f"render_{worker}.log",
        ))
    assert all(code == 0 for code in run_workers(commands, stagger=20)), "Render failed; inspect logs and resume"
    cameras = Path(f"{merged}.cameras")
    assert len(list(cameras.glob("demo_*_realsense_d435_rgb.mp4"))) == args.target
    print(f"Rendered {args.target} episodes: {cameras}. LeRobot/Zarr conversion is a separate next step.")


if __name__ == "__main__":
    main()
