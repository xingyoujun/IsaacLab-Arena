# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect drawer openings with two workers, then render randomized cameras and publish training datasets."""

from __future__ import annotations

import argparse
import fcntl
import h5py
import json
import numpy as np
import os
import shutil
import subprocess
import time
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "isaaclab_arena_cumotion/scripts"
LEROBOT = REPO / "isaaclab_arena_gr00t/lerobot"
ENVIRONMENT = "ur7e_open_drawer_gpt56"
TASK = "open_drawer_gpt56_v2"


def run_workers(commands: list[tuple[list[str], Path]], stagger: float = 0) -> list[int]:
    """Run at most two children concurrently, logging separately and joining both."""
    assert len(commands) <= 2
    children = []
    try:
        for command, log in commands:
            if children:
                time.sleep(stagger)
            print(f"Starting {log}", flush=True)
            with log.open("w") as output:
                children.append(
                    subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                )
        return [child.wait() for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                child.wait()


def run(command: list[str], log: Path) -> None:
    """Run a stage and require successful exit."""
    assert run_workers([(command, log)]) == [0], f"Stage failed; see {log}"


def usable_demos(
    path: Path, drawer_key: str = "drawer_rr_gpt56", open_sign: int = -1, min_open_m: float = 0.0775
) -> list[str]:
    """Verify successful recordings end open with a closed gripper command."""
    with h5py.File(path) as handle:
        names = []
        for name in sorted(handle["data"], key=lambda name: int(name.split("_")[-1])):
            demo = handle["data"][name]
            if not demo.attrs.get("success", False):
                continue
            targets = np.asarray(demo["joint_pos_target"])
            joint = np.asarray(demo[f"states/articulation/{drawer_key}/joint_position"])
            assert targets.shape == (demo.attrs["num_samples"], 7) and np.isfinite(targets).all()
            assert targets[-1, -1] > 0.7, f"{path}/{name}: gripper released before recording stopped"
            assert open_sign * float(joint[-1].reshape(-1)[0]) > min_open_m, f"{path}/{name}: final frame not open"
            names.append(name)
        return names


def inventory(
    raw: Path, drawer_key: str = "drawer_rr_gpt56", open_sign: int = -1, min_open_m: float = 0.0775
) -> list[tuple[Path, list[str]]]:
    """List readable successes, skipping files left unreadable by crashed workers."""
    result = []
    for path in sorted(raw.glob("run_*/worker_*/demos.hdf5")):
        try:
            names = usable_demos(path, drawer_key, open_sign, min_open_m)
        except OSError as error:
            print(f"Skipping unreadable {path}: {error}", flush=True)
            continue
        if names:
            result.append((path, names))
    return result


def merge_exact(inputs: list[tuple[Path, list[str]]], destination: Path, target: int) -> None:
    """Copy exactly target episodes, retaining recorder metadata and a source manifest."""
    partial = destination.with_suffix(".partial.hdf5")
    count = total = 0
    provenance = []
    with h5py.File(partial, "w") as output:
        data = output.create_group("data")
        for path, names in inputs:
            with h5py.File(path) as source:
                if count == 0:
                    for key, value in source.attrs.items():
                        output.attrs[key] = value
                    for key, value in source["data"].attrs.items():
                        data.attrs[key] = value
                for name in names:
                    if count >= target:
                        break
                    source.copy(source[f"data/{name}"], data, name=f"demo_{count}")
                    total += int(source[f"data/{name}"].attrs["num_samples"])
                    provenance.append({"demo": f"demo_{count}", "source": str(path), "source_demo": name})
                    count += 1
        assert count == target, f"Only {count}/{target} successes available"
        data.attrs["total"] = total
    partial.replace(destination)
    destination.with_suffix(".sources.json").write_text(json.dumps(provenance, indent=2) + "\n")


def validate_lerobot(root: Path, target: int) -> dict:
    """Check episode counts, finite modalities and calibrated video dimensions."""
    import pandas as pd

    info = json.loads((root / "meta/info.json").read_text())
    parquet, videos = sorted(root.glob("data/**/*.parquet")), sorted(root.glob("videos/**/*.mp4"))
    assert info["total_episodes"] == len(parquet) == len(videos) == target
    total = 0
    for path, video in zip(parquet, videos, strict=True):
        frame = pd.read_parquet(path)
        total += len(frame)
        for column, width in (("observation.state", 7), ("action", 7), ("observation.eef_9d", 9), ("action.eef_9d", 9)):
            array = np.stack(frame[column])
            assert array.shape == (len(frame), width) and np.isfinite(array).all(), f"Invalid {path}: {column}"
        probe = json.loads(
            subprocess.check_output([
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames",
                "-of",
                "json",
                str(video),
            ])
        )["streams"][0]
        assert (probe["width"], probe["height"], probe["r_frame_rate"]) == (640, 480, "15/1")
        assert int(probe["nb_frames"]) >= len(frame)
    assert total == info["total_frames"]
    return {"episodes": target, "training_frames": total, "video_size": [640, 480], "fps": 15}


def main(
    environment: str = ENVIRONMENT,
    task: str = TASK,
    drawer_key: str = "drawer_rr_gpt56",
    open_sign: int = -1,
    min_open_m: float = 0.0775,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-seed", type=int, default=0)
    parser.add_argument("--stagger", type=float, default=20)
    parser.add_argument("--raw", type=Path, default=Path(f"/home/ubuntu/playground/datasets/rr_sim2real_raw/{task}"))
    parser.add_argument("--final", type=Path, default=Path(f"/home/ubuntu/playground/datasets/rr_sim2real/{task}"))
    parser.add_argument("--skies", type=Path, default=Path("/home/ubuntu/playground/assets/skies"))
    parser.add_argument("--dp-python", default="/home/ubuntu/miniconda3/envs/robodiff/bin/python")
    args = parser.parse_args()
    assert args.target > 0 and args.batch_size > 0 and args.max_rounds > 0 and args.stagger >= 0
    args.raw, args.final = args.raw.resolve(), args.final.resolve()
    assert args.raw != args.final
    assert list(args.skies.glob("*.hdr")), f"No HDRI skies in {args.skies}"
    assert Path(args.dp_python).is_file()
    os.chdir(REPO)
    isaaclab_source = REPO / ".venv"
    isaaclab_source = isaaclab_source.resolve().parent / "submodules/IsaacLab/source/isaaclab"
    isaaclab_source = Path(os.environ.get("ARENA_ISAACLAB_SOURCE", str(isaaclab_source)))
    assert isaaclab_source.is_dir(), f"Missing native Isaac Lab source: {isaaclab_source}"
    os.environ.update(
        OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PYTHONPATH=f"{REPO}:{isaaclab_source}", PYTHONUNBUFFERED="1"
    )
    args.raw.mkdir(parents=True, exist_ok=True)
    lock = (args.raw / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(environment=environment, stop_after_pull=True, randomize=True)
    if environment != ENVIRONMENT:
        config.update(drawer_key=drawer_key, open_sign=open_sign, min_open_m=min_open_m)
    manifest = args.raw / "pipeline.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == config, "Resume with the original pipeline settings"
    else:
        assert not args.final.exists(), f"Output already exists: {args.final}"
        manifest.write_text(json.dumps(config, indent=2) + "\n")
    logs = args.raw / "logs"
    logs.mkdir(exist_ok=True)
    python = str(REPO / ".venv/bin/python")
    common = ["--env", environment, "--headless", "--device", "cuda:0"]
    merged = args.raw / f"{task}.hdf5"
    if not merged.exists():
        inputs = inventory(args.raw, drawer_key, open_sign, min_open_m)
        have = sum(len(names) for _, names in inputs)
        next_round = max((int(p.name.split("_")[1]) for p in args.raw.glob("run_*")), default=-1) + 1
        for round_index in range(next_round, args.max_rounds):
            print(f"[collect] {have}/{args.target} successes", flush=True)
            if have >= args.target:
                break
            remaining = args.target - have
            first = min(args.batch_size, (remaining + 1) // 2)
            second = min(args.batch_size, remaining - first)
            commands = []
            for worker, count in enumerate((first, second)):
                if not count:
                    continue
                directory = args.raw / f"run_{round_index:04d}" / f"worker_{worker}"
                directory.mkdir(parents=True)
                seed = args.seed + round_index * 2 + worker
                command = [
                    python,
                    str(SCRIPTS / "ur7e_open_drawer_cumotion.py"),
                    *common,
                    "--stop-after-pull",
                    "--num-demos",
                    str(count),
                    "--seed",
                    str(seed),
                    "--record-dir",
                    str(directory),
                    "--dataset-name",
                    "demos",
                ]
                (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                commands.append((command, directory / "collection.log"))
            print(f"[collect] round {round_index}, exits {run_workers(commands, args.stagger)}", flush=True)
            inputs = inventory(args.raw, drawer_key, open_sign, min_open_m)
            have = sum(len(names) for _, names in inputs)
        assert have >= args.target, f"Collected {have}/{args.target}; refusing to publish a partial dataset"
        merge_exact(inputs, merged, args.target)
    assert len(usable_demos(merged, drawer_key, open_sign, min_open_m)) == args.target
    print(f"[render] {args.target} openings; randomized rendering with two workers", flush=True)
    half = (args.target + 1) // 2
    commands = []
    for worker, (start, end) in enumerate(((0, half), (half, args.target))):
        if start == end:
            continue
        command = [
            python,
            str(SCRIPTS / "rerender_embodiment_cameras.py"),
            *common,
            "--hdf5",
            str(merged),
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
        ]
        commands.append((command, logs / f"render_{worker}.log"))
    sidecars = Path(f"{merged}.cameras")
    renders_complete = all(
        (sidecars / f"demo_{i}_realsense_d435_rgb.mp4").is_file()
        and (sidecars / f"demo_{i}_randomization.json").is_file()
        for i in range(args.target)
    )
    if not renders_complete:
        assert all(
            code == 0 for code in run_workers(commands, args.stagger)
        ), f"Rendering failed; resume using {manifest}"
    assert len(list(sidecars.glob("demo_*_realsense_d435_rgb.mp4"))) == args.target
    assert len(list(sidecars.glob("demo_*_randomization.json"))) == args.target

    converted = args.raw / task / "lerobot"
    marker = args.raw / "converted.json"
    if not marker.exists():
        if converted.exists():
            converted.rename(converted.with_name(f"lerobot_incomplete_{time.time_ns()}"))
        conversion = yaml.safe_load((LEROBOT / "config/ur7e_open_drawer_config.yaml").read_text())
        conversion.update(data_root=str(args.raw), hdf5_name=merged.name)
        yaml_path = args.raw / "lerobot_config.yaml"
        yaml_path.write_text(yaml.safe_dump(conversion, sort_keys=False))
        run([python, str(LEROBOT / "convert_hdf5_to_lerobot.py"), "--yaml_file", str(yaml_path)], logs / "convert.log")
        run(
            [python, str(LEROBOT / "add_ur7e_eef_9d.py"), "--hdf5", str(merged), "--lerobot", str(converted)],
            logs / "eef.log",
        )
        marker.write_text(json.dumps(validate_lerobot(converted, args.target), indent=2) + "\n")
    if not args.final.exists():
        args.final.parent.mkdir(parents=True, exist_ok=True)
        staging = args.final.with_name(f"{args.final.name}.staging_{time.time_ns()}")
        shutil.copytree(converted, staging)
        staging.rename(args.final)
    result = validate_lerobot(args.final, args.target)
    zarr = args.final.with_name(f"{args.final.name}_dp.zarr")
    if not (args.raw / "zarr.done").exists():
        if zarr.exists():
            zarr.rename(zarr.with_name(f"{zarr.name}.incomplete_{time.time_ns()}"))
        run(
            [
                args.dp_python,
                str(LEROBOT / "lerobot_to_diffusion_policy_zarr.py"),
                "--lerobot",
                str(args.final),
                "--out",
                str(zarr),
                "--dp-repo",
                os.environ.get("ARENA_DP_REPO", "/home/ubuntu/code/diffusion_policy"),
            ],
            logs / "zarr.log",
        )
        validation = (
            "import os,sys,zarr;"
            " sys.path.insert(0,os.environ.get('ARENA_DP_REPO','/home/ubuntu/code/diffusion_policy')); from"
            " diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs; register_codecs();"
            " r=zarr.open(sys.argv[1],mode='r'); assert len(r['meta/episode_ends'])==int(sys.argv[2]); assert"
            " int(r['meta/episode_ends'][-1])==int(sys.argv[3]); assert r['data/action'].shape==(int(sys.argv[3]),10);"
            " assert r['data/camera_0'].shape==(int(sys.argv[3]),240,320,3)"
        )
        run(
            [args.dp_python, "-c", validation, str(zarr), str(args.target), str(result["training_frames"])],
            logs / "zarr_check.log",
        )
        (args.raw / "zarr.done").touch()

    import imageio.v2 as iio

    videos = sorted(args.final.glob("videos/**/*.mp4"))
    rows = []
    for video in videos[:: max(1, len(videos) // 12)][:12]:
        reader = iio.get_reader(video)
        count = reader.count_frames()
        rows.append(np.concatenate([reader.get_data(i)[::2, ::2] for i in (0, count // 2, count - 1)], axis=1))
        reader.close()
    preview = args.final.with_name(f"{args.final.name}_preview.png")
    iio.imwrite(preview, np.concatenate(rows, axis=0))
    result.update(lerobot=str(args.final), zarr=str(zarr), preview=str(preview))
    (args.raw / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
