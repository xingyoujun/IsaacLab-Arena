# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resume all eight black-fingertip datasets; default is a read-only plan, --run executes."""

from __future__ import annotations

import argparse
import fcntl
import h5py
import hashlib
import json
import numpy as np
import os
import shutil
import subprocess
import time
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path

from collect_ur7e_open_drawer_gpt56 import LEROBOT, REPO, SCRIPTS, merge_exact, run, run_workers, validate_lerobot

DATASETS = Path("/home/ubuntu/playground/datasets")
ASSETS = Path("/home/ubuntu/playground/rr_ur")
APPEARANCE = {"gripper": "black_fingertips_direct", "diffuse_color": [0.015] * 3, "roughness": 0.5}
DEFAULT_WORK = DATASETS / "rr_sim2real_aux/black_gripper_all"


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    key: str
    sign: int
    rest: float
    threshold: float
    asset: str
    source: str | None = None
    joint_index: int = 0

    @property
    def environment(self):
        return f"ur7e_{self.name}"


CASES = (
    Case(
        "usdcraft_open_drawer",
        "drawer",
        "drawer_rr",
        1,
        0,
        0.075,
        "usdcraft_drawer_arena.usda",
        "open_drawer/open_drawer_trunc.hdf5",
    ),
    Case(
        "usdcraft_press_toaster",
        "toast",
        "toaster_rr",
        1,
        0,
        0.03525,
        "usdcraft_toast_arena.usda",
        "press_toaster/press_toaster_v2.hdf5",
    ),
    Case(
        "miniworkflow_gptsol_open_drawer",
        "drawer",
        "drawer_rr_gpt56",
        -1,
        0,
        0.0775,
        "miniworkflow_gptsol_drawer_arena.usda",
        "open_drawer_gpt56_v2/open_drawer_gpt56_v2.hdf5",
    ),
    Case(
        "articraft_open_drawer",
        "drawer",
        "drawer_articraft",
        1,
        0,
        0.08415,
        "articraft_drawer/articraft_drawer.usd",
        "open_drawer_articraft_v2/open_drawer_articraft_v2.hdf5",
    ),
    Case(
        "miniworkflow_astra_open_drawer",
        "drawer",
        "miniworkflow_astra_drawer",
        -1,
        0,
        0.075,
        "miniworkflow_astra_drawer_arena.usda",
    ),
    Case(
        "articraft_press_toaster",
        "toast",
        "articraft_toast",
        1,
        0,
        0.03579849936,
        "articraft_toast/articraft_toast.usd",
    ),
    Case(
        "miniworkflow_gptsol_press_toaster",
        "toast",
        "miniworkflow_gptsol_toast",
        -1,
        0.004,
        0.04575,
        "miniworkflow_gptsol_toast_arena.usda",
    ),
    Case(
        "miniworkflow_astra_press_toaster",
        "toast",
        "miniworkflow_astra_toast",
        -1,
        0,
        0.04575,
        "miniworkflow_astra_toast_arena.usda",
    ),
)

KNOB_CASES = tuple(
    Case(
        f"{method}_turn_toaster_knob", "knob", f"{method}_toast_knob", 0, 0, float(np.deg2rad(2)), asset, joint_index=1
    )
    for method, asset in (
        ("usdcraft", "usdcraft_toast_arena.usda"),
        ("articraft", "articraft_toast/articraft_toast.usd"),
        ("miniworkflow_gptsol", "miniworkflow_gptsol_toast_arena.usda"),
        ("miniworkflow_astra", "miniworkflow_astra_toast_arena.usda"),
    )
)


def write_json(path: Path, value) -> None:
    """Atomically checkpoint a stage without marking unfinished work complete."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprints() -> dict[str, str]:
    """Pin task code and all canonical object payloads to prevent mixed resume results."""
    paths = set((REPO / "isaaclab_arena_environments").glob("ur7e_*.py"))
    paths.update((REPO / "isaaclab_arena/embodiments/ur7e").glob("*.py"))
    paths.update(
        SCRIPTS / name
        for name in (
            "prepare_rr_sim2real_all.py",
            "collect_ur7e_open_drawer_gpt56.py",
            "ur7e_open_drawer_cumotion.py",
            "ur7e_press_toaster_cumotion.py",
            "ur7e_turn_toaster_knob_cumotion.py",
            "rerender_embodiment_cameras.py",
        )
    )
    paths.update(
        LEROBOT / name
        for name in (
            "convert_hdf5_to_lerobot.py",
            "add_ur7e_eef_9d.py",
            "lerobot_to_diffusion_policy_zarr.py",
            "config/ur7e_open_drawer_config.yaml",
            "config/ur7e_press_toaster_config.yaml",
        )
    )
    paths.update(ASSETS / case.asset for case in CASES)
    paths.add(ASSETS / "boxx.usdc")
    paths.add(REPO / "tools/rr_sim2real/asset_overlays/boxx_support.usda")
    paths.update(
        ASSETS / name
        for name in (
            "usdcraft_drawer.usdc",
            "usdcraft_toast.usdc",
            "miniworkflow_gptsol_drawer.usd",
            "miniworkflow_astra_drawer.usd",
            "miniworkflow_gptsol_toast.usd",
            "miniworkflow_astra_toast.usd",
        )
    )
    robot = ASSETS / "ur7e_usd/ur7e_gripper/Collected_ur7e_gripper"
    assert (robot / "ur7e_gripper.usd").is_file(), "Missing UR7e asset; do not collect with UR5e fallback"
    paths.update(p for p in robot.rglob("*") if p.is_file() and p.suffix in (".usd", ".usda", ".usdc"))
    return {str(p): digest(p) for p in sorted(paths)}


def good_demos(path: Path, case: Case, check_start: bool) -> list[str]:
    """Reject invalid/partial episodes, wrong end states and residual-velocity resets."""
    names = []
    with h5py.File(path, "r") as handle:
        for name in sorted(handle["data"], key=lambda n: int(n.split("_")[-1])):
            d = handle["data"][name]
            try:
                count = int(d.attrs["num_samples"])
                q = np.asarray(d[f"states/articulation/{case.key}/joint_position"])
                target = np.asarray(d["joint_pos_target"])
                assert d.attrs.get("success", False) and count > 1
                assert target.shape == (count, 7) and q.shape[0] == count
                assert np.isfinite(target).all() and np.isfinite(q).all()
                assert target[-1, -1] > 0.7
                travel = q[:, case.joint_index] - case.rest
                travel = abs(travel) if case.sign == 0 else case.sign * travel
                assert travel[-1] > case.threshold
                if case.kind == "knob":
                    assert np.flatnonzero(travel > case.threshold).tolist() == [count - 1]
                    for support in ("box_support_0", "box_support_1"):
                        assert f"states/rigid_object/{support}/root_pose" in d
                if check_start:
                    v = np.asarray(d[f"states/articulation/{case.key}/joint_velocity"])
                    assert abs(float(q[0, case.joint_index]) - case.rest) < 0.003
                    assert abs(float(v[0, case.joint_index])) < 1e-4
                names.append(name)
            except (AssertionError, KeyError, IndexError, ValueError) as error:
                print(f"Rejected {path}/{name}: {type(error).__name__}", flush=True)
    return names


def collect(case: Case, work: Path, args, python: str) -> Path:
    """Reuse source states or collect missing successes with bounded worker concurrency."""
    merged = work / f"{case.name}.hdf5"
    if merged.exists():
        assert len(good_demos(merged, case, not case.source)) == args.target
        return merged
    if case.source:
        source = DATASETS / "rr_sim2real_raw" / case.source
        names = good_demos(source, case, False)
        assert len(names) >= args.target
        merge_exact([(source, names)], merged, args.target)
        return merged

    def inventory():
        result = []
        pattern = "run_*/worker_*/recording/demo.hdf5" if case.kind == "knob" else "run_*/worker_*/demos.hdf5"
        for path in sorted(work.glob(pattern)):
            try:
                result.append((path, good_demos(path, case, True)))
            except OSError as error:
                print(f"Unreadable interrupted worker file (retained): {path}: {error}", flush=True)
        return result

    inputs = inventory()
    have = sum(len(names) for _, names in inputs)
    start = max((int(p.name.split("_")[1]) for p in work.glob("run_*")), default=-1) + 1
    driver = {
        "drawer": "ur7e_open_drawer_cumotion.py",
        "toast": "ur7e_press_toaster_cumotion.py",
        "knob": "ur7e_turn_toaster_knob_cumotion.py",
    }[case.kind]
    stalled_rounds = 0
    for index in range(start, args.max_rounds):
        if have >= args.target:
            break
        print(f"[{case.name}] collect {have}/{args.target}; round {index}", flush=True)
        commands = []
        remaining = args.target - have
        for worker in range(args.workers):
            count = min(args.batch_size, (remaining + args.workers - worker - 1) // (args.workers - worker))
            remaining -= count
            if not count:
                continue
            directory = work / f"run_{index:04d}/worker_{worker}"
            directory.mkdir(parents=True)
            command = [
                python,
                str(SCRIPTS / driver),
                "--env",
                case.environment,
                "--device",
                "cuda:0",
                "--num-demos",
                str(count),
                "--seed",
                str(args.seed + index * (3 if case.kind == "knob" else args.workers) + worker),
                "--record-dir",
                str(directory),
                "--dataset-name",
                "demos",
            ]
            if case.kind == "drawer":
                command.append("--stop-after-pull")
            if case.kind == "knob":
                command = command[: command.index("--record-dir")] + [
                    "--output",
                    str(directory / "recording"),
                    "--record-demo",
                    "--no-video",
                    "--randomize-pose",
                    "--init-joint-std",
                    "0.03",
                    "--audit-self-collision",
                ]
            write_json(directory / "command.json", command)
            commands.append((command, directory / "collect.log"))
        print(f"Worker exits: {run_workers(commands, args.stagger)}", flush=True)
        inputs = inventory()
        updated = sum(len(names) for _, names in inputs)
        if updated == have:
            stalled_rounds += 1
            if case.kind != "knob" or stalled_rounds >= 5:
                raise RuntimeError(f"{case.name}: {stalled_rounds} rounds without progress; inspect worker logs")
            print(f"[{case.name}] no valid new demos; retrying fresh seeds ({stalled_rounds}/5)", flush=True)
        else:
            stalled_rounds = 0
        have = updated
    assert have >= args.target, f"{case.name}: only {have}/{args.target}; increase --max-rounds if needed"
    merge_exact(inputs, merged, args.target)
    return merged


def render(case: Case, work: Path, merged: Path, args, python: str) -> Path:
    """Resume randomized replay in fresh, appearance-checked camera sidecars."""
    sidecars = Path(f"{merged}.cameras")
    marker = sidecars / "render_appearance.json"
    if marker.exists():
        assert json.loads(marker.read_text()) == APPEARANCE, "Old gripper cache; use another work root"
    commands = []
    for worker, indices in enumerate(np.array_split(np.arange(args.target), args.render_workers or args.workers)):
        if len(indices):
            commands.append((
                [
                    python,
                    str(SCRIPTS / "rerender_embodiment_cameras.py"),
                    "--env",
                    case.environment,
                    "--embodiment",
                    "ur7e_robotiq_joint_pos",
                    "--device",
                    "cuda:0",
                    "--hdf5",
                    str(merged),
                    "--streams",
                    "realsense_d435_rgb",
                    "--demo-range",
                    str(indices[0]),
                    str(indices[-1] + 1),
                    "--randomize",
                    "--randomize-seed",
                    str(args.randomize_seed),
                    "--skies-dir",
                    str(args.skies),
                ],
                work / "logs" / f"render_{worker}_{time.time_ns()}.log",
            ))
    complete = marker.exists() and all(
        (sidecars / f"demo_{i}_realsense_d435_rgb.mp4").exists()
        and (sidecars / f"demo_{i}_randomization.json").exists()
        for i in range(args.target)
    )
    if not complete:
        assert all(code == 0 for code in run_workers(commands, args.stagger)), f"{case.name}: render failed"
    assert json.loads(marker.read_text()) == APPEARANCE
    for i in range(args.target):
        assert (sidecars / f"demo_{i}_realsense_d435_rgb.mp4").stat().st_size > 0
        assert json.loads((sidecars / f"demo_{i}_randomization.json").read_text())
    return sidecars


def convert(case: Case, work: Path, merged: Path, args, python: str) -> tuple[Path, Path, dict]:
    """Build validated LeRobot and DP Zarr in staging, never inside the HF root."""
    converted = work / case.name / "lerobot"
    zarr = work / f"{case.name}_dp.zarr"
    logs = work / "logs"
    if not (work / "converted.json").exists():
        if converted.exists():
            converted.rename(converted.with_name(f"lerobot_interrupted_{time.time_ns()}"))
        template = "ur7e_open_drawer_config.yaml" if case.kind == "drawer" else "ur7e_press_toaster_config.yaml"
        config = yaml.safe_load((LEROBOT / "config" / template).read_text())
        config.update(data_root=str(work), hdf5_name=merged.name)
        if case.kind == "knob":
            config["language_instruction"] = "Grasp the toaster dial and rotate it slightly."
        config_path = work / "lerobot_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        run(
            [python, str(LEROBOT / "convert_hdf5_to_lerobot.py"), "--yaml_file", str(config_path)], logs / "convert.log"
        )
        run(
            [python, str(LEROBOT / "add_ur7e_eef_9d.py"), "--hdf5", str(merged), "--lerobot", str(converted)],
            logs / "eef.log",
        )
        result = validate_lerobot(converted, args.target)
        write_json(converted / "meta/render_appearance.json", APPEARANCE)
        write_json(work / "converted.json", result)
    result = validate_lerobot(converted, args.target)
    if not (work / "zarr.json").exists():
        if zarr.exists():
            zarr.rename(zarr.with_name(f"{zarr.name}.interrupted_{time.time_ns()}"))
        run(
            [
                args.dp_python,
                str(LEROBOT / "lerobot_to_diffusion_policy_zarr.py"),
                "--lerobot",
                str(converted),
                "--out",
                str(zarr),
                "--dp-repo",
                str(args.dp_repo),
            ],
            logs / "zarr.log",
        )
        check = (
            "import sys,zarr,numpy as np; sys.path.insert(0,sys.argv[1]);"
            " from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs; register_codecs();"
            " r=zarr.open(sys.argv[2],mode='r'); n=int(sys.argv[3]); t=int(sys.argv[4]);"
            " e=r['meta/episode_ends'][:]; assert len(e)==n and e[-1]==t and np.all(np.diff(e)>0);"
            " assert r['data/action'].shape==(t,10) and np.isfinite(r['data/action'][:]).all();"
            " assert r['data/camera_0'].shape==(t,240,320,3);"
            " assert r['data/camera_0'][0].shape==(240,320,3) and r['data/camera_0'][-1].shape==(240,320,3)"
        )
        run(
            [
                args.dp_python,
                "-c",
                check,
                str(args.dp_repo),
                str(zarr),
                str(args.target),
                str(result["training_frames"]),
            ],
            logs / "zarr_check.log",
        )
        write_json(work / "zarr.json", result)
    assert zarr.is_dir()
    return converted, zarr, result


def publish_pair(work: Path, final_root: Path, case: Case, sources: tuple[Path, Path], result: dict) -> None:
    """Publish with a recoverable journal; archive each old dataset outside the HF root."""
    journal = work / "publication.json"
    if journal.exists():
        transaction = json.loads(journal.read_text())
    else:
        transaction = {"id": f"{case.name}_{time.time_ns()}", "done": False}
        write_json(journal, transaction)
    identity = transaction["id"]
    archive = work.parent / "previous_datasets" / identity
    archive.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    for source, name in zip(sources, (case.name, f"{case.name}_dp.zarr"), strict=True):
        destination, backup = final_root / name, archive / name
        stamp = destination / "rr_sim2real_build.json"
        if stamp.exists() and json.loads(stamp.read_text()).get("build_id") == identity:
            continue
        stage = work / f"publish_{name}"
        if stage.exists() and not (stage / "rr_sim2real_build.json").exists():
            stage.rename(stage.with_name(f"{stage.name}.interrupted_{time.time_ns()}"))
        if not stage.exists():
            shutil.copytree(source, stage)
            write_json(
                stage / "rr_sim2real_build.json",
                dict(result, build_id=identity, case=case.name, appearance=APPEARANCE, randomize=True),
            )
        if destination.exists():
            assert not destination.is_symlink() and not backup.exists(), f"Unexpected output at {destination}"
            destination.rename(backup)
        stage.rename(destination)
    transaction["done"] = True
    write_json(journal, transaction)


def update_readme(root: Path) -> None:
    lines = [
        "# RR sim2real datasets",
        "",
        "Canonical names: `<method>_<task>` and `<method>_<task>_dp.zarr`.",
        "All exports use randomized backgrounds/materials and up to six distractors.",
        "LeRobot videos: 640x480 at 15 fps; DP images: 320x240. Raw, smoke, logs and backups are outside this root.",
        "",
        "| Dataset | Episodes | Frames | Appearance |",
        "| --- | ---: | ---: | --- |",
    ]
    for case in (*CASES, *KNOB_CASES):
        dataset = root / case.name
        info = dataset / "meta/info.json"
        if not info.exists():
            continue
        values = json.loads(info.read_text())
        marker = dataset / "meta/render_appearance.json"
        appearance = (
            "black fingertips"
            if marker.exists() and json.loads(marker.read_text()) == APPEARANCE
            else "legacy; replay pending"
        )
        lines.append(f"| `{case.name}` | {values['total_episodes']} | {values['total_frames']} | {appearance} |")
    lines += [
        "",
        "Each dataset has a sibling `_dp.zarr`. Upload only canonical datasets, their Zarr siblings and this README.",
        (
            "No checkpoints, raw HDF5, previews or smoke tests belong here. Do not train/upload during dataset"
            " replacement."
        ),
        (
            "Historical raw/checkpoint names remain unchanged for provenance; `codebase_version: v2.1` is the LeRobot"
            " format."
        ),
        "",
    ]
    temporary = root / "README.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(root / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute; without this only print/validate the plan")
    parser.add_argument("--only", nargs="+", choices=[case.name for case in (*CASES, *KNOB_CASES)])
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--render-workers", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-seed", type=int, default=0)
    parser.add_argument("--stagger", type=float, default=20)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--final-root", type=Path, default=DATASETS / "rr_sim2real")
    parser.add_argument("--skies", type=Path, default=Path("/home/ubuntu/playground/assets/skies"))
    parser.add_argument("--dp-python", default="/home/ubuntu/miniconda3/envs/robodiff/bin/python")
    parser.add_argument("--dp-repo", type=Path, default=Path("/home/ubuntu/code/diffusion_policy"))
    parser.add_argument("--min-free-gb", type=float, default=40)
    args = parser.parse_args()
    assert 0 < args.target <= 200 and args.batch_size > 0 and args.max_rounds > 0 and args.stagger >= 0
    for name in ("work_root", "final_root", "skies", "dp_repo"):
        setattr(args, name, getattr(args, name).resolve())
    assert args.target == 200 or args.work_root != DEFAULT_WORK, "Smoke runs require a separate --work-root"
    assert args.work_root != args.final_root and args.final_root not in args.work_root.parents
    assert args.work_root not in args.final_root.parents
    assert args.work_root != Path("/") and args.work_root != Path.home()
    assert Path(args.dp_python).is_file() and (args.dp_repo / "diffusion_policy").is_dir()
    assert list(args.skies.glob("*.hdr")) and shutil.which("ffprobe")
    lab = Path(
        os.environ.get(
            "ARENA_ISAACLAB_SOURCE", str((REPO / ".venv").resolve().parent / "submodules/IsaacLab/source/isaaclab")
        )
    )
    assert lab.is_dir()
    os.chdir(REPO)
    os.environ.update(
        OMNI_KIT_ACCEPT_EULA="YES",
        ACCEPT_EULA="Y",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{REPO}:{lab}",
        ARENA_DP_REPO=str(args.dp_repo),
    )
    python = str(REPO / ".venv/bin/python")
    selected = [case for case in (*CASES, *KNOB_CASES) if case.name in args.only] if args.only else list(CASES)
    overrides = {
        "ARENA_DRAWER_USD": "usdcraft_drawer_arena.usda",
        "ARENA_DRAWER_GPT56_USD": "miniworkflow_gptsol_drawer_arena.usda",
        "ARENA_DRAWER_ASTRA_USD": "miniworkflow_astra_drawer_arena.usda",
        "ARENA_DRAWER_ARTICRAFT_USD": "articraft_drawer/articraft_drawer.usd",
        "ARENA_TOASTER_USD": "usdcraft_toast_arena.usda",
        "ARENA_TOASTER_GPTSOL_USD": "miniworkflow_gptsol_toast_arena.usda",
        "ARENA_TOASTER_ASTRA_USD": "miniworkflow_astra_toast_arena.usda",
        "ARENA_TOASTER_ARTICRAFT_USD": "articraft_toast/articraft_toast.usd",
        "ARENA_UR7E_USD": "ur7e_usd/ur7e_gripper/Collected_ur7e_gripper/ur7e_gripper.usd",
    }
    for variable, default in overrides.items():
        assert (
            variable not in os.environ or Path(os.environ[variable]).resolve() == (ASSETS / default).resolve()
        ), f"Unexpected {variable}; unset asset overrides for this matched comparison"
    print(
        f"{len(selected)} cases; {sum(bool(c.source) for c in selected)} replay existing, "
        f"{sum(not c.source for c in selected)} new collections; {args.target} demos each; "
        f"{args.workers} collection / {args.render_workers or args.workers} render workers"
    )
    if args.target != 200:
        print("SMOKE ONLY: no publication to the final dataset root")
    for case in selected:
        assert (ASSETS / case.asset).is_file()
        if case.source:
            assert (DATASETS / "rr_sim2real_raw" / case.source).is_file()
        print(f"  {case.name}: {'replay ' + case.source if case.source else 'collect -> replay'} -> LeRobot -> DP Zarr")
    print(f"Work/logs: {args.work_root}\nFinal: {args.final_root}\nBackups: {args.work_root}/previous_datasets")
    if not args.run:
        return
    pinned = fingerprints()
    pinned.update({str(p): digest(p) for p in sorted(args.skies.glob("*.hdr"))})
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    assert not active, f"GPU has active processes ({active}); stop competing jobs yourself before launch"
    assert shutil.disk_usage(DATASETS).free >= args.min_free_gb * 2**30, "Insufficient free disk space"
    args.work_root.mkdir(parents=True, exist_ok=True)
    lock = (args.work_root / "master.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    final_lock = (DATASETS / "rr_sim2real_aux/publication.lock").open("a")
    fcntl.flock(final_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for case in selected:
        started = time.time()
        work = args.work_root / case.name
        work.mkdir(exist_ok=True)
        (work / "logs").mkdir(exist_ok=True)
        config = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in ("run", "only", "max_rounds", "min_free_gb")
        }
        if case.kind == "knob":
            for execution_option in ("workers", "render_workers", "stagger"):
                config.pop(execution_option, None)
        config.update(case=asdict(case), appearance=APPEARANCE, fingerprints=pinned)
        if case.source:
            config["source_sha256"] = digest(DATASETS / "rr_sim2real_raw" / case.source)
        plan = work / "plan.json"
        if plan.exists():
            assert json.loads(plan.read_text()) == config, f"Settings/asset changed: {plan}; use a fresh work root"
        else:
            write_json(plan, config)
        if (work / "complete.json").exists():
            if args.target == 200:
                transaction = json.loads((work / "publication.json").read_text())
                for name in (case.name, f"{case.name}_dp.zarr"):
                    stamp = args.final_root / name / "rr_sim2real_build.json"
                    assert stamp.exists() and json.loads(stamp.read_text())["build_id"] == transaction["id"]
            print(f"[{case.name}] complete, skipping", flush=True)
            continue
        assert shutil.disk_usage(args.work_root).free >= args.min_free_gb * 2**30
        merged = collect(case, work, args, python)
        assert len(good_demos(merged, case, not case.source)) == args.target
        render(case, work, merged, args, python)
        lerobot, zarr, result = convert(case, work, merged, args, python)
        if args.target == 200:
            publish_pair(work, args.final_root, case, (lerobot, zarr), result)
            update_readme(args.final_root)
        result.update(case=case.name, elapsed_this_run_s=round(time.time() - started), published=args.target == 200)
        write_json(work / "complete.json", result)
        print(f"DONE {result}", flush=True)
    if args.target == 200:
        update_readme(args.final_root)
    print("ALL SELECTED CASES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
