#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate twelve RR checkpoints serially, with two independent workers per case.

Run with robodiff Python; --plan-only writes the inventory without starting CUDA.
Each worker owns an independent policy server, RNG, simulator and output folder.
Execution errors stop the batch and preserve partial outputs; task failures count
normally. SIGTERM stops only subprocess groups launched by this orchestrator.
"""

import argparse
import csv
import fcntl
import json
import math
import os
import pickle
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import zmq

ROOT = Path(__file__).resolve().parents[2]
DP = Path("/home/ubuntu/code/diffusion_policy")
PYTHON = "/home/ubuntu/miniconda3/envs/robodiff/bin/python"
DATA = Path("/home/ubuntu/playground/datasets/rr_sim2real")
METHODS = ["usdcraft", "articraft", "miniworkflow_gptsol", "miniworkflow_astra"]
TASKS = ["open_drawer", "press_toaster", "turn_toaster_knob"]
CHILDREN = []


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def stop_children():
    for child in CHILDREN:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while any(p.poll() is None for p in CHILDREN) and time.monotonic() < deadline:
        time.sleep(0.5)
    for child in CHILDREN:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    CHILDREN.clear()


def launch(command, logfile, env):
    with logfile.open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    CHILDREN.append(process)
    print(f"START pid={process.pid} log={logfile}", flush=True)
    return process


def wait_ready(server, port):
    context = zmq.Context()
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            assert server.poll() is None, "Policy server exited; inspect server.log"
            connection = context.socket(zmq.REQ)
            connection.setsockopt(zmq.LINGER, 0)
            try:
                connection.connect(f"tcp://127.0.0.1:{port}")
                connection.send(pickle.dumps({"cmd": "meta"}))
                if connection.poll(1000):
                    meta = pickle.loads(connection.recv())
                    assert meta["n_obs_steps"] == 2 and meta["n_action_steps"] == 8, meta
                    assert meta["shape_meta"]["action"]["shape"] == [10], meta
                    return meta
            finally:
                connection.close()
        raise TimeoutError("Policy server readiness timed out")
    finally:
        context.term()


def inventory():
    cases = []
    for task in TASKS:
        for method in METHODS:
            case = f"{method}_{task}"
            checkpoint = DP / "ckpts" / f"{case}.ckpt"
            assert checkpoint.is_file(), checkpoint
            info = json.loads((DATA / case / "meta/info.json").read_text())
            demos = [json.loads(line) for line in (DATA / case / "meta/episodes.jsonl").read_text().splitlines()]
            longest = max(int(demo["length"]) for demo in demos)
            fps = float(info["fps"])
            seconds = 30.0 if task == "open_drawer" else math.ceil(longest / fps * 1.5)
            if task == "open_drawer":
                key = dict(
                    zip(METHODS, ["drawer_rr", "drawer_articraft", "drawer_rr_gpt56", "miniworkflow_astra_drawer"])
                )[method]
            elif task == "press_toaster":
                key = "toaster_rr" if method == "usdcraft" else f"{method}_toast"
            else:
                key = f"{method}_toast_knob"
            cases.append(
                dict(
                    case=case,
                    task=task,
                    checkpoint=str(checkpoint),
                    checkpoint_bytes=checkpoint.stat().st_size,
                    checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                    dataset_max_frames=longest,
                    dataset_fps=fps,
                    dataset_max_seconds=longest / fps,
                    episode_length_s=seconds,
                    audit_object=key,
                    status="pending",
                )
            )
    return cases


def collect_worker(directory, expected_episodes=10):
    result = json.loads((directory / "experiment/arena_experiment_result.json").read_text())
    episodes = []
    for run in result["runs"].values():
        assert run["status"] == "completed", run
        for rebuild in run["rebuilds"]:
            episodes.extend(rebuild["episodes"])
    assert len(episodes) == expected_episodes, len(episodes)
    assert (directory / "experiment/index.html").is_file()
    assert list((directory / "experiment").rglob("episode_results*.jsonl"))
    videos = sorted((directory / "experiment").rglob("*.mp4"))
    assert len(videos) == 2 * expected_episodes, f"Expected two camera videos per episode: {len(videos)}"
    for video in videos:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)], text=True
        )
        assert float(json.loads(probe)["format"]["duration"]) > 0, video
    return episodes, [str(path) for path in videos]


def execute_case(case, output, episodes_per_worker=10, num_workers=2, record_trajectories=False):
    assert shutil.disk_usage(output).free > 20 * 1024**3, "Less than 20 GiB disk space remaining"
    checkpoint = Path(case["checkpoint"])
    assert checkpoint.stat().st_size == case["checkpoint_bytes"]
    assert checkpoint.stat().st_mtime_ns == case["checkpoint_mtime_ns"], "Checkpoint changed during evaluation"
    workers = []
    for worker, seed in enumerate(42 + 1000 * index for index in range(num_workers)):
        port = 5760 + worker
        with socket.socket() as check:
            check.bind(("127.0.0.1", port))
        directory = output / case["case"] / f"worker{worker}"
        directory.mkdir(parents=True)
        env = os.environ.copy()
        env.update(
            OMNI_KIT_ACCEPT_EULA="YES",
            ACCEPT_EULA="Y",
            PYTHONUNBUFFERED="1",
            ARENA_RR_DP_AUDIT="1",
            OMP_NUM_THREADS="4",
            PYTHONPATH=f"{ROOT}:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab",
        )
        server = launch(
            [
                PYTHON,
                str(DP / "serve_ur7e_drawer_policy.py"),
                "--ckpt",
                case["checkpoint"],
                "--port",
                str(port),
                "--num-inference-steps",
                "100",
            ],
            directory / "server.log",
            env,
        )
        write_json(directory / "policy_meta.json", wait_ready(server, port))
        environment = dict(
            type="ur7e_" + case["case"],
            embodiment="ur7e_robotiq_joint_pos",
            enable_cameras=True,
            episode_length_s=case["episode_length_s"],
        )
        if case["task"] == "open_drawer":
            environment["randomize_drawer_pose"] = True
        else:
            environment.update(randomize_toaster_pose=True, pedestal=case["task"] == "press_toaster")
        config = dict(
            shared=dict(
                environment=environment,
                environment_builder=dict(
                    num_envs=1, seed=seed, device="cuda:0", record_trajectories=record_trajectories
                ),
                policy=dict(type="ur7e_dp_remote", host="127.0.0.1", port=port, audit_object=case["audit_object"]),
                rollout_limit=dict(num_episodes=episodes_per_worker),
            ),
            runs={f"{output.name}_{case['case']}_w{worker}": {}},
        )
        # JSON is valid YAML; .yaml selects the typed experiment loader.
        write_json(directory / "experiment.yaml", config)
        simulation = launch(
            [
                str(ROOT / ".venv/bin/python"),
                "isaaclab_arena/evaluation/experiment_runner.py",
                "--experiment_config",
                str(directory / "experiment.yaml"),
                "--experiment_output_directory",
                str(directory / "experiment"),
                "--viz",
                "none",
                "--enable_cameras",
                "--record_camera_video",
            ],
            directory / "eval.log",
            env,
        )
        write_json(
            directory / "processes.json",
            dict(server_pid=server.pid, simulation_pid=simulation.pid, seed=seed, port=port),
        )
        workers.append((directory, server, simulation))
        if worker + 1 < num_workers:
            time.sleep(20)
    deadline = time.monotonic() + 3 * 3600
    while any(sim.poll() is None for _, _, sim in workers):
        for directory, server, sim in workers:
            assert sim.poll() in (None, 0), f"Simulation failed: {directory}"
            assert server.poll() is None, f"Server failed: {directory}"
        assert time.monotonic() < deadline, "Case exceeded three-hour wall-time safety limit"
        time.sleep(5)
    episodes, videos = [], []
    for worker, (directory, _, simulation) in enumerate(workers):
        assert simulation.returncode == 0, f"Simulation failed: {directory}"
        worker_episodes, worker_videos = collect_worker(directory, episodes_per_worker)
        if record_trajectories:
            assert list((directory / "experiment").rglob("dataset_*_rebuild0.hdf5")), directory
        episodes.extend(dict(record, worker=worker) for record in worker_episodes)
        videos.extend(worker_videos)
    success = sum(bool(episode["success"]) for episode in episodes)
    write_json(
        output / case["case"] / "results.json",
        dict(
            case=case,
            episodes=episodes,
            successes=success,
            total=len(episodes),
            success_rate=success / len(episodes),
            videos=videos,
        ),
    )
    case.update(status="completed", successes=success, episodes=len(episodes), success_rate=success / len(episodes))
    stop_children()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--only", nargs="+", choices=[f"{method}_{task}" for task in TASKS for method in METHODS])
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--episodes-per-worker", type=int, default=10)
    parser.add_argument("--record-trajectories", action="store_true")
    args = parser.parse_args()
    assert args.episodes_per_worker > 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = inventory()
    if args.only:
        cases = [case for case in cases if case["case"] in args.only]
    if args.plan_only:
        print(json.dumps(cases, indent=2))
        return
    lock = (ROOT / "outputs/.rr_dp_eval_all.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (output / "manifest.json").exists(), "Use a fresh output directory"
    manifest = dict(
        status="running",
        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        episodes_per_case=args.episodes_per_worker * args.workers,
        workers=args.workers,
        record_trajectories=args.record_trajectories,
        inference_steps=100,
        timeout_rule="Drawer: 30s; press/knob: ceil(1.5 * longest dataset metadata length / fps)",
        appearance="Existing black gripper; randomized object pose; standard evaluation background",
        cases=cases,
    )

    def interrupted(signum, frame):
        raise RuntimeError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for case in cases:
            case["status"] = "running"
            write_json(output / "manifest.json", manifest)
            print(f"CASE {case['case']} timeout={case['episode_length_s']}s", flush=True)
            execute_case(case, output, args.episodes_per_worker, args.workers, args.record_trajectories)
            with (output / "summary.csv").open("w") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["case", "status", "successes", "episodes", "success_rate", "episode_length_s"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(cases)
            write_json(output / "manifest.json", manifest)
            print(f"COMPLETED {case['case']}: {case['successes']}/{case['episodes']}", flush=True)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        for case in cases:
            if case["status"] == "running":
                case["status"] = "execution_failed"
        raise
    finally:
        stop_children()
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
