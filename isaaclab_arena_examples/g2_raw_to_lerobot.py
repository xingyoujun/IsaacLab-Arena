# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline native three-camera rendering and transactional LeRobot v2.1 export for compact G2 raw."""

import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

from g2_dataset_io import CAMERAS, TASK, array_stats, atomic_json, feature_schema, file_digest, read_core, write_parquet

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def write_metadata(root, entries, joint_names, fps, task=TASK):
    """Rebuild LeRobot metadata from committed contiguous episodes only."""
    import numpy as np

    assert [entry["episode_index"] for entry in entries] == list(range(len(entries)))
    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    episodes = [{"episode_index": e["episode_index"], "tasks": [task], "length": e["length"]} for e in entries]
    stats = [{"episode_index": e["episode_index"], "stats": e["stats"]} for e in entries]
    for name, records in (
        ("episodes", episodes),
        ("episodes_stats", stats),
        ("tasks", [{"task_index": 0, "task": task}]),
    ):
        temporary = meta / f"{name}.jsonl.tmp"
        temporary.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
        temporary.replace(meta / f"{name}.jsonl")
    aggregate = {}
    for key in entries[0]["stats"]:
        values = [e["stats"][key] for e in entries]
        weights = np.array([v["count"][0] for v in values], dtype=np.float64)
        weights /= weights.sum()
        means = np.array([v["mean"] for v in values])
        stds = np.array([v["std"] for v in values])
        mean = np.average(means, axis=0, weights=weights)
        variance = np.average(stds**2 + (means - mean) ** 2, axis=0, weights=weights)
        aggregate[key] = {
            "min": np.min([v["min"] for v in values], axis=0).tolist(),
            "max": np.max([v["max"] for v in values], axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [sum(v["count"][0] for v in values)],
        }
    atomic_json(meta / "stats.json", aggregate)
    atomic_json(
        meta / "modality.json",
        {
            "state": {
                "joints": {"original_key": "observation.state", "start": 0, "end": len(joint_names)},
                "right_eef": {
                    "original_key": "observation.state",
                    "start": len(joint_names),
                    "end": len(joint_names) + 7,
                },
                "left_eef": {
                    "original_key": "observation.state",
                    "start": len(joint_names) + 7,
                    "end": len(joint_names) + 14,
                },
            },
            "action": {
                "right_arm": {"start": 0, "end": 7},
                "right_gripper": {"start": 7, "end": 8},
                "left_arm": {"start": 8, "end": 15},
                "left_gripper": {"start": 15, "end": 16},
            },
            "video": {name: {"original_key": f"observation.images.{name}"} for name in CAMERAS},
            "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        },
    )
    atomic_json(
        meta / "info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "agibot_g2_omnipicker",
            "fps": fps,
            "total_episodes": len(entries),
            "total_frames": sum(e["length"] for e in entries),
            "total_tasks": 1,
            "total_videos": len(entries) * 3,
            "chunks_size": 1000,
            "total_chunks": (len(entries) + 999) // 1000,
            "splits": {"train": f"0:{len(entries)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feature_schema(joint_names),
        },
    )
    atomic_json(
        meta / "g2_conventions.json",
        {
            "joint_names": joint_names,
            "joint_units": "radians and radians/second",
            "eef_order": "right xyz qx qy qz qw, left xyz qx qy qz qw",
            "eef_pose_frame": "robot root/base",
            "eef_pose_world_frame": "simulation world; tabletop z=0",
            "position_units": "metres",
            "action": "absolute right joints[7], binary gripper, left joints[7], binary gripper",
            "gripper_action": "+1 open, -1 close; measured gripper joints remain in joint_position",
            "time_alignment": "state[t] and all three images[t] precede action[t]; timestamp=t/fps",
            "rendering": "all RGB re-rendered from recorded physical states, never re-simulated actions",
            "image_statistics": "all frames, every 16th pixel in each spatial dimension, RGB normalized to [0,1]",
        },
    )


class ImageStats:
    """Accumulate actual sampled RGB statistics without retaining image tensors."""

    def __init__(self):
        import numpy as np

        self.total = np.zeros(3)
        self.squares = np.zeros(3)
        self.minimum = np.ones(3)
        self.maximum = np.zeros(3)
        self.pixels = 0
        self.frames = 0

    def add(self, frame):
        import numpy as np

        pixels = frame[::16, ::16].reshape(-1, 3).astype(np.float64) / 255
        self.total += pixels.sum(0)
        self.squares += (pixels**2).sum(0)
        self.minimum = np.minimum(self.minimum, pixels.min(0))
        self.maximum = np.maximum(self.maximum, pixels.max(0))
        self.pixels += len(pixels)
        self.frames += 1

    def result(self):
        import numpy as np

        mean = self.total / self.pixels
        std = np.sqrt(np.maximum(self.squares / self.pixels - mean**2, 0))
        return {
            **{
                key: value.reshape(3, 1, 1).tolist()
                for key, value in (
                    ("min", self.minimum),
                    ("max", self.maximum),
                    ("mean", mean),
                    ("std", std),
                )
            },
            "count": [self.frames],
        }


def render_episode(base, path, arrays, staging, fps):
    """Render all native cameras from pre-action scene states and verify TCP alignment."""
    import h5py
    import numpy as np
    import torch

    import imageio.v2 as imageio

    count = len(arrays["action"])
    stats = {name: ImageStats() for name in CAMERAS}
    max_eef_error = 0.0
    with ExitStack() as stack:
        dataset = stack.enter_context(h5py.File(path, "r"))
        episode = dataset["data/demo_0"]
        writers = {
            name: stack.enter_context(
                imageio.get_writer(
                    str(staging / f"{name}.mp4"),
                    fps=fps,
                    codec="libx264",
                    quality=8,
                    output_params=["-threads", "2", "-movflags", "+faststart"],
                )
            )
            for name in CAMERAS
        }
        env_ids = torch.tensor([0], dtype=torch.int32, device=base.device)
        for index in range(count):
            group = episode["initial_state"] if index == 0 else episode["states"]
            source_index = 0 if index == 0 else index - 1
            state = {
                category: {
                    name: {
                        key: torch.tensor(value[source_index], device=base.device).unsqueeze(0)
                        for key, value in fields.items()
                    }
                    for name, fields in objects.items()
                }
                for category, objects in group.items()
            }
            base.scene.reset_to(state, env_ids=env_ids, is_relative=True)
            base.sim.forward()
            base.sim.render_context.reset_scene_state_cadence()
            base.scene.update(base.step_dt)
            base.sim.render()
            base.sim.render()
            images = base.observation_manager.compute()["camera_obs"]
            for name in CAMERAS:
                frame = images[f"{name}_camera_rgb"][0].cpu().numpy()
                assert frame.dtype == np.uint8 and frame.shape == (*CAMERAS[name], 3)
                writers[name].append_data(frame)
                stats[name].add(frame)
            for offset, sensor_name in ((0, "ee_frame"), (7, "left_ee_frame")):
                actual = base.scene[sensor_name].data.target_pos_w.torch[0, 0].cpu().numpy()
                error = float(np.linalg.norm(actual - arrays["observation.eef_pose_world"][index, offset : offset + 3]))
                max_eef_error = max(error, max_eef_error)
                assert error < 0.001, f"Replay/core EEF timestamps disagree at frame {index}: {error} m"
            if index % 200 == 0:
                print(f"RENDER frame={index}/{count}", flush=True)
    # Decode every frame: file existence alone does not prove a complete encoder output.
    for name in CAMERAS:
        reader = imageio.get_reader(str(staging / f"{name}.mp4"))
        assert reader.count_frames() == count and abs(reader.get_meta_data()["fps"] - fps) < 1e-6
        reader.close()
    return {f"observation.images.{name}": value.result() for name, value in stats.items()}, max_eef_error


def commit_episode(root, work, base, raw, episode_index, frame_offset):
    """Commit one validated parquet/video bundle; retain raw and any interrupted staging output."""
    path = Path(raw["raw_dir"]) / "episodes.hdf5"
    assert file_digest(path) == raw["sha256"], "Raw changed after successful-core validation"
    arrays, report = read_core(path)
    assert abs(report["step_dt"] - base.step_dt) < 1e-8
    count = len(arrays["action"])
    fps = 1.0 / base.step_dt
    staging_root = work / "render_staging"
    staging_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"episode_{episode_index:06d}_", dir=staging_root))
    stats, error = render_episode(base, path, arrays, staging, fps)
    stats.update({key: array_stats(value.astype("float64")) for key, value in arrays.items()})
    write_parquet(staging / "episode.parquet", arrays, episode_index, frame_offset, fps)
    chunk = episode_index // 1000
    destinations = {"episode.parquet": f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"}
    destinations.update({
        f"{name}.mp4": f"videos/chunk-{chunk:03d}/observation.images.{name}/episode_{episode_index:06d}.mp4"
        for name in CAMERAS
    })
    hashes = {}
    for source, destination in destinations.items():
        target = root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        (staging / source).replace(target)
        hashes[destination] = file_digest(target)
    entry = {
        "episode_index": episode_index,
        "length": count,
        "stats": stats,
        "raw_sha256": raw["sha256"],
        "raw_dir": raw["raw_dir"],
        "seed": raw["seed"],
        "object_positions": {
            key: value
            for key, value in report["configuration"].items()
            if key in ("bowl_positions", "peg_x", "peg_y", "sleeve_x", "sleeve_y")
        },
        "replay_max_eef_error_m": error,
        "artifacts": hashes,
        "joint_names": report["joint_names"],
    }
    if "bowl_positions" in report["configuration"]:
        entry["bowl_positions"] = report["configuration"]["bowl_positions"]
    atomic_json(work / "episodes" / f"episode_{episode_index:06d}.json", entry)
    staging.rmdir()  # Only the empty staging directory created by this function.
    return entry


def run(args):
    from isaaclab_arena.embodiments.g2.g2 import G2CameraCfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.g2_stack_bowls_environment import (
        G2StackBowlsEnvironment,
        G2StackBowlsEnvironmentCfg,
    )

    config = json.loads((args.work_dir / "collection_config.json").read_text())
    manifest = json.loads((args.work_dir / "collection_manifest.json").read_text())
    raw_episodes = manifest["successes"][: args.demos]
    assert len(raw_episodes) == args.demos
    task = TASK
    if config["task"] == "peg_into_sleeve":
        from isaaclab_arena_environments.g2_sleeve_environment import G2SleeveEnvironment, G2SleeveEnvironmentCfg

        task = "Pick up the cylindrical peg and insert it fully into the fixed upright sleeve."
        description = G2SleeveEnvironment().build(
            G2SleeveEnvironmentCfg(enable_cameras=True, table_height_m=config["table_height_m"])
        )
    else:
        description = G2StackBowlsEnvironment().build(
            G2StackBowlsEnvironmentCfg(
                enable_cameras=True,
                hdr=None,
                table_height_m=config["table_height_m"],
            )
        )
    if config["task"] == "peg_into_sleeve":
        # Offline replay restores every pose without stepping physics. A dynamic
        # replay body also accepts the recorded zero velocity during reset_to.
        description.scene.assets["sleeve"].object_cfg.spawn.rigid_props.kinematic_enabled = False
    description.embodiment.camera_config = G2CameraCfg()
    args.num_envs = 1
    env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
    entries = []
    try:
        env.reset()
        for index, raw in enumerate(raw_episodes):
            marker = args.work_dir / "episodes" / f"episode_{index:06d}.json"
            if marker.exists():
                entry = json.loads(marker.read_text())
                assert entry["raw_sha256"] == raw["sha256"], "Episode mapping changed"
                for name, digest in entry["artifacts"].items():
                    assert file_digest(args.root / name) == digest, f"Committed artifact changed: {name}"
                print(f"SKIP committed episode={index}", flush=True)
            else:
                print(f"EXPORT episode={index}/{len(raw_episodes)} raw={raw['raw_dir']}", flush=True)
                entry = commit_episode(
                    args.root, args.work_dir, env.unwrapped, raw, index, sum(e["length"] for e in entries)
                )
            entries.append(entry)
            write_metadata(args.root, entries, entry["joint_names"], 1.0 / env.unwrapped.step_dt, task=task)
        print(f"LEROBOT_COMPLETE episodes={len(entries)}", flush=True)
    finally:
        env.close()


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--headless", action="store_true", help="Run without a viewer (also the GA default)")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--demos", type=int, required=True)
    args = parser.parse_args()
    assert args.enable_cameras
    shutil.copyfile(__file__, args.work_dir / "logs" / "renderer_source.py")
    with SimulationAppContext(args):
        run(args)


if __name__ == "__main__":
    main()
