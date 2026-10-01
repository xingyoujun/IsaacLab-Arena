# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""LeRobot metadata and image statistics for unified G2 exports."""

import json

from data_engine.g2.collection.dataset_io import CAMERAS, TASK, atomic_json, feature_schema


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
            "rendering": "live pre-action RGB or bounded microstep replay; inspect per-episode replay manifests",
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
