# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G2 LeRobot v2.1 arrays and metadata helpers without simulation imports."""

import hashlib
import json
from pathlib import Path

TASK = "Stack bowl_1 into bowl_2, then stack bowl_3 on top of bowl_1."
CAMERAS = {"head": (400, 640), "left_wrist": (528, 640), "right_wrist": (528, 640)}
ACTION_NAMES = (
    [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
    + ["right_gripper_open_close"]
    + [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
    + ["left_gripper_open_close"]
)
POSE_NAMES = [f"{side}_{axis}" for side in ("right", "left") for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]


def atomic_json(path, value):
    """Atomically replace a JSON checkpoint in its own directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, indent=2, allow_nan=False))
    staging.replace(path)


def file_digest(path):
    """Get a SHA256 digest without loading a whole artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_schema(joint_names):
    """Build explicit G2 feature names without copying Agibot's padded joint dimensions."""
    names = {
        "observation.state": joint_names + POSE_NAMES,
        "observation.joint_position": joint_names,
        "observation.joint_velocity": joint_names,
        "observation.eef_pose": POSE_NAMES,
        "observation.eef_pose_world": POSE_NAMES,
        "action": ACTION_NAMES,
    }
    features = {key: {"dtype": "float32", "shape": [len(value)], "names": value} for key, value in names.items()}
    for key in ("episode_index", "frame_index", "index", "task_index", "timestamp"):
        features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
    for camera, (height, width) in CAMERAS.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.height": height,
                "video.width": width,
                "video.fps": 15.0,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features


def write_parquet(path, arrays, episode_index, frame_offset, fps):
    """Write a LeRobot-compatible Arrow table with Hugging Face feature metadata."""
    import numpy as np

    import pyarrow as pa
    import pyarrow.parquet as pq

    count = len(arrays["action"])
    columns, hf_features = {}, {}
    for key, value in arrays.items():
        columns[key] = pa.array(value.tolist(), type=pa.list_(pa.float32(), value.shape[1]))
        hf_features[key] = {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": value.shape[1],
            "_type": "Sequence",
        }
    scalars = {
        "episode_index": np.full(count, episode_index, dtype=np.int64),
        "frame_index": np.arange(count, dtype=np.int64),
        "index": np.arange(frame_offset, frame_offset + count, dtype=np.int64),
        "task_index": np.zeros(count, dtype=np.int64),
        "timestamp": (np.arange(count) / fps).astype(np.float32),
    }
    for key, value in scalars.items():
        columns[key] = pa.array(value)
        hf_features[key] = {"dtype": str(value.dtype), "_type": "Value"}
    table = pa.table(columns).replace_schema_metadata(
        {b"huggingface": json.dumps({"info": {"features": hf_features}}).encode()}
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def array_stats(array):
    """Compute per-dimension LeRobot episode statistics."""
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [len(array)],
    }
