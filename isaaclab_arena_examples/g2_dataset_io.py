# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compact G2 raw validation and LeRobot v2.1 metadata helpers (no simulation imports)."""

import hashlib
import json
import random
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


def sample_bowls(seed, noise):
    """Sample reproducible independent XY offsets, rejecting overlaps and table-edge violations."""
    assert 0 <= noise <= 0.05, "Use an XY perturbation between 0 and 5 cm"
    rng = random.Random(seed)
    nominal = ((-0.12, -0.24), (-0.10, 0.0), (-0.12, 0.24))
    for _ in range(1000):
        pairs = [(x + rng.uniform(-noise, noise), y + rng.uniform(-noise, noise)) for x, y in nominal]
        inside = all(abs(x) + 0.079 < 0.30 and abs(y) + 0.079 < 0.60 for x, y in pairs)
        separated = all(
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= 0.18**2 for i, a in enumerate(pairs) for b in pairs[i + 1 :]
        )
        if inside and separated:
            return [value for pair in pairs for value in pair]
    raise RuntimeError("Could not sample a non-overlapping bowl layout")


def read_core(path):
    """Validate a successful raw episode and load its small time-aligned training arrays."""
    import h5py
    import numpy as np

    report = json.loads((Path(path).parent / "report.json").read_text())
    assert len(report["attempts"]) == 1 and report["attempts"][0]["success"], "Raw attempt did not succeed"
    with h5py.File(path, "r") as dataset:
        episode = dataset["data/demo_0"]
        actions = episode["actions"][:].astype(np.float32)
        arrays = {f"observation.{key}": value[:].astype(np.float32) for key, value in episode["core"].items()}
        assert "camera_obs" not in episode, "Permanent raw must not contain RGB arrays"
        assert actions.ndim == 2 and actions.shape[1] == 16 and len(actions) > 30
        for key, array in arrays.items():
            assert len(array) == len(actions) and np.isfinite(array).all(), f"Invalid {key}"
        assert np.isfinite(actions).all()
        for key in ("observation.eef_pose", "observation.eef_pose_world"):
            assert arrays[key].shape[1] == 14
            for start in (3, 10):
                assert np.allclose(np.linalg.norm(arrays[key][:, start : start + 4], axis=1), 1, atol=1e-3)
        # Tight final physical stability check, independent of the recorded success flag.
        final = []
        for name in ("bowl_2", "bowl_1", "bowl_3"):
            positions = episode[f"states/rigid_object/{name}/root_pose"][-10:, :3]
            assert np.max(np.ptp(positions, axis=0)) < 0.005, "Final stack is still moving"
            final.append(positions[-1])
        assert final[0][2] < final[1][2] < final[2][2]
        assert max(np.linalg.norm(pos[:2] - final[0][:2]) for pos in final[1:]) < 0.05
        arrays["observation.state"] = np.concatenate(
            (arrays["observation.joint_position"], arrays["observation.eef_pose"]), axis=1
        )
        arrays["action"] = actions
    return arrays, report


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
