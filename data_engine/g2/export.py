# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export a successful native G2 recording and its validated three-camera replay locally."""

import h5py
import json
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path

from data_engine.g2.collection.dataset_io import CAMERAS, array_stats, atomic_json, file_digest, write_parquet
from data_engine.g2.collection.metadata import ImageStats, write_metadata
from data_engine.g2.collection.validation import audit_raw
from data_engine.recording.alignment import pre_step_states, validate_video


def export(raw_dir, render_dir, destination, task_id="clean_workcell_table"):
    """Publish an atomic local episode only after raw, source pairing and all video frames pass."""
    import imageio.v2 as imageio

    assert not destination.exists(), f"Refusing to replace a dataset: {destination}"
    raw = raw_dir / "episodes.hdf5"
    raw_audit = audit_raw(raw_dir, task_id)
    assert raw_audit["raw_audit_pass"] and raw_audit["task_success"]
    replay = json.loads((render_dir / "render_report.json").read_text())
    assert replay["validation_passed"] and not replay["diagnostic_only"]
    assert replay["observation_alignment"] == "pre_step" and replay["source_episode"] == "demo_0"
    assert replay["source_hdf5_sha256"] == file_digest(raw), "Video and raw source differ"
    with h5py.File(raw, "r") as file:
        demo = file["data/demo_0"]
        assert bool(demo.attrs["success"])
        pre_step_states(demo)
        metadata = json.loads(file["data"].attrs["env_args"])
        arrays = {f"observation.{name}": values[:].astype(np.float32) for name, values in demo["core"].items()}
        arrays["observation.state"] = np.concatenate(
            [arrays["observation.joint_position"], arrays["observation.eef_pose"]], axis=1
        )
        arrays["action"] = demo["actions"][:].astype(np.float32)
    count, fps = len(arrays["action"]), replay["control_fps"]
    assert replay["frames"] == count and replay["source_frame_indices"] == list(range(count))
    assert arrays["action"].shape == (count, 16)
    assert all(len(array) == count and np.isfinite(array).all() for array in arrays.values())
    for name, shape in CAMERAS.items():
        validate_video(render_dir / f"{name}_camera.mp4", count, fps, shape)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}_", dir=destination.parent))
    # Interrupted staging remains inspectable; the final dataset name is created only on success.
    data_path = staging / "data/chunk-000/episode_000000.parquet"
    data_path.parent.mkdir(parents=True)
    write_parquet(data_path, arrays, 0, 0, fps)
    stats = {name: array_stats(array.astype(np.float64)) for name, array in arrays.items()}
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
    for name in CAMERAS:
        key = f"observation.images.{name}"
        source = render_dir / f"{name}_camera.mp4"
        target = staging / f"videos/chunk-000/{key}/episode_000000.mp4"
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        image_stats = ImageStats()
        with imageio.get_reader(source) as reader:
            for frame in reader:
                image_stats.add(frame)
        assert image_stats.frames == count
        stats[key] = image_stats.result()
    entry = {"episode_index": 0, "length": count, "stats": stats}
    write_metadata(staging, [entry], metadata["joint_names"], fps, metadata["task"]["instruction"])
    atomic_json(staging / "meta/replay/episode_000000.json", replay)
    atomic_json(
        staging / "meta/raw_provenance.json",
        {
            "source_sha256": file_digest(raw),
            "source_episode": "demo_0",
            "env_args": metadata,
            "raw_audit": raw_audit,
            "visual_review_required": True,
            "artifacts_sha256": {
                str(p.relative_to(staging)): file_digest(p) for p in staging.rglob("*") if p.is_file()
            },
        },
    )
    staging.rename(destination)
    print(f"EXPORTED_G2_EPISODE {destination} frames={count} cameras=3 fps={fps}")


if __name__ == "__main__":
    raise SystemExit("Use data_engine/g2/cli.py export --run-dir RUN")
