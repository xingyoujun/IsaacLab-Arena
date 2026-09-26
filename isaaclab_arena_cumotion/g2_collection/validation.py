# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independently validate G2 recordings without starting a simulator."""

import h5py
import hashlib
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from isaaclab_arena.recording.alignment import pre_step_states


def audit_raw(raw_dir, task_id):
    """Require real successful transitions and task-specific terminal geometry."""
    if task_id == "clean_workcell_table":
        from isaaclab_arena_cumotion.g2_collection.workcell.audit import audit

        return audit(raw_dir)
    report = json.loads((raw_dir / "report.json").read_text())
    assert report["task_success"] and report["planner"] == "native_cumotion"
    with h5py.File(raw_dir / "episodes.hdf5") as file:
        assert list(file["data"]) == ["demo_0"]
        demo = file["data/demo_0"]
        assert bool(demo.attrs["success"])
        pre_step_states(demo)
        count = len(demo["actions"])
        assert count == report["frames"] and demo["actions"].shape == (count, 16)
        assert np.isfinite(demo["actions"][:]).all()
        for key, width in [("joint_position", 46), ("joint_velocity", 46), ("eef_pose", 14), ("eef_pose_world", 14)]:
            assert demo[f"core/{key}"].shape == (count, width)
            assert np.isfinite(demo[f"core/{key}"][:]).all()
        objects = {}
        for group in demo["states"].values():
            for name, fields in group.items():
                if name != "robot":
                    objects[name] = {key: value[:] for key, value in fields.items()}
        if task_id == "stack_bowls":
            names = ["bowl_2", "bowl_1", "bowl_3"]
            positions = np.stack([objects[n]["root_pose"][-10:, :3] for n in names], axis=1)
            delta = np.diff(positions, axis=1)
            assert np.all(np.linalg.norm(delta[:, :, :2], axis=-1) < 0.04)
            assert np.all((delta[:, :, 2] > 0.005) & (delta[:, :, 2] < 0.06))
            for name in names:
                states = objects[name]
                axis = Rotation.from_quat(states["root_pose"][-10:, 3:7]).apply([0, 0, 1])
                assert np.all(axis[:, 2] > np.cos(np.deg2rad(7 if name == "bowl_2" else 45)))
                assert np.all(np.linalg.norm(states["root_velocity"][-10:, :3], axis=1) < 0.1)
            for name in ("bowl_1", "bowl_3"):
                assert np.ptp(objects[name]["root_pose"][:, 2]) > 0.1
        else:
            assert task_id == "peg_into_sleeve"
            peg, sleeve = objects["peg"], objects["sleeve"]
            relative = Rotation.from_quat(sleeve["root_pose"][-15:, 3:7]).inv()
            delta = relative.apply(peg["root_pose"][-15:, :3] - sleeve["root_pose"][-15:, :3])
            axis = relative.apply(Rotation.from_quat(peg["root_pose"][-15:, 3:7]).apply([0, 0, 1]))
            assert np.all(np.linalg.norm(delta[:, :2], axis=1) <= 0.001)
            assert np.all(np.abs(delta[:, 2]) <= 0.003)
            assert np.all(axis[:, 2] >= np.cos(np.deg2rad(1)))
            assert np.all(np.linalg.norm(peg["root_velocity"][-15:, :3], axis=1) < 0.03)
            assert np.max(np.linalg.norm(sleeve["root_pose"][:, :3] - sleeve["root_pose"][0, :3], axis=1)) < 0.001
            assert np.ptp(peg["root_pose"][:, 2]) > 0.15
        assert np.all(demo["actions"][-10:, [7, 15]] == 1), "Final grippers must be open"
    return dict(raw_audit_pass=True, task_success=True, steps=count, task_id=task_id)


def audit_run(root, official_loader=False):
    """Audit the recorded task and bind the result to its raw HDF5 bytes."""
    run = json.loads((root / "run.json").read_text())
    result = audit_raw(root / "raw", run["task_id"])
    with (root / "raw/episodes.hdf5").open("rb") as stream:
        result["raw_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    result["asset_manifest_sha256"] = run["assets"]["manifest_sha256"]
    fallbacks = []
    for stage in ("collect", "render"):
        log = root / f"{stage}.log"
        if log.is_file():
            for line in log.read_text(errors="replace").splitlines():
                if "fall back to CPU" in line:
                    fallbacks.append(dict(stage=stage, message=line))
    result["runtime_device_audit"] = dict(
        requested_device=run["device"],
        native_planner_cuda_verified=False,
        cpu_collision_fallback_detected=bool(fallbacks),
        cpu_collision_fallback_warnings=fallbacks,
        all_cuda_verified=False,
    )
    if (root / "render").exists():
        from isaaclab_arena.recording.alignment import validate_video

        replay = json.loads((root / "render/render_report.json").read_text())
        assert replay["validation_passed"] and not replay["diagnostic_only"]
        assert replay["source_hdf5_sha256"] == result["raw_sha256"]
        assert replay["frames"] == result["steps"]
        assert replay["source_frame_indices"] == list(range(result["steps"]))
        for name, shape in {"head": (400, 640), "left_wrist": (528, 640), "right_wrist": (528, 640)}.items():
            validate_video(root / f"render/{name}_camera.mp4", result["steps"], 15.0, shape)
        result["replay_passed"] = True
    if (root / "lerobot").exists():
        import pyarrow.parquet as pq

        dataset = root / "lerobot"
        info = json.loads((dataset / "meta/info.json").read_text())
        provenance = json.loads((dataset / "meta/raw_provenance.json").read_text())
        assert provenance["source_sha256"] == result["raw_sha256"]
        assert info["total_frames"] == result["steps"] and info["total_episodes"] == 1
        frame = pq.read_table(dataset / "data/chunk-000/episode_000000.parquet").to_pandas()
        with h5py.File(root / "raw/episodes.hdf5") as file:
            np.testing.assert_array_equal(np.stack(frame["action"]), file["data/demo_0/actions"][:])
        for relative, expected in provenance["artifacts_sha256"].items():
            artifact = dataset / relative
            assert not Path(relative).is_absolute() and artifact.resolve().is_relative_to(dataset.resolve())
            with artifact.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, relative
        result["export_passed"] = True
    if official_loader:
        import torch

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset("local/g2_dataset", root=root / "lerobot", video_backend="pyav")
        assert len(dataset) == result["steps"]
        indices = [0, len(dataset) // 2, len(dataset) - 1]
        for index in indices:
            row = dataset[index]
            assert row["action"].shape == (16,) and row["observation.state"].shape == (60,)
            for key, shape in {
                "head": (3, 400, 640),
                "left_wrist": (3, 528, 640),
                "right_wrist": (3, 528, 640),
            }.items():
                image = row[f"observation.images.{key}"]
                assert image.shape == shape and torch.isfinite(image).all() and image.std() > 0.01
        result["official_loader_indices"] = indices
    return result
