# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate one successful G2 sleeve raw episode and its aligned camera videos."""

import argparse
import json
from pathlib import Path

from g2_dataset_io import atomic_json, file_digest


def validate(directory: Path):
    """Check measured insertion, physical lift, finite aligned data, and video lengths.

    Args:
        directory: Single-attempt collector output containing report.json and episodes.hdf5.

    Returns:
        Summary of the successful recording, including its raw checksum.
    """
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation

    import imageio.v2 as imageio

    report = json.loads((directory / "report.json").read_text())
    assert len(report["attempts"]) == 1, "Validate a single-attempt recording"
    assert report["task_variant"] == "movable_peg_fixed_sleeve"
    attempt = report["attempts"][0]
    assert attempt["success"] and attempt["lift_m"] > 0.15, "Physical task execution did not succeed"
    raw = directory / "episodes.hdf5"
    with h5py.File(raw, "r") as dataset:
        episode = dataset["data/demo_0"]
        actions = episode["actions"][:]
        count = len(actions)
        assert actions.shape == (count, 16) and count > 30 and np.isfinite(actions).all()
        assert count == attempt["frames"], "Report/action frame count mismatch"
        for name, dim in (("joint_position", 46), ("joint_velocity", 46), ("eef_pose", 14), ("eef_pose_world", 14)):
            values = episode[f"core/{name}"][:]
            assert values.shape == (count, dim) and np.isfinite(values).all(), f"Invalid {name}"
            if "pose" in name:
                for start in (3, 10):
                    assert np.allclose(np.linalg.norm(values[:, start : start + 4], axis=1), 1, atol=1e-3)
        sleeve = episode["states/rigid_object/sleeve/root_pose"][:]
        peg = episode["states/rigid_object/peg/root_pose"][:]
        assert sleeve.shape == peg.shape == (count, 7)
        assert np.isfinite(sleeve).all() and np.isfinite(peg).all()
        assert np.max(np.linalg.norm(sleeve[:, :3] - sleeve[0, :3], axis=1)) < 0.001, "Fixed sleeve moved"
        assert np.max(peg[:, 2]) - peg[0, 2] > 0.15, "Recorded peg was not lifted"
        assert np.max(np.ptp(peg[-15:, :3], axis=0)) < 0.003, "Final peg is not stable"
        peg_rotation = Rotation.from_quat(peg[-15:, 3:])
        sleeve_rotation = Rotation.from_quat(sleeve[-15:, 3:])
        relative = sleeve_rotation.inv().apply(peg[-15:, :3] - sleeve[-15:, :3])
        axes = (sleeve_rotation.inv() * peg_rotation).apply(np.tile([0, 0, 1], (15, 1)))
        radial = np.linalg.norm(relative[:, :2], axis=1)
        axial = np.abs(relative[:, 2])
        angle = np.degrees(np.arccos(np.clip(axes[:, 2], -1, 1)))
        assert np.all(radial <= 0.001) and np.all(axial <= 0.003) and np.all(angle <= 1.0), "Not fully seated"
        assert np.all(actions[-15:, 7] == 1), "Right gripper was not released"
    videos = {}
    if report["configuration"]["record_video"]:
        for camera in ("overview_camera", "head_camera", "left_wrist_camera", "right_wrist_camera"):
            path = directory / f"attempt_000_{camera}.mp4"
            with imageio.get_reader(str(path)) as reader:
                frames = reader.count_frames()
                meta = reader.get_meta_data()
            assert frames == count, f"Video/action alignment mismatch: {camera}"
            assert abs(meta["fps"] - 1 / report["step_dt"]) < 0.01
            videos[camera] = {"frames": frames, "size": meta["size"], "sha256": file_digest(path)}
    summary = {
        "success": True,
        "frames": count,
        "fps": 1 / report["step_dt"],
        "raw_sha256": file_digest(raw),
        "max_final_radial_error_m": float(radial.max()),
        "max_final_axial_error_m": float(axial.max()),
        "max_final_axis_error_deg": float(angle.max()),
        "videos": videos,
    }
    atomic_json(directory / "validation.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2))


if __name__ == "__main__":
    main()
