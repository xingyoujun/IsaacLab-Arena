# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared transition indexing and provenance for G2 and Pine WM recordings."""

import hashlib
import json
import numpy as np
import subprocess
from pathlib import Path

REPLAY_DT = 1e-6
SCHEMA = "arena.transitions.v1"


def pre_step_states(demo):
    """Return T pre-action states from initial_state and T post-action states, without dropping actions."""
    count = len(demo["actions"])
    assert count > 0 and np.isfinite(demo["actions"][:]).all(), "Empty or nonfinite actions"
    states = {}
    for kind, assets in demo["states"].items():
        states[kind] = {}
        for name, fields in assets.items():
            states[kind][name] = {}
            for field, dataset in fields.items():
                initial = np.asarray(demo[f"initial_state/{kind}/{name}/{field}"])
                values = np.asarray(dataset)
                assert len(values) == count, (kind, name, field, len(values), count)
                assert initial.shape == (1, *values.shape[1:]), (initial.shape, values.shape)
                assert np.isfinite(values).all() and np.isfinite(initial).all()
                states[kind][name][field] = np.concatenate([initial, values[:-1]], axis=0)
    expected = states["articulation"]["robot"]["joint_position"]
    for key in ("core/joint_position", "obs/robot_joint_pos"):
        if key in demo:
            assert demo[key].shape == expected.shape, f"Pre-action joint shape mismatch: {key}"
            assert np.allclose(demo[key][:], expected, atol=1e-6, rtol=0), f"Pre-action joint mismatch: {key}"
    return states


def transition_metadata(embodiment, cameras, fps=15.0):
    """Describe transition timing independently of a robot's action and observation dimensions."""
    root = Path(__file__).resolve().parents[2]
    from isaaclab_arena.assets.usdcraft_scene import bundle_root

    manifest = bundle_root() / "manifest.json"
    provenance = None
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        provenance = {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "local_extension": data.get("local_extension"),
            "published_release": json.loads((root / "tools/usdcraft_scene/release.json").read_text()),
            "release_lineage": data.get("release_lineage"),
        }
        provenance["matches_published_release"] = (
            provenance["manifest_sha256"] == provenance["published_release"]["manifest_sha256"]
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    source_hashes = {}
    for name in sorted(set(changed + untracked)):
        path = root / name
        if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".usda"}:
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_changes_sha256": source_hashes,
        "schema": SCHEMA,
        "embodiment": embodiment,
        "control_fps": fps,
        "observation_alignment": "pre_step",
        "state_sequence": "initial_state_then_previous_post_step_state",
        "transition": "observation[t], image[t], action[t], next_state[t]",
        "timestamp_rule": "source_frame_index / control_fps",
        "quaternion_order": "xyzw",
        "camera_streams": cameras,
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "assets": provenance,
    }


def validate_video(path, count, fps, shape=None):
    """Count decoded frames and check dimensions and frame rate; never silently truncate."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    assert int(stream["nb_read_frames"]) == count, (path, stream, count)
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    assert abs(numerator / denominator - fps) < 1e-6
    if shape is not None:
        assert (stream["height"], stream["width"]) == tuple(shape)
    return stream


class ReplayDrift:
    """Measure restored poses from the physics backend, rather than write-through caches."""

    def __init__(self):
        self.joint = self.position = self.rotation = 0.0
        self.joint_errors = {}
        self.joint_names = {}

    def update(self, env, states, index):
        import warp as wp

        for kind, assets in states.items():
            for name, fields in assets.items():
                view = env.scene[name].root_view
                transforms = view.get_root_transforms() if kind == "articulation" else view.get_transforms()
                pose = wp.to_torch(transforms)[0].cpu().numpy()
                reference = fields["root_pose"][index]
                self.position = max(self.position, float(np.max(np.abs(pose[:3] - reference[:3]))))
                q, r = pose[3:].astype(float), reference[3:].astype(float)
                cosine = abs(q @ r) / (np.linalg.norm(q) * np.linalg.norm(r))
                self.rotation = max(self.rotation, float(2 * np.arccos(np.clip(cosine, 0, 1))))
                if kind == "articulation":
                    joints = wp.to_torch(view.get_dof_positions())[0].cpu().numpy()
                    if joints.size:
                        self.joint_names[name] = env.scene[name].joint_names
                        errors = np.abs(joints - fields["joint_position"][index])
                        self.joint_errors[name] = np.maximum(self.joint_errors.get(name, np.zeros_like(errors)), errors)
                        self.joint = max(self.joint, float(errors.max()))

    def report(self):
        """Apply Pine WM's separate camera-chain, compliant-finger and object bounds."""
        camera_joint = 0.0
        object_joint = 0.0
        for asset, errors in self.joint_errors.items():
            if asset == "robot":
                for name, error in zip(self.joint_names[asset], errors):
                    # Include G2 body/head and both arms; exclude the compliant gripper linkages.
                    is_g2 = any("_arm_" in joint for joint in self.joint_names[asset])
                    ur_arm = {
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",
                    }
                    camera_bearing = ("gripper" not in name and "chassis" not in name) if is_g2 else name in ur_arm
                    if camera_bearing:
                        camera_joint = max(camera_joint, float(error))
            else:
                object_joint = max(object_joint, float(errors.max()))
        return {
            "replay_validation_profile": "bounded_camera_chain_v1",
            "joint_coordinate_errors": {name: errors.tolist() for name, errors in self.joint_errors.items()},
            "max_joint_error_rad": self.joint,
            "max_camera_chain_joint_error_rad": camera_joint,
            "max_articulated_object_coordinate_error": object_joint,
            "max_root_position_error_m": self.position,
            "max_root_rotation_error_rad": self.rotation,
            "state_error_source": "physics_backend",
            "state_validation_passed": (
                camera_joint < 1e-3
                and self.joint < 5e-3
                and object_joint < 5e-4
                and self.position < 5e-4
                and self.rotation < 5e-3
            ),
            "limits": {
                "camera_chain_joint_rad": 1e-3,
                "all_joint_coordinates": 5e-3,
                "object_joint_coordinates": 5e-4,
                "position_m": 5e-4,
                "rotation_rad": 5e-3,
            },
        }


def recorded_contract(demo):
    """Read an explicit recording contract; legacy files remain on their historical export path."""
    metadata = json.loads(demo.parent.attrs.get("env_args", "{}"))
    contract = metadata.get("collection_contract")
    if contract is not None:
        assert contract["schema"] == SCHEMA and contract["observation_alignment"] == "pre_step"
    return contract


def export_rows(demo):
    """Keep every validated transition for new recordings; retain legacy compatibility explicitly."""
    if recorded_contract(demo) is None:
        return slice(None, -1)
    assert bool(
        demo.attrs.get("success", False)
    ), "Failed/unknown episodes are diagnostic, not successful training data"
    pre_step_states(demo)
    return slice(None)
