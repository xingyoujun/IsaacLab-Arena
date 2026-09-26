# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject time-shifted or failed recordings and preserve valid final transitions."""

import h5py
import json
import numpy as np

import pytest

from isaaclab_arena.recording.alignment import SCHEMA, export_rows, pre_step_states


@pytest.fixture
def episode(tmp_path):
    with h5py.File(tmp_path / "raw.hdf5", "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"collection_contract": {"schema": SCHEMA, "observation_alignment": "pre_step"}}
        )
        demo = data.create_group("demo_0")
        demo.attrs["success"] = True
        demo["actions"] = np.arange(6).reshape(3, 2)
        demo["states/articulation/robot/joint_position"] = [[1.0], [2.0], [3.0]]
        demo["initial_state/articulation/robot/joint_position"] = [[0.0]]
        demo["core/joint_position"] = [[0.0], [1.0], [2.0]]
        yield demo


def test_pre_step_and_last_action(episode):
    states = pre_step_states(episode)
    np.testing.assert_array_equal(states["articulation"]["robot"]["joint_position"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(episode["actions"][export_rows(episode)][-1], [4, 5])


def test_shifted_observations_rejected(episode):
    episode["core/joint_position"][:] = [[1], [2], [3]]
    with pytest.raises(AssertionError, match="Pre-action joint mismatch"):
        export_rows(episode)


def test_failed_episode_not_marked_success(episode):
    episode.attrs["success"] = False
    with pytest.raises(AssertionError, match="diagnostic"):
        export_rows(episode)


def test_mismatched_state_count_rejected(episode):
    del episode["states/articulation/robot/joint_position"]
    episode["states/articulation/robot/joint_position"] = [[1], [2]]
    with pytest.raises(AssertionError):
        pre_step_states(episode)


def test_legacy_rows_preserved(episode):
    episode.parent.attrs["env_args"] = "{}"
    assert len(episode["actions"][export_rows(episode)]) == 2


def test_pine_export_preserves_all_rows_and_failure_gate(tmp_path):
    import yaml
    from pathlib import Path

    from isaaclab_arena_gr00t.lerobot.config.dataset_config import Gr00tDatasetConfig
    from isaaclab_arena_gr00t.lerobot.convert_hdf5_to_lerobot import convert_trajectory_to_df
    from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml

    with h5py.File(tmp_path / "pine.hdf5", "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"collection_contract": {"schema": SCHEMA, "observation_alignment": "pre_step"}}
        )
        demo = data.create_group("demo_0")
        demo.attrs["success"] = True
        demo["actions"] = np.zeros((3, 7))
        demo["joint_pos_target"] = np.zeros((3, 7))
        demo["obs/robot_joint_pos"] = np.zeros((3, 12))
        demo["states/articulation/robot/joint_position"] = np.zeros((3, 12))
        demo["initial_state/articulation/robot/joint_position"] = np.zeros((1, 12))
        config_data = yaml.safe_load(
            Path("isaaclab_arena_gr00t/lerobot/config/ur7e_open_drawer_config.yaml").read_text()
        )
        config_data.update(data_root=str(tmp_path), hdf5_name="pine.hdf5", sidecar_camera_streams={})
        config_path = tmp_path / "export.yaml"
        config_path.write_text(yaml.safe_dump(config_data))
        config = create_config_from_yaml(config_path, Gr00tDatasetConfig)

        result = convert_trajectory_to_df(demo, 0, 0, config)
        assert result["length"] == 3
        assert result["data"]["timestamp"].iloc[-1] == pytest.approx(2 / 15)
        demo.attrs["success"] = False
        with pytest.raises(AssertionError, match="diagnostic"):
            convert_trajectory_to_df(demo, 0, 0, config)
        data.attrs["env_args"] = "{}"
        assert convert_trajectory_to_df(demo, 0, 0, config)["length"] == 2


def test_initialization_derivative_is_traceable(episode, tmp_path):
    from tools.data_collection.derive_episode import derive

    episode.file.flush()
    source = tmp_path / "raw.hdf5"
    target = tmp_path / "settled.hdf5"
    before = source.read_bytes()
    derive(source, target, "demo_0", 1, "Exclude measured reset initialization")
    assert source.read_bytes() == before
    with h5py.File(target) as file:
        derived = file["data/demo_0"]
        assert len(derived["actions"]) == 2
        np.testing.assert_array_equal(derived["initial_state/articulation/robot/joint_position"][:], [[1]])
        assert derived.attrs["success"]
        assert json.loads(file["data"].attrs["env_args"])["derivation"]["source_frame_range"] == [1, 3]
        pre_step_states(derived)


def test_asset_lookup_does_not_import_simulation():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from isaaclab_arena.assets.g2_asset_paths import planning_path; assert 'pxr.Usd' not in"
                " sys.modules"
            ),
        ],
        check=True,
    )


def test_camera_chain_and_compliant_fingers_have_distinct_bounds():
    from isaaclab_arena.recording.alignment import ReplayDrift

    drift = ReplayDrift()
    drift.joint_names = {"robot": ["idx61_arm_r_joint1", "idx81_gripper_r_outer_joint1"]}
    drift.joint_errors = {"robot": np.array([0.0005, 0.004])}
    drift.joint = 0.004
    assert drift.report()["state_validation_passed"]
    drift.joint_errors["robot"][0] = 0.002
    assert not drift.report()["state_validation_passed"]
    drift.joint_errors["robot"][:] = [0.0005, 0.104]
    drift.joint = 0.104
    assert not drift.report()["state_validation_passed"]
