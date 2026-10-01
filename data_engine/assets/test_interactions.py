# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Negative cases for stale metadata, moving-link composition and conditional locks."""

import hashlib
import json
import numpy as np

import pytest

from data_engine.assets.interactions import load_interactions, world_pose
from data_engine.assets.latches import next_engaged


def test_stale_sidecar_rejected(tmp_path):
    usd = tmp_path / "model.usdc"
    usd.write_bytes(b"original")
    data = {
        "schema": "usdcraft.training_interactions",
        "schema_version": 1,
        "source": {"usd_sha256": hashlib.sha256(b"original").hexdigest()},
        "conventions": {"quat_order": "wxyz", "meters_per_unit": 1, "up_axis": "Z"},
        "interactions": [],
    }
    usd.with_name("interaction_annotations.json").write_text(json.dumps(data))
    assert load_interactions(usd) == data
    usd.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="Stale"):
        load_interactions(usd)


def test_moving_body_frame_and_quaternion_order():
    candidate = {"pose": {"position_m": [0, -0.1, 0], "quat_wxyz": [1, 0, 0, 0]}}
    s = np.sqrt(0.5)
    np.testing.assert_allclose(world_pose(candidate, [1, 2, 3, 0, 0, s, s]), [1.1, 2, 3, 0, 0, s, s])


def test_release_has_priority_and_requires_actual_threshold():
    rule = dict(release_when="release_joint >= release_at", release_at=0.0024, hold=0, engage_tolerance=0.017)
    assert next_engaged(rule, True, 0, 0.0023)
    assert not next_engaged(rule, True, 0, 0.0024)
    assert not next_engaged(rule, False, 0, 0.003)
    assert not next_engaged(rule, False, 0.8, 0)
    assert next_engaged(rule, False, 0.005, 0)
    with pytest.raises(AssertionError):
        next_engaged(rule, True, float("nan"), 0)


def test_toaster_does_not_catch_at_rest():
    rule = dict(release_when="release_joint >= release_at", release_at=0.002, hold=0.06, engage_tolerance=0.001)
    assert not next_engaged(rule, False, 0, 0)
    assert next_engaged(rule, False, 0.0595, 0)
    assert not next_engaged(rule, True, 0.06, 0.002)


def test_latch_preserves_reset_defaults_and_isolates_environments():
    import torch
    from types import SimpleNamespace

    from data_engine.assets.latches import LatchController

    class Asset:
        joint_names = ["held", "release"]
        num_instances = 2

        def __init__(self):
            self.data = SimpleNamespace(
                joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[0.0, 0.06], [0.0, 0.003]]] * 2)),
                default_joint_pos=SimpleNamespace(torch=torch.zeros((2, 2))),
                joint_pos=SimpleNamespace(torch=torch.zeros((2, 2))),
            )

        def write_joint_position_limit_to_sim_index(self, limits, **kwargs):
            self.data.joint_pos_limits.torch.copy_(limits)
            self.data.default_joint_pos.torch.clamp_(min=limits[:, :, 0], max=limits[:, :, 1])

        def write_joint_position_to_sim_index(self, position, env_ids):
            self.data.joint_pos.torch[env_ids] = position

        def write_joint_velocity_to_sim_index(self, velocity, env_ids):
            assert not velocity.any()

    rule = dict(
        name="catch",
        joint={"name": "held"},
        release_joint={"name": "release"},
        release_when="release_joint >= release_at",
        release_at=0.002,
        hold=0.06,
        engage_tolerance=0.001,
        initially_engaged=False,
    )
    asset = Asset()
    controller = LatchController(asset, [rule])
    asset.data.joint_pos.torch[0, 0] = 0.06
    controller.update()
    assert controller.engaged == [[True], [False]]
    assert not asset.data.default_joint_pos.torch.any()
    controller.reset([1])
    assert controller.engaged == [[True], [False]]
    assert asset.data.joint_pos.torch[0, 0] == pytest.approx(0.06)
    asset.data.joint_pos.torch[0, 1] = 0.003
    controller.update()
    assert controller.engaged == [[False], [False]]
    assert asset.data.joint_pos_limits.torch[0, 0, 0] == 0
    assert asset.data.joint_pos.torch[0, 0] == pytest.approx(0.06), "Release must not teleport the held joint"
