# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that an optimized carry can opt into the same command speed limits as a graph path."""

import numpy as np
import torch
from types import SimpleNamespace

from isaaclab_arena_examples.g2_workcell.motion import prepare_execution_points


def test_optimized_carry_speed_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "isaaclab_arena_examples.g2_workcell.trajectory.planner_feasibility", lambda planner: lambda points: True
    )
    args = SimpleNamespace(
        graph_phases=[], retime_phases=["carry_to_bin"], mode="pick_place", tool="metal_stock", output_dir=tmp_path
    )
    base = SimpleNamespace(step_dt=1 / 15, device="cpu")
    entry = {}
    commands = prepare_execution_points(args, object(), torch.tensor([[0.0], [1.0]]), "carry_to_bin", base, entry)
    velocity = torch.diff(commands, dim=0) / base.step_dt
    assert torch.max(abs(velocity)) <= 0.25 * 1.001
    assert len(commands) > 2
    assert np.array_equal(np.load(tmp_path / "carry_to_bin_joint_targets.npy"), commands.numpy())
    assert entry["retiming"]["velocity_peak_rad_s"] <= 0.25 * 1.001
