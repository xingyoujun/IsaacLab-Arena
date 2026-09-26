# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise geometric simplification and finite-difference execution limits."""

import numpy as np

from isaaclab_arena_examples.g2_workcell.trajectory import prepare_trajectory, segment


def test_obstacle_detour_and_limits():
    def feasible(points):
        return bool(np.all(np.linalg.norm(points, axis=1) > 0.5))

    corners = np.array([[-1, 0], [-1, 1], [1, 1], [1, 0]], dtype=float)
    path = np.concatenate([segment(a, b, 0.001) for a, b in zip(corners[:-1], corners[1:])])
    result, stats = prepare_trajectory(path, feasible, 1 / 15)
    assert feasible(result)
    assert np.array_equal(result[0], path[0]) and np.allclose(result[-1], path[-1])
    assert stats["knots"] >= 3 and stats["output_points"] < stats["input_points"]
    padded = np.pad(result, ((3, 3), (0, 0)), mode="edge")
    for order, bound in enumerate([0.25, 0.5, 2.0], 1):
        assert np.max(abs(np.diff(padded, n=order, axis=0) * 15**order)) <= bound * 1.001


def test_reject_infeasible_original():
    import pytest

    with pytest.raises(AssertionError, match="infeasible"):
        prepare_trajectory(np.array([[0.0], [1.0]]), lambda points: False, 1 / 15)


def test_nonuniform_commands_reject_shortcut_missed_by_uniform_grid():
    from isaaclab_arena_examples.g2_workcell.trajectory import timed_segment

    first, last = np.array([-1.0, 0]), np.array([1.0, 0])
    uniform = segment(first, last)
    commands = timed_segment(first, last, 1 / 15, 0.25, 0.5, 2.0)
    distances = np.min(abs(commands[:, None, 0] - uniform[None, :, 0]), axis=1)
    obstacle = commands[int(np.argmax(distances))]
    radius = float(distances.max() / 3)

    def feasible(points):
        return bool(np.all(np.linalg.norm(points - obstacle, axis=1) > radius))

    assert feasible(uniform) and not feasible(commands)
    corners = np.array([first, [0, 0.1], last])
    path = np.concatenate([segment(a, b, 0.001) for a, b in zip(corners[:-1], corners[1:])])
    result, stats = prepare_trajectory(path, feasible, 1 / 15)
    assert feasible(result) and stats["knots"] >= 3
