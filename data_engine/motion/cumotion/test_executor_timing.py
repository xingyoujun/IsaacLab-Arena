# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native trajectories reject even a one-ULP overshoot of their final timestamp."""

import numpy as np
from types import SimpleNamespace

from data_engine.motion.cumotion.executor import ArmExecutor


def test_final_timestamp_is_inside_closed_native_interval():
    timestamps = []

    def target(time):
        timestamps.append(time)
        if not 0 <= time <= 0.1:
            return None
        return SimpleNamespace(joints=SimpleNamespace(positions=SimpleNamespace(numpy=lambda: np.array([time]))))

    trajectory = SimpleNamespace(duration=0.1, get_target_state=target)
    planner = SimpleNamespace(
        trajectory_generator=SimpleNamespace(generate_trajectory_from_cspace_waypoints=lambda points: trajectory),
        joint_positions=lambda: np.array([0.0]),
        safety=SimpleNamespace(check_path=lambda *args: None, check_actual=lambda: None),
    )
    executor = ArmExecutor.__new__(ArmExecutor)
    executor.planner, executor.dt, executor.settle_steps = planner, 0.1, 0
    executor.step = lambda **kwargs: None
    path = SimpleNamespace(get_waypoints=lambda: SimpleNamespace(numpy=lambda: np.array([[0.0], [0.1]])))
    executor.follow(path, speed=0.3)
    assert len(timestamps) == 4
    assert timestamps[-1] == 0.1
