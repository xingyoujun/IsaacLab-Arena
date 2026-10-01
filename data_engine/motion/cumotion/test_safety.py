# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent checks must catch unsafe interior samples and actual-body overlaps."""

import numpy as np
import torch
import yaml
from types import SimpleNamespace

import pytest

from data_engine.motion.cumotion.safety import (
    MeasuredSphereGuard,
    MotionSafety,
    MotionSafetyError,
    joint_samples,
    line_error,
)


def test_interior_collision_is_rejected_even_with_safe_endpoints():
    guard = MotionSafety.__new__(MotionSafety)
    guard.planner = SimpleNamespace(joint_limits=np.array([[-2, 2]]))
    guard.inspector = SimpleNamespace(
        frames_in_self_collision=lambda q: [("arm", "head")] if abs(q.item()) < 0.05 else []
    )
    guard.report = dict(planned_samples=0, max_joint_step_rad=0.02, failures=[])
    guard.path_constraint = None
    with pytest.raises(MotionSafetyError, match="self_collision"):
        guard.check_path([[-1], [1]])
    assert guard.report["failures"][0]["stage"] == "planned_path"


def test_joint_sampling_and_finite_cartesian_segment():
    samples = np.array(list(joint_samples([[0, 0], [0.1, 0.31]], 0.02)))
    assert np.max(np.abs(np.diff(samples, axis=0))) <= 0.0200001
    np.testing.assert_allclose(samples[-1], [0.1, 0.31])
    assert line_error(np.array([0, 0.02, 0.5]), np.zeros(3), np.array([0, 0, 1])) == pytest.approx(0.02)
    assert line_error(np.array([0, 0, 2]), np.zeros(3), np.array([0, 0, 1])) == pytest.approx(1)


def test_measured_body_motion_and_explicit_ignore_pairs(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                collision_spheres=[
                    {"arm": [dict(center=[0, 0, 0], radius=0.2)]},
                    {"head": [dict(center=[0, 0, 0], radius=0.2)]},
                ]
            )
        )
    )
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="fixture"/>')
    data = SimpleNamespace(
        body_names=["arm", "head"],
        body_pos_w=torch.tensor([[[0.0, 0, 0], [1.0, 0, 0]]]),
        body_quat_w=torch.tensor([[[0.0, 0, 0, 1], [0.0, 0, 0, 1]]]),
    )
    cfg = SimpleNamespace(
        lula_robot_description=str(path),
        robot_urdf=str(urdf),
        tool_frame="arm",
        sim_tool_body=None,
        self_collision_ignore={},
    )
    planner = SimpleNamespace(robot=SimpleNamespace(data=data), env=SimpleNamespace(device="cpu"), cfg=cfg)
    guard = MeasuredSphereGuard(planner)
    assert guard.check()[0] == []
    # Changing the inactive head's measured position must be observed without rebuilding FK.
    data.body_pos_w[0, 1, 0] = 0.3
    pairs, clearance = guard.check()
    assert pairs == [("arm", "head")] and clearance == pytest.approx(-0.1)
    cfg.self_collision_ignore = {"head": ["arm"]}
    assert MeasuredSphereGuard(planner).check()[0] == []


def test_planner_success_flag_cannot_bypass_independent_path_check():
    from data_engine.motion.cumotion.planner import CumotionArmPlanner, JointPath

    planner = CumotionArmPlanner.__new__(CumotionArmPlanner)
    planner.joint_limits = np.array([[-2, 2]])
    planner.tool_correction = np.eye(3)
    planner._planner = SimpleNamespace(
        plan_to_pose_target=lambda *args: JointPath([[-1], [1]]),
        plan_to_cspace_target=lambda *args: JointPath([[-1], [1]]),
    )
    guard = MotionSafety.__new__(MotionSafety)
    guard.planner = planner
    guard.inspector = SimpleNamespace(
        frames_in_self_collision=lambda q: [("arm", "head")] if abs(q.item()) < 0.05 else []
    )
    guard.report = dict(planned_samples=0, max_joint_step_rad=0.02, failures=[])
    guard.path_constraint = None
    planner.safety = guard
    assert planner.plan_pose(np.array([-1]), np.zeros(3), np.array([1, 0, 0, 0])) is None
    assert planner.plan_config(np.array([-1]), np.array([1])) is None


def test_readonly_native_fk_arrays_work_in_cartesian_check():
    from scipy.spatial.transform import Rotation

    translation = np.array([0.0, 0.0, 0.5])
    translation.flags.writeable = False
    matrix = np.eye(3)
    matrix.flags.writeable = False
    pose = SimpleNamespace(translation=translation, rotation=SimpleNamespace(matrix=lambda: matrix))
    guard = MotionSafety.__new__(MotionSafety)
    guard.planner = SimpleNamespace(
        joint_limits=np.array([[-2, 2]]),
        cfg=SimpleNamespace(tool_frame="tool"),
        kinematics=SimpleNamespace(pose=lambda *args: pose),
        base_pos=np.zeros(3),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    guard.inspector = SimpleNamespace(frames_in_self_collision=lambda q: [])
    guard.report = dict(planned_samples=0, failures=[])
    guard.path_constraint = dict(
        start=np.zeros(3),
        end=np.array([0.0, 0.0, 1.0]),
        rotation=Rotation.identity(),
        position_tolerance_m=0.003,
        angle_tolerance_rad=0.05,
    )
    guard.check_configuration([0])
