# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Slow contact approaches and measured grasp checks for isolated single-tool experiments."""

import math
import numpy as np
from scipy.spatial.transform import Rotation


def tcp_pose(session, side):
    """Read the measured TCP world pose."""
    sensor = session.base.scene["ee_frame" if side == "right" else "left_ee_frame"]
    return sensor.data.target_pos_w.torch[0, 0].cpu().numpy().copy(), Rotation.from_quat(
        sensor.data.target_quat_w.torch[0, 0].cpu().numpy()
    )


def object_sample(session, name, side):
    """Measure object clearance and its transform relative to the active TCP."""
    position, rotation = session.object_pose(name)
    tcp, tcp_rot = tcp_pose(session, side)
    vertices = rotation.apply(session.meshes[name]["vertices"]) + position
    return dict(
        position=position.tolist(),
        quaternion_xyzw=rotation.as_quat().tolist(),
        minimum_z_m=float(vertices[:, 2].min()),
        relative_position=tcp_rot.inv().apply(position - tcp).tolist(),
        relative_quaternion_xyzw=(tcp_rot.inv() * rotation).as_quat().tolist(),
    )


def planned_move(session, side, name, target, rotation, source):
    """Execute one native cuMotion segment with the intended contact exclusion."""
    s = session
    result = s.check_target(s.planner_for(side), side, name, target, rotation, True, exclude=source)
    assert result.get("executed") and result.get("pose_pass"), f"{name}: Motion planner execution failed"
    s.save()


def run_grasp_lift(session, config, initialize=True):
    """Physically approach, close, lift and hold the selected tool without resetting its state."""
    s = session
    source = getattr(s.args, "tool", "drill")
    side = config["tools"][source]["arm"]
    gripper_index = 7 if side == "right" else 15
    grasp = {**config["grasp"], **config["tools"][source].get("grasp", {})}
    s.report.update(
        mode="grasp_lift",
        tool=source,
        arm=side,
        destination=config["mapping"][source],
        task_success=False,
        grasp_lift_success=False,
        scope=f"Physical {source} approach, closure and lift; no object teleportation or physical attachment",
    )
    if initialize:
        s.reset()
    home_position, home_rotation = tcp_pose(s, side)
    s.report["home_tcp"] = dict(position=home_position.tolist(), quaternion_xyzw=home_rotation.as_quat().tolist())
    s.planner_for(side)
    initial = object_sample(s, source, side)
    candidate = next(
        c for c in config["tools"][source]["candidates"] if c["id"] == config["selected_grasp_candidates"][source]
    )
    position, rotation = s.object_pose(source)
    target = position + rotation.apply(candidate["centre_asset_m"])
    target[2] = grasp.get("tcp_height_m", candidate["grasp_tcp_world_z_m"])
    yaw = (
        math.degrees(math.atan2(rotation.as_matrix()[1, 0], rotation.as_matrix()[0, 0]))
        + candidate["closing_yaw_asset_deg"]
    )
    orientation = Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_euler("x", 180, degrees=True)
    pregrasp = target + np.array([0, 0, candidate["pregrasp_clearance_m"]])
    planner = s.planner_for(side)
    result = s.check_target(planner, side, "pregrasp", pregrasp, orientation, True)
    assert result.get("executed") and result.get("pose_pass") and result.get("objects_undisturbed"), "Pregrasp failed"
    planned_move(s, side, "approach", target, orientation, source)
    s.phase[0] = "close"
    s.action[0, gripper_index] = -1
    for _ in range(30):
        s.step()
    closed = object_sample(s, source, side)
    s.report["grasp"] = dict(initial=initial, closed=closed)
    s.save()
    closed_tcp, closed_rotation = tcp_pose(s, side)
    lift = closed_tcp + np.array([0, 0, grasp["lift_m"]])
    planned_move(s, side, "lift", lift, closed_rotation, source)
    s.phase[0] = "hold"
    samples = []
    for _ in range(30):
        s.step()
        samples.append(object_sample(s, source, side))
    from isaaclab_arena_cumotion.g2_collection.workcell.grasp_evaluation import evaluate_hold

    metrics = evaluate_hold(initial, samples)
    s.report["grasp"].update(hold=samples, **metrics)
    s.save()
    assert metrics["passed"], "Grasp hold failed: insufficient clearance or unstable relative pose"
    s.report["grasp_lift_success"] = True
    s.save()
