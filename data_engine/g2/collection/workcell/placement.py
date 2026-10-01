# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Carry and release a physically grasped tool using its measured TCP transform."""

import itertools
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

from data_engine.g2.collection.workcell.grasp_motion import object_sample, planned_move, run_grasp_lift, tcp_pose


def mesh_slab_cells(vertices, faces, cell_m=0.018):
    """Cover each mesh slab by a box enclosing its clipped triangles and interior."""
    triangles = vertices[faces]
    low, high = vertices.min(0), vertices.max(0)
    axis = int(np.argmax(high - low))
    edges = np.linspace(low[axis], high[axis], int(np.ceil((high[axis] - low[axis]) / cell_m)) + 1)
    cells = []
    for start, stop in zip(edges[:-1], edges[1:]):
        selected = triangles[(triangles[:, :, axis].min(1) <= stop) & (triangles[:, :, axis].max(1) >= start)]
        if not len(selected):
            continue
        points = selected.reshape(-1, 3)
        clipped = [points[(points[:, axis] >= start) & (points[:, axis] <= stop)]]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            first, second = selected[:, i], selected[:, j]
            delta = second - first
            for plane in (start, stop):
                valid = (
                    (np.minimum(first[:, axis], second[:, axis]) <= plane)
                    & (np.maximum(first[:, axis], second[:, axis]) >= plane)
                    & (abs(delta[:, axis]) > 1e-12)
                )
                fraction = (plane - first[valid, axis]) / delta[valid, axis]
                clipped.append(first[valid] + fraction[:, None] * delta[valid])
        points = np.concatenate(clipped)
        if not len(points):
            continue
        slab_low, slab_high = points.min(0), points.max(0)
        counts = np.maximum(1, np.ceil((slab_high - slab_low) / cell_m).astype(int))
        sizes = (slab_high - slab_low) / counts
        centres = np.array(
            list(itertools.product(*[slab_low[i] + (np.arange(counts[i]) + 0.5) * sizes[i] for i in range(3)]))
        )
        cells.extend(np.c_[centres, np.full(len(centres), np.linalg.norm(sizes / 2) + 0.002)])
    return np.array(cells)


def payload_cover(vertices, relative_position, relative_rotation, capacity=384, faces=None):
    """Cover the full source volume with spheres; mesh slabs tighten non-box tools."""
    if faces is None:
        low, high = vertices.min(0), vertices.max(0)
        counts = np.maximum(1, np.ceil((high - low) / 0.024).astype(int))
        sizes = (high - low) / counts
        centres = np.array(
            list(itertools.product(*[low[i] + (np.arange(counts[i]) + 0.5) * sizes[i] for i in range(3)]))
        )
        cells = np.c_[centres, np.full(len(centres), np.linalg.norm(sizes / 2) + 0.002)]
        coverage = "Entire scaled source AABB volume"
    else:
        cells = mesh_slab_cells(vertices, faces)
        coverage = "Union of full-volume slab boxes enclosing clipped source triangles"
    assert 0 < len(cells) <= capacity, f"Payload requires {len(cells)} spheres, capacity {capacity}"
    spheres = np.zeros((capacity, 4), dtype=np.float32)
    spheres[:, 3] = -1
    spheres[: len(cells), :3] = relative_rotation.apply(cells[:, :3]) + relative_position
    spheres[: len(cells), 3] = cells[:, 3]
    return spheres, dict(
        spheres=len(cells),
        radius_m=float(cells[:, 3].max()),
        coverage=coverage + "; planning attachment only, no physical constraint",
    )


def bin_sample(session, source, destination):
    """Measure the full source geometry in the current destination-bin frame."""
    p, r = session.object_pose(source)
    bp, br = session.object_pose(destination)
    vertices = br.inv().apply(r.apply(session.meshes[source]["vertices"]) + p - bp)
    return dict(
        position=p.tolist(),
        quaternion_xyzw=r.as_quat().tolist(),
        minimum_bin=vertices.min(0).tolist(),
        maximum_bin=vertices.max(0).tolist(),
    )


def evaluate_placement(samples, geometry):
    """Require stable geometry wholly inside the sampled cavity and close to its floor."""
    positions = np.array([x["position"] for x in samples])
    rotations = Rotation.from_quat([x["quaternion_xyzw"] for x in samples])
    low = np.min([x["minimum_bin"] for x in samples], axis=0)
    high = np.max([x["maximum_bin"] for x in samples], axis=0)
    bounds = np.array(geometry["inner_xy_bounds_m"])
    floor = geometry["floor_z_m"]
    drift = float(np.linalg.norm(positions - positions[0], axis=1).max())
    angle = float(np.degrees((rotations[0].inv() * rotations).magnitude()).max())
    contained = bool(
        np.all(low[:2] > bounds[0] + 0.003)
        and np.all(high[:2] < bounds[1] - 0.003)
        and high[2] < geometry["rim_max_z_m"] - 0.005
    )
    supported = bool(floor - 0.004 < low[2] and max(x["minimum_bin"][2] for x in samples) < floor + 0.008)
    return dict(
        contained=contained,
        near_floor=supported,
        position_drift_m=drift,
        rotation_drift_deg=angle,
        minimum_bin=low.tolist(),
        maximum_bin=high.tolist(),
        passed=bool(contained and supported and drift < 0.005 and angle < 5),
    )


def placement_candidates(session, source, bin_position, bin_rotation, yaws=None, offset_bin=(0.0, 0.0), tilt=None):
    """Yield geometry-centred poses fitting the destination cavity."""
    vertices = session.meshes[source]["vertices"]
    bounds = np.array(session.geometry["bin"]["inner_xy_bounds_m"])
    if yaws is None:
        yaws = (270, 180, 0, 90) if source == "drill" else (90, 270, 0, 180)
    for yaw in yaws:
        rotation = Rotation.from_euler("z", yaw, degrees=True)
        if tilt is not None:
            rotation = rotation * tilt
        local = bin_rotation.inv().apply(rotation.apply(vertices))
        low, high = local.min(0), local.max(0)
        if np.any(high[:2] - low[:2] > bounds[1] - bounds[0] - 0.006):
            continue
        centre = (bounds[0] + bounds[1] - low[:2] - high[:2]) / 2 + np.array(offset_bin)
        if np.any(low[:2] + centre < bounds[0] + 0.003) or np.any(high[:2] + centre > bounds[1] - 0.003):
            continue
        world_offset = bin_rotation.apply([*centre, 0])
        offsets = [tuple(world_offset[:2])]
        for offset in offsets:
            yield yaw, offset, rotation, bin_position + np.array([*offset, 0.0])


def run_pick_place(session, config, initialize=True):
    """Run one continuous grasp, carry, lower, release and retreat episode."""
    s = session
    run_grasp_lift(s, config, initialize=initialize)
    source = getattr(s.args, "tool", "drill")
    side = config["tools"][source]["arm"]
    destination = config["mapping"][source]
    gripper_index = 7 if side == "right" else 15
    s.report.update(
        mode="pick_place",
        place_success=False,
        scope=f"Single {source} grasp and placement; full-task success remains false",
    )
    s.report[f"{source}_place_success"] = False
    sample = object_sample(s, source, side)
    relative_position = np.array(sample["relative_position"])
    relative_rotation = Rotation.from_quat(sample["relative_quaternion_xyzw"])
    spheres, cover = payload_cover(
        s.meshes[source]["vertices"],
        relative_position,
        relative_rotation,
        capacity=getattr(s.args, "payload_capacities", {}).get(side, 384),
        faces=(
            s.meshes[source]["faces"]
            if source != "drill" or config["tools"][source].get("mesh_payload", False)
            else None
        ),
    )
    s.attachments[side] = torch.tensor(spheres, device=s.base.device)
    s.report["payload"] = dict(
        **cover,
        relative_position=relative_position.tolist(),
        relative_quaternion_xyzw=relative_rotation.as_quat().tolist(),
    )
    s.report["carry_max_drift_m"] = 0.0
    s.report["carry_max_angle_deg"] = 0.0

    def guard():
        current = object_sample(s, source, side)
        drift = float(np.linalg.norm(np.array(current["relative_position"]) - relative_position))
        angle = float(
            np.degrees((relative_rotation.inv() * Rotation.from_quat(current["relative_quaternion_xyzw"])).magnitude())
        )
        s.report["carry_max_drift_m"] = max(s.report["carry_max_drift_m"], drift)
        s.report["carry_max_angle_deg"] = max(s.report["carry_max_angle_deg"], angle)
        assert drift < 0.005 and angle < 5, f"Payload slipping: {drift:.4f} m / {angle:.2f} deg"

    s.step_guard[0] = guard
    escape = config["tools"][source].get("escape_before_transfer_world_m")
    if escape is not None:
        _, current_rotation = tcp_pose(s, side)
        target = np.asarray(escape, dtype=float)
        assert target.shape == (3,) and np.isfinite(target).all(), "Invalid carry escape waypoint"
        s.report["transfer_route"] = dict(
            escape_tcp_world_m=target.tolist(), orientation="Hold measured grasp rotation"
        )
        planned_move(s, side, "escape_before_transfer", target, current_rotation, source)
    elif config["tools"][source].get("vertical_raise_before_transfer", True):
        current_tcp, current_rotation = tcp_pose(s, side)
        raised = current_tcp.copy()
        raised[2] = 0.29
        planned_move(s, side, "raise_above_rim", raised, current_rotation, source)
    else:
        s.report["transfer_route"] = "Full collision-checked loaded plan from verified lift; no forced vertical raise"
    execution_urdf = s.args.robot_urdf
    from isaaclab_arena.assets.g2_asset_paths import transfer_urdf as resolve_transfer_urdf

    transfer_urdf = resolve_transfer_urdf(config["tools"][source])
    if transfer_urdf:
        s.args.robot_urdf = Path(transfer_urdf)
        s.planners.pop(side)
    planner = s.planner_for(side)
    chosen = None
    bin_pos, bin_rot = s.object_pose(destination)
    # Probe full poses first. All candidates retain the original scene and box dimensions.
    placement = config["tools"][source].get("placement", {})
    tilt = None
    if placement.get("preserve_grasp_tilt", False):
        held_rotation = s.object_pose(source)[1]
        held_yaw = np.arctan2(held_rotation.as_matrix()[1, 0], held_rotation.as_matrix()[0, 0])
        tilt = Rotation.from_euler("z", -held_yaw) * held_rotation
    for yaw, offset, object_rot, object_target in placement_candidates(
        s,
        source,
        bin_pos,
        bin_rot,
        placement.get("yaws_world_deg"),
        placement.get("offset_bin_m", (0.0, 0.0)),
        tilt=tilt,
    ):
        tcp_rot = object_rot * relative_rotation.inv()
        object_target[2] = placement.get("object_height_world_m", s.geometry["bin"]["rim_max_z_m"] + 0.06)
        tcp_target = object_target - tcp_rot.apply(relative_position)
        name = f"preplace_ik_{yaw}_{offset[0]}_{offset[1]}"
        result = s.check_target(planner, side, name, tcp_target, tcp_rot, False, exclude=source)
        if result["ik_success"]:
            chosen = (object_target, object_rot, tcp_target, tcp_rot)
            break
    assert chosen is not None, "No collision-free loaded preplace IK candidate"
    object_target, object_rot, tcp_target, tcp_rot = chosen
    s.report["chosen_placement"] = dict(
        object_position_world=object_target.tolist(),
        object_quaternion_xyzw=object_rot.as_quat().tolist(),
        tcp_position_world=tcp_target.tolist(),
        tcp_quaternion_xyzw=tcp_rot.as_quat().tolist(),
    )
    result = s.check_target(planner, side, "carry_to_bin", tcp_target, tcp_rot, True, exclude=source)
    assert result.get("executed"), "Loaded carry planning failed"
    if transfer_urdf:
        s.args.robot_urdf = execution_urdf
        s.planners.pop(side)
        s.planner_for(side)
        s.report["joint_margin_policy"] = (
            "Transfer plans reserve 0.03 rad; local feedback respects original hard limits"
        )
    # Correct load-induced tracking error while the attached collision cover remains active.
    planned_move(s, side, "preplace_align", tcp_target, tcp_rot, source)
    rotated = object_rot.apply(s.meshes[source]["vertices"])
    release_object = object_target.copy()
    release_object[2] = bin_pos[2] + s.geometry["bin"]["floor_z_m"] - rotated[:, 2].min() + 0.020
    release_tcp = release_object - tcp_rot.apply(relative_position)
    planned_move(s, side, "lower_into_bin", release_tcp, tcp_rot, source)
    s.report["before_release"] = bin_sample(s, source, destination)
    s.step_guard[0] = None
    s.phase[0] = "release"
    s.action[0, gripper_index] = 1
    s.step(30)
    s.planner_for(side).detach_spheres_from_robot()
    del s.attachments[side]
    s.report["after_release"] = bin_sample(s, source, destination)
    released = evaluate_placement([s.report["after_release"]], s.geometry["bin"])
    s.report["release_check"] = released
    s.save()
    assert released["contained"] and released["near_floor"], "Tool has not settled into bin before retreat"
    # The initially touching pads may leave the source; the bin and other obstacles remain active.
    if config["tools"][source].get("planned_retreat", False):
        clearance, _ = tcp_pose(s, side)
        clearance[2] += config["tools"][source].get("release_clearance_m", 0.035)
        planned_move(s, side, "release_clearance", clearance, tcp_rot, source)
        retreat = np.array(s.report["home_tcp"]["position"])
        retreat[2] = 0.32
        retreat_rotation = Rotation.from_quat(s.report["home_tcp"]["quaternion_xyzw"])
        result = s.check_target(s.planner_for(side), side, "retreat_from_bin", retreat, retreat_rotation, True)
        assert result.get("executed") and result.get("objects_undisturbed"), "Unloaded retreat failed or moved objects"
        planned_move(s, side, "retreat_align", retreat, retreat_rotation, source)
    else:
        retreat = release_tcp.copy()
        retreat[2] = max(0.29, tcp_target[2])
        planned_move(s, side, "retreat_from_bin", retreat, tcp_rot, source)
    s.phase[0] = "verify_placement"
    samples = []
    for _ in range(30):
        s.step()
        samples.append(bin_sample(s, source, destination))
    metrics = evaluate_placement(samples, s.geometry["bin"])
    s.report["placement"] = dict(samples=samples, **metrics)
    s.save()
    assert metrics["passed"], "Final tool placement failed containment/support/stability checks"
    assert tcp_pose(s, side)[0][2] > s.geometry["bin"]["rim_max_z_m"] + 0.07, "Gripper not clear of bin"
    assert float(s.action[0, gripper_index]) == 1
    joints = dict(zip(s.robot.joint_names, s.robot.data.joint_pos.torch[0].cpu().tolist()))
    outer = "idx41_gripper_l_outer_joint1" if side == "left" else "idx81_gripper_r_outer_joint1"
    inner = "idx31_gripper_l_inner_joint1" if side == "left" else "idx71_gripper_r_inner_joint1"
    assert joints[outer] > 0.65 and joints[inner] < -0.65, "Gripper did not physically open"
    s.report["place_success"] = True
    s.report[f"{source}_place_success"] = True
    s.save()
