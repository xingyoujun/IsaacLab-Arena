# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert geometric joint paths to sampled, collision-checked rest-to-rest motions."""

import numpy as np


def segment(first, last, resolution=0.01):
    """Sample a joint segment with a bounded maximum joint increment."""
    count = max(1, int(np.ceil(np.max(np.abs(last - first)) / resolution)))
    return np.linspace(first, last, count + 1)


def compress_path(points, tolerance=0.002):
    """Keep ordered corners using a maximum joint-space line deviation tolerance."""
    points = np.asarray(points, dtype=float)
    assert points.ndim == 2 and len(points) >= 2 and np.isfinite(points).all()
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        first, last = pending.pop()
        if last <= first + 1:
            continue
        delta = points[last] - points[first]
        norm = np.dot(delta, delta)
        fraction = np.clip((points[first + 1 : last] - points[first]) @ delta / max(norm, 1e-30), 0, 1)
        errors = np.max(abs(points[first + 1 : last] - points[first] - fraction[:, None] * delta), axis=1)
        index = first + 1 + int(np.argmax(errors))
        if errors.max() > tolerance:
            keep.add(index)
            pending.extend([(first, index), (index, last)])
    return sorted(keep)


def timed_segment(first, last, dt, velocity, acceleration, jerk):
    """Sample a quintic segment including both endpoints at an integer number of steps."""
    distance = float(np.max(abs(last - first)))
    duration = max(
        1.875 * distance / velocity, np.sqrt(5.773503 * distance / acceleration), np.cbrt(60 * distance / jerk), dt
    )
    count = int(np.ceil(duration / dt))
    u = np.arange(count + 1) / count
    progress = 10 * u**3 - 15 * u**4 + 6 * u**5
    return first + progress[:, None] * (last - first)


def dense_segments(points):
    """Include exact command samples and interpolate between them at <= 0.01 rad."""
    return np.concatenate([segment(a, b) for a, b in zip(points[:-1], points[1:])])


def prepare_trajectory(points, feasible, dt, velocity=0.25, acceleration=0.5, jerk=2.0):
    """Simplify only verified segments and retime them with quintic rest-to-rest interpolation."""
    points = np.asarray(points, dtype=float)
    assert dt > 0 and min(velocity, acceleration, jerk) > 0
    corners = compress_path(points)

    def valid_chord(first, last):
        commands = timed_segment(points[first], points[last], dt, velocity, acceleration, jerk)
        # Check the actual nonuniform execution samples before accepting a shortcut.
        return feasible(segment(points[first], points[last])) and feasible(dense_segments(commands))

    # A compression chord may cross an obstacle: recursively restore the original path there.
    def verified(first, last):
        if valid_chord(first, last):
            return [last]
        assert last > first + 1, (
            f"Original geometric path has an infeasible segment {first}:{last}: "
            f"{getattr(feasible, 'last_failure', None)}"
        )
        middle = (first + last) // 2
        return verified(first, middle) + verified(middle, last)

    indices = [0]
    for first, last in zip(corners[:-1], corners[1:]):
        indices.extend(verified(first, last))
    # Greedily skip only a small number of nearby corners, keeping planning cost bounded.
    simplified = [indices[0]]
    cursor = 0
    while cursor < len(indices) - 1:
        next_cursor = cursor + 1
        for candidate in range(min(cursor + 8, len(indices) - 1), cursor + 1, -1):
            if valid_chord(indices[cursor], indices[candidate]):
                next_cursor = candidate
                break
        simplified.append(indices[next_cursor])
        cursor = next_cursor
    knots = points[simplified]
    sampled = [knots[:1]]
    durations = []
    for first, last in zip(knots[:-1], knots[1:]):
        commands = timed_segment(first, last, dt, velocity, acceleration, jerk)
        sampled.append(commands[1:])
        durations.append((len(commands) - 1) * dt)
    result = np.concatenate(sampled)
    # Check both the retimed samples and all line segments (<= 0.01 rad sampling).
    dense = dense_segments(result)
    assert feasible(dense), f"Retimed path collision validation failed: {getattr(feasible, 'last_failure', None)}"
    padded = np.pad(result, ((3, 3), (0, 0)), mode="edge")
    peaks = [float(np.max(abs(np.diff(padded, n=order, axis=0) / dt**order))) for order in (1, 2, 3)]
    assert np.all(np.array(peaks) <= np.array([velocity, acceleration, jerk]) * 1.001)
    return result, dict(
        input_points=len(points),
        knots=len(knots),
        output_points=len(result),
        collision_checked_output_samples=len(dense),
        collision_checker="Single-state execution constraints with both collision costs enabled",
        duration_s=float(sum(durations)),
        velocity_peak_rad_s=peaks[0],
        acceleration_peak_rad_s2=peaks[1],
        jerk_peak_rad_s3=peaks[2],
        limits=dict(velocity=velocity, acceleration=acceleration, jerk=jerk),
        collision_sampling_max_joint_increment_rad=0.01,
        scope="Sampled collision validation; no continuous collision certificate",
    )


def planner_feasibility(planner, maximum_tcp_tilt_deg=None):
    """Use the execution runner's single-state constraints for simplified paths."""
    import torch

    from curobo.types.state import JointState

    limits = planner.kinematics.get_joint_limits().position

    def feasible(points):
        if not np.isfinite(points).all():
            return False
        q = torch.as_tensor(points, device=limits.device, dtype=limits.dtype)
        if bool(((q < limits[0]) | (q > limits[1])).any()):
            feasible.last_failure = {"kind": "joint_limits"}
            return False
        if maximum_tcp_tilt_deg is not None:
            pose = planner.compute_kinematics(
                JointState.from_position(q, joint_names=planner.kinematics.joint_names)
            ).ee_pose
            quat = pose.quaternion
            vertical = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
            if bool((-vertical < np.cos(np.deg2rad(maximum_tcp_tilt_deg))).any()):
                feasible.last_failure = {"kind": "tcp_tilt"}
                return False
        for index in range(len(q)):
            planner.rollout_fn.robot_self_collision_constraint.enable_cost()
            planner.rollout_fn.primitive_collision_constraint.enable_cost()
            valid, reason = planner.check_start_state(
                JointState.from_position(q[index : index + 1], joint_names=planner.kinematics.joint_names)
            )
            # check_start_state may leave self-collision disabled after a rejected world collision.
            planner.rollout_fn.robot_self_collision_constraint.enable_cost()
            planner.rollout_fn.primitive_collision_constraint.enable_cost()
            if not valid:
                feasible.last_failure = dict(
                    kind="constraint", offset=index, joint_position=q[index].cpu().tolist(), single_reason=str(reason)
                )
                return False
        return True

    return feasible
