# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless spatial predicates and geometric checks."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse

from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
from isaaclab_arena.tasks.predicates.predicate_utils import (
    get_env,
    get_root_lin_vel_w,
    get_root_pos_w,
    get_root_quat_w,
    select,
)
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.gripper import Gripper
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def gripper_distance_from_object_exceeds_threshold(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    gripper: Gripper,
    distance_threshold_m: float,
) -> torch.Tensor:
    """Check that a gripper is farther than a threshold from an object.

    Args:
        env: Environment supplying object and frame positions through ArenaWorld.
        subject_name: Object asset whose origin defines the distance.
        gripper: Embodiment-owned gripper implementation.
        distance_threshold_m: Strict minimum distance, in meters.

    Returns:
        Boolean tensor with one result per environment.
    """
    assert (
        math.isfinite(distance_threshold_m) and distance_threshold_m >= 0.0
    ), "Distance threshold must be non-negative and finite."
    gripper_position_W = gripper.get_position_w(env.arena_world)
    subject_position_W = env.arena_world.get_pose_w(subject_name)[:, :3]
    return torch.linalg.vector_norm(subject_position_W - gripper_position_W, dim=-1) > distance_threshold_m


def object_bounds_center_over_destination(
    object_centroid_W: torch.Tensor,
    T_W_D: torch.Tensor,
    destination_bounds_D: AxisAlignedBoundingBox,
) -> torch.Tensor:
    """Check whether an object's bounds center is over a destination.

    The check requires the object's bounds center to be inside the destination's
    X/Y footprint and above its lower Z bound. The upper Z bound is intentionally
    ignored so the same check works for open containers and supporting surfaces.
    This is a center-point test, not full-object containment.

    Args:
        object_centroid_W: Object geometry centroid expressed in world frame
            ``W``. Shape is ``(num_envs, 3)``.
        T_W_D: Destination poses mapping points from destination frame ``D``
            into world frame ``W``. Shape is ``(num_envs, 7)`` with quaternion
            order ``(x, y, z, w)``.
        destination_bounds_D: Destination bounds aligned with frame ``D`` and
            measured from its origin.

    Returns:
        One Boolean result per environment.
    """
    t_W_D, q_W_D = T_W_D[:, :3], T_W_D[:, 3:]

    object_centroid_D = quat_apply_inverse(
        q_W_D,
        object_centroid_W - t_W_D,
    )
    center_inside_horizontal_bounds = (
        (object_centroid_D[:, :2] >= destination_bounds_D.min_point[:, :2])
        & (object_centroid_D[:, :2] <= destination_bounds_D.max_point[:, :2])
    ).all(dim=-1)
    center_above_destination_bottom = object_centroid_D[:, 2] >= destination_bounds_D.min_point[:, 2]
    return center_inside_horizontal_bounds & center_above_destination_bottom


def contact_force_is_upward_support(
    contact_force_w: torch.Tensor,
    force_threshold: float,
    support_cone_half_angle_rad: float,
) -> torch.Tensor:
    """Check whether contact forces point upward strongly enough.

    Args:
        contact_force_w: World-frame force vectors with shape
            ``(num_envs, 3)`` in newtons.
        force_threshold: Minimum force magnitude in newtons.
        support_cone_half_angle_rad: Maximum angle in radians between the force
            vector and world ``+Z``. Zero accepts only a straight-up force.

    Returns:
        One Boolean result per environment.
    """
    assert (
        contact_force_w.ndim == 2 and contact_force_w.shape[1] == 3
    ), f"contact_force_w must have shape (num_envs, 3), got {tuple(contact_force_w.shape)}."
    assert force_threshold >= 0.0, f"force_threshold must be non-negative, got {force_threshold}."
    assert (
        0.0 <= support_cone_half_angle_rad < math.pi / 2
    ), f"support_cone_half_angle_rad must be in [0, pi / 2), got {support_cone_half_angle_rad}."

    force_magnitude = torch.linalg.vector_norm(contact_force_w, dim=-1)
    upward_force = contact_force_w[:, 2]
    minimum_upward_fraction = math.cos(support_cone_half_angle_rad)
    return (
        (force_magnitude >= force_threshold)
        & (upward_force > 0.0)
        & (upward_force >= force_magnitude * minimum_upward_fraction)
    )


def object_is_moving_slowly(
    object_linear_velocity_w: torch.Tensor,
    velocity_threshold: float,
) -> torch.Tensor:
    """Check whether object linear speed is below the threshold."""
    return torch.linalg.vector_norm(object_linear_velocity_w, dim=-1) < velocity_threshold


def _position_relative_to_target(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    receiver_name: str,
    target_offset_xyz: tuple[float, float, float],
    subject_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return a subject-local point relative to a target, expressed in the receiver frame."""
    arena_world = env.arena_world
    T_W_S = arena_world.get_pose_w(subject_name)
    T_W_R = arena_world.get_pose_w(receiver_name)
    subject_offset_S = torch.as_tensor(subject_offset_xyz, dtype=T_W_S.dtype, device=T_W_S.device)
    subject_point_W = T_W_S[:, :3] + quat_apply(T_W_S[:, 3:], subject_offset_S.expand(env.num_envs, -1))
    position_R = quat_apply_inverse(T_W_R[:, 3:], subject_point_W - T_W_R[:, :3])
    target_position_R = torch.as_tensor(target_offset_xyz, dtype=position_R.dtype, device=position_R.device)
    return position_R - target_position_R


def _normalized_axis(axis: tuple[float, float, float], reference: torch.Tensor) -> torch.Tensor:
    """Return a unit axis matching the reference tensor's dtype and device."""
    vector = torch.as_tensor(axis, dtype=reference.dtype, device=reference.device)
    assert vector.shape == (3,) and torch.linalg.vector_norm(vector) > 0
    return vector / torch.linalg.vector_norm(vector)


def _relative_axial_distances(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    receiver_name: str,
    target_offset_xyz: tuple[float, float, float],
    subject_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    receiver_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed axial depth and perpendicular distance from a receiver-local target."""
    position_R = _position_relative_to_target(env, subject_name, receiver_name, target_offset_xyz, subject_offset_xyz)
    axis_R = _normalized_axis(receiver_axis, position_R)
    depth = torch.sum(position_R * axis_R, dim=-1)
    lateral = torch.linalg.vector_norm(position_R - depth[:, None] * axis_R, dim=-1)
    return depth, lateral


def lateral_in_proximity(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    receiver_name: str,
    target_offset_xyz: tuple[float, float, float],
    tolerance_lateral: float,
    *,
    subject_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    receiver_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Check point proximity perpendicular to a receiver-local axis.

    Args:
        env: Environment providing asset poses.
        subject_name: Subject asset name.
        receiver_name: Receiver asset name.
        target_offset_xyz: Target point in the receiver frame.
        tolerance_lateral: Maximum perpendicular distance from the target axis.
        subject_offset_xyz: Point in the subject frame; defaults to its origin.
        receiver_axis: Plane normal in the receiver frame; defaults to +Z.

    Returns:
        One Boolean result per environment.
    """
    _, lateral = _relative_axial_distances(
        env, subject_name, receiver_name, target_offset_xyz, subject_offset_xyz, receiver_axis
    )
    return lateral <= tolerance_lateral


def depth_in_range(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    receiver_name: str,
    target_offset_xyz: tuple[float, float, float],
    depth_min: float,
    depth_max: float | None,
    *,
    subject_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    receiver_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Check point depth relative to a receiver-local target along its configured axis.

    Args:
        env: Environment providing asset poses.
        subject_name: Subject asset name.
        receiver_name: Receiver asset name.
        target_offset_xyz: Target point in the receiver frame.
        depth_min: Inclusive minimum signed depth.
        depth_max: Inclusive maximum signed depth, or None for no upper limit.
        subject_offset_xyz: Point in the subject frame; defaults to its origin.
        receiver_axis: Depth axis in the receiver frame; defaults to +Z.

    Returns:
        One Boolean result per environment.
    """
    assert (
        depth_max is None or depth_min <= depth_max
    ), f"depth_min ({depth_min}) must not exceed depth_max ({depth_max})."
    depth, _ = _relative_axial_distances(
        env, subject_name, receiver_name, target_offset_xyz, subject_offset_xyz, receiver_axis
    )
    result = depth >= depth_min
    if depth_max is not None:
        result &= depth <= depth_max
    return result


def tilt_axis_aligned(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    receiver_name: str,
    max_tilt_rad: float,
    subject_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    receiver_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    *,
    allow_antiparallel: bool = False,
) -> torch.Tensor:
    """Check the angle between configured axes, optionally accepting opposite directions."""
    assert 0.0 <= max_tilt_rad <= math.pi, f"max_tilt_rad must be in [0, pi], got {max_tilt_rad}."
    arena_world = env.arena_world
    T_W_S = arena_world.get_pose_w(subject_name)
    T_W_R = arena_world.get_pose_w(receiver_name)
    subject_axis_F = _normalized_axis(subject_axis, T_W_S)
    receiver_axis_F = _normalized_axis(receiver_axis, T_W_R)
    subject_axis_w = quat_apply(T_W_S[:, 3:], subject_axis_F.expand(env.num_envs, -1))
    receiver_axis_w = quat_apply(T_W_R[:, 3:], receiver_axis_F.expand(env.num_envs, -1))
    axis_dot = torch.sum(subject_axis_w * receiver_axis_w, dim=-1)
    if allow_antiparallel:
        axis_dot = torch.abs(axis_dot)
    return axis_dot >= math.cos(max_tilt_rad)


def velocity_below_threshold(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float | None = None,
) -> torch.Tensor:
    """Check subject root linear speed and, optionally, angular speed."""
    arena_world = env.arena_world
    linear_velocity_w = arena_world.get_root_linear_velocity_w(subject_name)
    result = torch.linalg.vector_norm(linear_velocity_w, dim=-1) <= linear_velocity_threshold
    if angular_velocity_threshold is not None:
        angular_velocity_w = arena_world.get_root_angular_velocity_w(subject_name)
        result &= torch.linalg.vector_norm(angular_velocity_w, dim=-1) <= angular_velocity_threshold
    return result


def object_is_above_height(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_name: str,
    surface_height: float | None = None,
    use_settled_state: bool = False,
    distance: float = 1e-2,
) -> torch.Tensor:
    """Checks if an object is above a certain height.

    The reference height is either a fixed ``surface_height`` or, when ``use_settled_state`` is set, the
    object's recorded resting height (see ``objects_settled``). For envs where no settled state
    has been recorded, the result is always False.

    Returns True when ``object_name`` is at least ``distance`` m above a height reference.
    """

    assert (
        surface_height is not None
    ) != use_settled_state, "object_is_above_height requires exactly one of surface_height or use_settled_state"

    object_z = env.arena_world.get_position_w(object_name)[:, 2]
    if use_settled_state:
        settled_pos, has_settled = get_object_initial_rest_state(env, object_name)
        result = has_settled & (object_z > (settled_pos[:, 2] + distance))
    else:
        result = object_z > (surface_height + distance)
    return result


def object_moving(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_name: str,
    velocity_threshold: float = 1e-2,
) -> torch.Tensor:
    """Check whether an object is moving above a velocity threshold.

    Returns True when object_name's linear speed exceeds velocity_threshold (m/s).
    """

    arena_world = env.arena_world
    object_mean_linear_velocity_w = arena_world.get_mean_linear_velocity_w(object_name)
    speed = torch.linalg.vector_norm(object_mean_linear_velocity_w, dim=-1)
    return speed > velocity_threshold


def objects_in_proximity(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_y_separation: float,
    max_x_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Determine if two objects are within a certain proximity of each other.

    Returns True when the object is within a certain proximity of the target object.
    """

    arena_world = env.arena_world
    object_position_w = arena_world.get_position_w(object_cfg.name)
    target_object_position_w = arena_world.get_position_w(target_object_cfg.name)

    # object to target object
    x_separation = torch.abs(object_position_w[:, 0] - target_object_position_w[:, 0])
    y_separation = torch.abs(object_position_w[:, 1] - target_object_position_w[:, 1])
    z_separation = torch.abs(object_position_w[:, 2] - target_object_position_w[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def object_supported_by(
    object_vertices_pos_w: torch.Tensor,
    destination_bound: AxisAlignedBoundingBox,
    support_tolerance: float = 0.03,
    low_point_tolerance: float = 0.01,
    minimum_support_fraction: float = 0.5,
) -> torch.Tensor:
    """Check whether a large fraction of object's lowest vertices are close to the destination's top surface.

    This is a geometric-only implementation to replace contact-sensor based contact_force_is_upward_support.
    Use this for deformable objects which don't have contact sensor support yet, see
    https://github.com/isaac-sim/IsaacLab/issues/4410

    Args:
        object_vertices_pos_w: Object vertices in world frame ``W``.
            Shape is ``(num_envs, num_vertices, 3)``.
        destination_bound: Axis-aligned bounds of the destination object in ``W``.
        support_tolerance: Maximum vertical distance in meters between a object vertex
            and the destination top surface to count as supported.
        low_point_tolerance: Band above the lowest node height used to select
            bottom nodes for the support-fraction denominator.
        minimum_support_fraction: Minimum fraction of low nodes that must lie
            on the destination footprint near the top surface.

    Returns:
        One Boolean result per environment.
    """
    low_z = object_vertices_pos_w[..., 2].amin(dim=1, keepdim=True)
    low_mask = object_vertices_pos_w[..., 2] <= low_z + low_point_tolerance
    # TODO(qianl, 2026-09-15): use destination's vertices instead of AABB top surface/footprint for closeness check.
    near_top = torch.abs(object_vertices_pos_w[..., 2] - destination_bound.top_surface_z[:, None]) <= support_tolerance
    inside_footprint = (
        (object_vertices_pos_w[..., :2] >= destination_bound.min_point[:, None, :2])
        & (object_vertices_pos_w[..., :2] <= destination_bound.max_point[:, None, :2])
    ).all(dim=-1)
    supported_points = low_mask & near_top & inside_footprint
    support_fraction = supported_points.sum(dim=1) / low_mask.sum(dim=1).clamp_min(1)
    return support_fraction >= minimum_support_fraction


def object_on_destination(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    destination_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg | None,
    force_threshold: float,
    velocity_threshold: float,
    support_cone_half_angle_rad: float = math.pi / 4,
) -> torch.Tensor:
    """Check whether an object is stably placed on its destination.

    The object's spawned-bounds center must be over the destination footprint
    and above its bottom, and the object's linear speed must be below the configured threshold.
    Rigid objects must also have upward support force. Deformable objects must have enough
    low nodal points near the destination's top surface. Destinations must be rooted objects.

    Args:
        env: The live Arena manager-based environment.
        object_cfg: The object being placed.
        destination_cfg: The rooted object or scene entry receiving the object.
        contact_sensor_cfg: The object's contact sensor filtered to the destination. None for a deformable object.
        force_threshold: Minimum upward support force in newtons.
        velocity_threshold: Maximum object linear speed in meters per second.
        support_cone_half_angle_rad: Maximum angle in radians from world ``+Z`` for the support force.

    Returns:
        One Boolean result per environment.
    """

    arena_world = env.arena_world
    assert (
        destination_cfg.name not in env.scene.deformable_objects
    ), "object_on_destination does not support deformable destinations"

    object_center_over_destination = object_bounds_center_over_destination(
        object_centroid_W=arena_world.get_centroid_w(object_cfg.name),
        T_W_D=arena_world.get_pose_w(destination_cfg.name),
        destination_bounds_D=arena_world.get_aabb_in_local_frame(destination_cfg.name),
    )

    if object_cfg.name in env.scene.deformable_objects:
        # Use geometric support for deformable objects.
        object_vertices_w = arena_world.get_vertices_w(object_cfg.name)
        destination_vertices_w = arena_world.get_vertices_w(destination_cfg.name)
        destination_bound = AxisAlignedBoundingBox(
            min_point=destination_vertices_w.amin(dim=1),
            max_point=destination_vertices_w.amax(dim=1),
        )
        destination_provides_upward_support = object_supported_by(
            object_vertices_pos_w=object_vertices_w,
            destination_bound=destination_bound,
        )
    else:
        # Use contact sensor for rigid objects.
        assert contact_sensor_cfg is not None, "object_on_destination requires a contact sensor for rigid objects"
        contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
        force_matrix_w = contact_sensor.data.force_matrix_w
        assert force_matrix_w is not None, f"Contact sensor '{contact_sensor_cfg.name}' has no filtered force matrix."
        force_matrix_w = force_matrix_w.torch
        assert force_matrix_w.shape == (env.num_envs, 1, 1, 3), (
            f"Contact sensor '{contact_sensor_cfg.name}' must provide one sensed body and one filtered body; "
            f"got force shape {tuple(force_matrix_w.shape)}."
        )
        # The two zeros select the sensor's single sensed body and single filtered destination body.
        support_force_on_object_w = force_matrix_w[:, 0, 0, :]
        destination_provides_upward_support = contact_force_is_upward_support(
            contact_force_w=support_force_on_object_w,
            force_threshold=force_threshold,
            support_cone_half_angle_rad=support_cone_half_angle_rad,
        )

    object_mean_linear_velocity_w = arena_world.get_mean_linear_velocity_w(object_cfg.name)
    object_moves_slowly = object_is_moving_slowly(object_mean_linear_velocity_w, velocity_threshold)
    return object_center_over_destination & destination_provides_upward_support & object_moves_slowly


def _object_axis_tilt(env: ManagerBasedRLEnv, object_name: str, axis: tuple[float, float, float]) -> torch.Tensor:
    """Return the angle, in radians, between an object's local ``axis`` and world +Z."""
    rot = matrix_from_quat(get_root_quat_w(env, object_name))
    local_axis = torch.tensor(axis, device=rot.device, dtype=rot.dtype)
    world_axis = rot @ local_axis
    return torch.acos(world_axis[:, 2].clamp(-1.0, 1.0))


def objects_upright(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    threshold_rad: float,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that every named object is upright.

    An object counts as upright when its local ``axis``, rotated into the world, is within
    ``threshold_rad`` of world +Z.

    Args:
        env: The environment.
        object_names: Objects to check.
        threshold_rad: How far the axis may tilt away from world +Z.
        axis: The object-local axis that should point up.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when all of object_names are upright.
    """
    result = torch.ones_like(_object_axis_tilt(env, object_names[0], axis), dtype=torch.bool)
    for name in object_names:
        result &= _object_axis_tilt(env, name, axis) < threshold_rad
    return select(result, env_id)


def lowest_object_upright(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    threshold_rad: float,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that whichever of the named objects sits lowest is upright.

    Used for stacking, where the bottom item carries the pile and so has to be squarer than the
    ones resting on it.

    Args:
        env: The environment.
        object_names: Objects to check.
        threshold_rad: How far the lowest object's axis may tilt away from world +Z.
        axis: The object-local axis that should point up.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when the lowest of object_names is upright to within threshold_rad.
    """
    heights = torch.stack([get_root_pos_w(env, name)[:, 2] for name in object_names], dim=-1)
    tilts = torch.stack([_object_axis_tilt(env, name, axis) for name in object_names], dim=-1)
    lowest = heights.argmin(dim=-1, keepdim=True)
    return select(tilts.gather(-1, lowest).squeeze(-1) < threshold_rad, env_id)


def objects_stacked(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    xy_threshold: float,
    min_z_gap: float = 0.005,
    max_z_gap: float | None = None,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that the named objects form a single pile.

    The objects are ordered by height, then every neighbouring pair must be within
    ``xy_threshold`` of each other horizontally and separated vertically by at least
    ``min_z_gap`` and, when given, at most ``max_z_gap``.

    The lower bound distinguishes a stack from objects standing side by side. The upper bound
    matters just as much in practice: without it an object held in the gripper directly above
    the pile reads as the top of it, so a run counts as solved while the last item is still in
    the air. Set it to roughly one object height.

    Args:
        env: The environment.
        object_names: Objects that should form the pile.
        xy_threshold: How far apart neighbouring objects may be horizontally.
        min_z_gap: Minimum height difference between neighbouring objects.
        max_z_gap: Maximum height difference between neighbouring objects, or None to leave the
            pile unbounded above.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when object_names form a stack.
    """
    positions = torch.stack([get_root_pos_w(env, name) for name in object_names], dim=1)
    order = positions[:, :, 2].argsort(dim=1)
    ordered = positions.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))

    lower, upper = ordered[:, :-1, :], ordered[:, 1:, :]
    xy_distance = torch.linalg.vector_norm(upper[:, :, :2] - lower[:, :, :2], dim=-1)
    z_gap = upper[:, :, 2] - lower[:, :, 2]
    ok = (xy_distance < xy_threshold) & (z_gap >= min_z_gap)
    if max_z_gap is not None:
        ok &= z_gap <= max_z_gap
    return select(ok.all(dim=1), env_id)


def objects_at_rest(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    velocity_threshold: float = 0.02,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that none of the named objects is still moving.

    Used to insist a manipulation has actually been let go of and settled, rather than being
    judged mid-motion while the robot still holds something. Calibrate the threshold against the
    contact configuration the task ends in: PhysX reports a standing velocity for bodies in a
    resting contact stack that never decays, so a value taken from a lone object is unreachable.

    Args:
        env: The environment.
        object_names: Objects to check.
        velocity_threshold: Speed below which an object counts as settled, in m/s.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when every object in object_names is slower than velocity_threshold.
    """
    speeds = torch.stack(
        [torch.linalg.vector_norm(get_root_lin_vel_w(env, name), dim=-1) for name in object_names], dim=-1
    )
    return select((speeds < velocity_threshold).all(dim=-1), env_id)


def objects_upright_about_any_axis(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    threshold_rad: float,
    axes: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
    ),
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that every named object has at least one of ``axes`` pointing up.

    Objects that are flat slabs stand on an edge rather than on a face, so no single local axis
    identifies "the right way up" -- any of the four in-plane axes will do. Defaults to those four.

    Args:
        env: The environment.
        object_names: Objects to check.
        threshold_rad: How far the best-aligned axis may tilt away from world +Z.
        axes: Object-local axes, any one of which pointing up counts.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when every object in object_names has some axis in axes within threshold_rad of up.
    """
    result = torch.ones_like(_object_axis_tilt(env, object_names[0], axes[0]), dtype=torch.bool)
    for name in object_names:
        per_axis = torch.stack([_object_axis_tilt(env, name, axis) for axis in axes], dim=-1)
        result &= (per_axis < threshold_rad).any(dim=-1)
    return select(result, env_id)


def any_object_near_body(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    body_name: str,
    max_distance_m: float,
    min_height_m: float | None = None,
    articulation_name: str = "robot",
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that at least one of the named objects sits within reach of a robot body.

    Distance is measured between the object's origin and the body's origin, so the threshold has
    to allow for both the tool frame's offset from the contact and the object's own size. With
    ``min_height_m`` set, an object only counts while its origin is above that world height --
    which separates one held at the gripper from one lying wherever the gripper happens to be.

    Args:
        env: The environment.
        object_names: Candidate objects; only one has to be near.
        body_name: Body of the articulation to measure from, e.g. a gripper's tool frame.
        max_distance_m: How far the object's origin may sit from the body's, in metres.
        min_height_m: World height the object's origin must also clear, or None for no floor.
        articulation_name: Scene key of the articulation carrying the body.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when some object in object_names is within max_distance_m of the body.
    """
    articulation = get_env(env).scene.articulations[articulation_name]
    body_index = list(articulation.data.body_names).index(body_name)
    body_pos = wp.to_torch(articulation.data.body_pos_w)[:, body_index]
    near = []
    for name in object_names:
        position = get_root_pos_w(env, name)
        ok = torch.linalg.vector_norm(position - body_pos, dim=-1) <= max_distance_m
        if min_height_m is not None:
            ok &= position[:, 2] > min_height_m
        near.append(ok)
    return select(torch.stack(near, dim=-1).any(dim=-1), env_id)


def _objects_in_frame_box(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    frame_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float | None],
) -> torch.Tensor:
    """Returns, per environment and per object, whether the object's origin is inside the box.

    The box is axis-aligned in ``frame_name``'s own frame, so it follows that object as it moves.

    Returns:
        A boolean tensor of shape (num_envs, len(object_names)).
    """
    frame_rot = matrix_from_quat(get_root_quat_w(env, frame_name))
    frame_pos = get_root_pos_w(env, frame_name)
    z_min, z_max = z_range

    inside = []
    for name in object_names:
        # World offset rotated back into the frame's axes; the env origin cancels in the delta.
        local = torch.einsum("nij,nj->ni", frame_rot.transpose(1, 2), get_root_pos_w(env, name) - frame_pos)
        ok = (local[:, 0] >= x_range[0]) & (local[:, 0] <= x_range[1])
        ok &= (local[:, 1] >= y_range[0]) & (local[:, 1] <= y_range[1])
        ok &= local[:, 2] >= z_min
        if z_max is not None:
            ok &= local[:, 2] <= z_max
        inside.append(ok)
    return torch.stack(inside, dim=-1)


def any_object_in_frame_box(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    frame_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float | None],
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that at least one of the named objects sits inside a box fixed to another object.

    Args:
        env: The environment.
        object_names: Candidate objects; only one has to be inside.
        frame_name: Object whose frame the box is expressed in.
        x_range: Minimum and maximum along the frame's local X, in metres.
        y_range: Minimum and maximum along the frame's local Y, in metres.
        z_range: Minimum and maximum along the frame's local Z; the maximum may be None.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when some object in object_names is inside the box.
    """
    return select(_objects_in_frame_box(env, object_names, frame_name, x_range, y_range, z_range).any(dim=-1), env_id)


def count_objects_in_frame_box(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    frame_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float | None],
    count: int,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks that exactly ``count`` of the named objects sit inside a box fixed to another object.

    Args:
        env: The environment.
        object_names: Objects to count.
        frame_name: Object whose frame the box is expressed in.
        x_range: Minimum and maximum along the frame's local X, in metres.
        y_range: Minimum and maximum along the frame's local Y, in metres.
        z_range: Minimum and maximum along the frame's local Z; the maximum may be None.
        count: How many objects must be inside, exactly.
        env_id: Restrict the result to a single environment, or None for all of them.

    Returns:
        True when the number of object_names inside the box equals count.
    """
    inside = _objects_in_frame_box(env, object_names, frame_name, x_range, y_range, z_range)
    return select(inside.sum(dim=-1) == count, env_id)
