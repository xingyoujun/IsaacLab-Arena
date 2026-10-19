# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Success relations of a G2-Eval task, judged on the objects' poses and state (pure NumPy).

A copy of the data engine's ``g2_eval.evaluate`` (SimEngine ``data_engine/assets/g2_eval.py``). ``in``, ``on`` and
``upright`` judge as its ``harness/relations.py`` does: each object is the convex hull of its scaled mesh
(``geometry.json``), whose bounding box at any pose is that of all its points. ``placed``, ``above``, ``inserted`` and
``in_bin`` are the checks of the G2 tasks collected before the data engine (stack bowls, peg into sleeve, clean
workcell table), which also read velocities and contact forces. The pack's export checked them on a successful end.
"""

import math
import numpy as np

ON_GAP_MAX_M = 0.01
"""``on``: the object's bottom is within this of the target's top."""
TILT_MAX_DEG = 5.0
"""``on`` and ``upright``: the default largest tilt of the object's up axis from the vertical."""
LEGACY_RELATIONS = ("placed", "above", "inserted", "in_bin")
EVALUATED = ("in", "on", "upright", *LEGACY_RELATIONS)


def tilt_deg(T: np.ndarray) -> float:
    """Angle between an object's up axis and the world's."""
    return float(np.degrees(np.arccos(np.clip(T[2, 2], -1.0, 1.0))))


def evaluate(success: list[dict], geometry: dict, poses: dict, state: dict | None = None) -> list[dict]:
    """Judge each success relation on the objects' poses and state.

    Args:
        success: ``task.json``'s ``success`` (``relation``, ``object``, ``target``, ``params``).
        geometry: ``geometry.json``'s ``objects``: per object its convex hull ``hull_m`` (points in its body frame,
            scaled), its receive opening ``receive_m`` (a body-frame point, or None) and its scaled local bounding box
            ``box_m`` (``[min, max]``).
        poses: Per object its pose T_W_O (4x4) in the task frame (the table top at ``table_top_z_m``).
        state: For the legacy relations: ``lin_vel`` and ``ang_vel`` per object (world, m/s and rad/s) and
            ``contact_n`` per ``"<object>><target>"`` (the contact force between them, N; None: not measured, the
            relation is then judged without it and says so).

    Returns:
        Per relation ``relation``, ``object``, ``target``, ``holds`` and ``measured``.
    """
    state = state or {}

    def points(name):
        T_W_O = np.asarray(poses[name], float)
        return np.asarray(geometry[name]["hull_m"], float) @ T_W_O[:3, :3].T + T_W_O[:3, 3]

    def opening(name):
        receive = geometry[name].get("receive_m")
        if receive is None:
            return None
        T_W_O = np.asarray(poses[name], float)
        return T_W_O[:3, :3] @ np.asarray(receive, float) + T_W_O[:3, 3]

    def inside(name, target, margin):
        top = opening(target)
        if top is None:
            return False, {}
        own = points(name)
        center = (own.min(axis=0) + own.max(axis=0)) / 2
        other = points(target)
        radius = float(min(other.max(axis=0)[:2] - other.min(axis=0)[:2])) / 2
        offset = float(np.linalg.norm(center[:2] - top[:2]))
        depth = float(top[2] - own.min(axis=0)[2])
        measured = {"bottom_below_opening_m": depth, "axis_offset_m": offset, "radius_m": radius}
        return depth > margin and offset < radius, measured

    results = []
    for entry in success:
        relation, params = entry["relation"], entry.get("params") or {}
        name, target = entry.get("object"), entry.get("target")
        assert relation in EVALUATED, f"no G2-Eval evaluator for relation {relation!r}"
        tilt = tilt_deg(np.asarray(poses[name], float))
        if relation in LEGACY_RELATIONS:
            holds, measured = _legacy(relation, params, name, target, geometry, poses, state)
        elif relation == "upright":
            holds, measured = tilt <= params.get("tilt_max_deg", TILT_MAX_DEG), {"tilt_deg": tilt}
        elif relation == "in":
            margin = params.get("rim_margin_m", 0.0)
            holds, measured = inside(name, target, margin)
            for other in geometry:
                if holds or other in (name, target) or geometry[other].get("receive_m") is None:
                    continue
                # Nesting counts: inside something that is itself inside the target.
                inner, inner_measured = inside(name, other, margin)
                if inner and inside(other, target, margin)[0]:
                    holds, measured = True, {**inner_measured, "via": other}
        else:
            own, other = points(name), points(target)
            center = (own.min(axis=0) + own.max(axis=0)) / 2
            gap = float(own.min(axis=0)[2] - other.max(axis=0)[2])
            over = bool(np.all(center[:2] >= other.min(axis=0)[:2]) and np.all(center[:2] <= other.max(axis=0)[:2]))
            limit = params.get("tilt_max_deg", TILT_MAX_DEG)
            holds = abs(gap) <= ON_GAP_MAX_M and over and tilt <= limit
            measured = {"gap_m": gap, "over_target": over, "tilt_deg": tilt}
        results.append(
            {"relation": relation, "object": name, "target": target, "holds": bool(holds), "measured": measured}
        )
    return results


def _legacy(relation, params, name, target, geometry, poses, state) -> tuple[bool, dict]:
    """A legacy relation (``LEGACY_RELATIONS``) and what it measured."""
    T_o, T_t = np.asarray(poses[name], float), np.asarray(poses[target], float)
    if relation == "above":
        # The stack's order (the collection's check after its last step): the object's root above the target's.
        rise = float(T_o[2, 3] - T_t[2, 3])
        return rise > 0.0, {"rise_m": rise}
    if relation == "placed":
        # PickAndPlaceTask: in contact with the target, slow, and its root near the target's along each axis.
        separation = np.abs(T_o[:3, 3] - T_t[:3, 3])
        speed = float(np.linalg.norm(state["lin_vel"][name]))
        force = state.get("contact_n", {}).get(f"{name}>{target}")
        near = bool(np.all(separation < np.asarray(params["max_separation_m"], float)))
        holds = near and speed < params["speed_max_m_s"] and (force is None or force > params["force_threshold_n"])
        return holds, {"separation_m": separation.tolist(), "speed_m_s": speed, "contact_n": force}
    if relation == "inserted":
        # SleeveTask: the peg's root (its bottom centre) at the sleeve's bore floor, its axis along the sleeve's.
        offset = T_t[:3, :3].T @ (T_o[:3, 3] - T_t[:3, 3])
        axis = T_t[:3, :3].T @ T_o[:3, 2]
        radial, seating = float(np.linalg.norm(offset[:2])), float(abs(offset[2]))
        holds = (
            radial <= params["radial_tolerance_m"]
            and seating <= params["seating_tolerance_m"]
            and axis[2] >= math.cos(math.radians(params["axis_tolerance_deg"]))
        )
        return holds, {"radial_m": radial, "seating_m": seating, "axis_deg": math.degrees(math.acos(min(1.0, axis[2])))}
    # in_bin (the clean-workcell task): the object's box wholly inside the bin's cavity, on its floor, below its rim,
    # and the object and the bin at rest.
    low, high = (np.asarray(v, float) for v in geometry[name]["box_m"])
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    inside = ((corners @ T_o[:3, :3].T + T_o[:3, 3]) - T_t[:3, 3]) @ T_t[:3, :3]
    bounds = np.asarray(params["inner_xy_bounds_m"], float)
    margin = params["planar_margin_m"]
    planar = bool(np.all(inside[:, :2] >= bounds[0] + margin) and np.all(inside[:, :2] <= bounds[1] - margin))
    bottom, top = float(inside[:, 2].min()), float(inside[:, 2].max())
    floor = abs(bottom - params["floor_z_m"]) <= params["floor_tolerance_m"]
    below_rim = top <= params["rim_max_z_m"] + params["rim_tolerance_m"]
    speeds = {
        "object_m_s": float(np.linalg.norm(state["lin_vel"][name])),
        "object_rad_s": float(np.linalg.norm(state["ang_vel"][name])),
        "bin_m_s": float(np.linalg.norm(state["lin_vel"][target])),
    }
    still = (
        speeds["object_m_s"] < params["max_linear_speed_m_s"]
        and speeds["object_rad_s"] < params["max_angular_speed_rad_s"]
        and speeds["bin_m_s"] < params["max_linear_speed_m_s"]
    )
    measured = {"inside_xy": planar, "bottom_m": bottom, "top_m": top, **speeds}
    return planar and floor and below_rim and still, measured
