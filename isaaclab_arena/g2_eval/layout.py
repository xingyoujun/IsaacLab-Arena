# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Each episode's object layout of a G2-Eval task (pure NumPy).

A copy of the data engine's ``collect/perturb.py`` sampler for objects standing on the table, as its policy
evaluation uses it: each free object moves by a uniform offset within ``xy_m`` and turns by a uniform angle within
``yaw_deg`` (none for a rotationally symmetric object). A draw is kept when every moved centre stays in the workspace
and in a cell some arm reaches, every moved object stays in the head camera's image, and no two objects come closer
than their footprint radii plus ``min_gap_m`` (or than they started, if closer); otherwise it is drawn again.
``task.json``'s ``randomization`` holds the settings and, under ``objects``, each object's layout entry.
"""

import numpy as np

MAX_TRIES = 500
VIEW_MARGIN_PX = 8
"""An object's outline must stay this far inside the camera image."""
SAMPLE_KEYS = ("xy_m", "yaw_deg", "workspace_m", "min_gap_m", "per_object", "shuffle", "face_camera", "region")
"""The ``randomization`` keys ``sample`` takes."""


def in_view(camera: dict, points: np.ndarray) -> bool:
    """Whether world points (N x 3) are in front of ``camera`` and inside its image by the margin."""
    T = np.asarray(camera["T_W_C"], float)
    p_C = (np.atleast_2d(points) - T[:3, 3]) @ T[:3, :3]
    if np.any(p_C[:, 2] <= 0.0):
        return False
    u = camera["fx"] * p_C[:, 0] / p_C[:, 2] + camera["ppx"]
    v = camera["fy"] * p_C[:, 1] / p_C[:, 2] + camera["ppy"]
    width, height = camera["size_px"]
    m = VIEW_MARGIN_PX
    return bool(np.all((u >= m) & (u <= width - m) & (v >= m) & (v <= height - m)))


def outline(spec: dict, centre: np.ndarray) -> np.ndarray:
    """Points around an object's footprint circle at its base and top heights (N x 3), for the view check."""
    angles = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    ring = centre + spec["radius_m"] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    base, top = spec["base_z_m"], spec["base_z_m"] + spec["height_m"]
    return np.concatenate([np.column_stack([ring, np.full(8, base)]), np.column_stack([ring, np.full(8, top)])])


def reached(reach: dict, xy) -> bool:
    """Whether ``xy`` lies in a cell of ``reach``."""
    half = reach["step_m"] / 2 + 1e-9
    return any(abs(xy[0] - x) <= half and abs(xy[1] - y) <= half for x, y in reach["cells_xy_m"])


def movable(spec: dict, name: str, per_object: dict, face_camera: dict) -> bool:
    """Whether the sampler moves an object: not fixed, and an articulated one only when the task names it."""
    if spec.get("fixed") or per_object.get(name, {}).get("fixed"):
        return False
    return not spec.get("articulated") or name in per_object or name in face_camera


def sample(
    objects: dict,
    rng: np.random.Generator,
    xy_m: float,
    yaw_deg: float,
    workspace_m=None,
    min_gap_m: float = 0.02,
    per_object: dict | None = None,
    shuffle: list | None = None,
    face_camera: dict | None = None,
    region: dict | None = None,
) -> dict:
    """One draw of the layout ``objects`` (``randomization["objects"]``).

    Args:
        objects: Per object ``xy_m``, ``radius_m``, ``yaw_deg``, ``base_z_m``, ``height_m`` and optionally ``fixed``,
            ``articulated``, ``symmetric``.
        rng: The episode's random generator.
        xy_m, yaw_deg: Half-ranges of the offset along x and y and of the turn.
        workspace_m: ``[x0, x1, y0, y1]`` the moved centres stay in (None: anywhere).
        min_gap_m: Clearance between footprints.
        per_object, shuffle, face_camera: The task's additions (``per_object``: own ranges or fixed; ``shuffle``:
            groups whose positions are permuted; ``face_camera``: turned to face the head camera).
        region: ``{"reach": {"step_m", "cells_xy_m"}, "camera": {...}}`` or None.

    Returns:
        Per object ``dxy_m`` (offset) and ``dyaw_deg`` (turn); zeros for an object that stays.
    """
    per_object, shuffle, face_camera = per_object or {}, shuffle or [], face_camera or {}
    names = sorted(objects)
    for name in names:
        assert "supporter" not in objects[name], f"{name} stands on another object; G2-Eval layouts are on the table"
    centre = {n: np.asarray(objects[n]["xy_m"], float) for n in names}
    free = {n for n in names if movable(objects[n], n, per_object, face_camera)}
    for group in shuffle:
        assert set(group) <= free, f"shuffle group {group}: every member must be free to move"
    camera = (region or {}).get("camera")
    for _ in range(MAX_TRIES):
        base = dict(centre)
        for group in shuffle:
            order = rng.permutation(len(group))
            base.update({name: centre[group[i]] for name, i in zip(group, order)})
        drawn = {}
        for name in names:
            spec = objects[name]
            if name not in free:
                drawn[name] = {"dxy_m": [0.0, 0.0], "dyaw_deg": 0.0}
                continue
            own = per_object.get(name, {})
            reach_xy, reach_yaw = own.get("xy_m", xy_m), own.get("yaw_deg", yaw_deg)
            position = base[name] + rng.uniform(-reach_xy, reach_xy, 2)
            if name in face_camera:
                facing = face_camera[name]
                eye = np.asarray(camera["T_W_C"], float)[:2, 3] - position
                towards = np.degrees(np.arctan2(eye[1], eye[0])) - facing.get("front_deg", 0.0)
                tolerance = facing.get("tolerance_deg", 10.0)
                turn = towards + rng.uniform(-tolerance, tolerance) - spec["yaw_deg"]
                turn = (turn + 180.0) % 360.0 - 180.0
            else:
                turn = 0.0 if spec.get("symmetric") else float(rng.uniform(-reach_yaw, reach_yaw))
            drawn[name] = {"dxy_m": np.round(position - centre[name], 5).tolist(), "dyaw_deg": round(float(turn), 3)}
        if accepted(objects, drawn, centre, workspace_m, min_gap_m, region, base):
            return drawn
    raise AssertionError(f"no layout within {xy_m} m keeps the objects apart, in the workspace and in view")


def accepted(
    objects: dict, drawn: dict, centre: dict, workspace_m, min_gap_m: float, region: dict | None = None, base=None
) -> bool:
    """Whether a draw keeps every moved centre in the workspace and the region, every moved object in view, and the
    objects apart (``base``: the layout positions after a shuffle, ``centre`` when None)."""
    base = base or centre
    moved = {n: centre[n] + np.asarray(drawn[n]["dxy_m"]) for n in centre}
    shifted = {n for n in centre if drawn[n]["dxy_m"] != [0.0, 0.0] or drawn[n]["dyaw_deg"] != 0.0}
    if workspace_m is not None:
        x0, x1, y0, y1 = workspace_m
        for name, (x, y) in moved.items():
            if name in shifted and not (x0 <= x <= x1 and y0 <= y <= y1):
                return False
    reach = (region or {}).get("reach")
    if reach is not None and not all(reached(reach, moved[n]) for n in shifted):
        return False
    camera = (region or {}).get("camera")
    if camera is not None:
        for name in shifted:
            if "height_m" in objects[name] and not in_view(camera, outline(objects[name], moved[name])):
                return False
    names = sorted(centre)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            needed = objects[a]["radius_m"] + objects[b]["radius_m"] + min_gap_m
            needed = min(needed, float(np.linalg.norm(base[a] - base[b])))
            if np.linalg.norm(moved[a] - moved[b]) < needed - 1e-9:
                return False
    return True


def sample_uniform(
    positions: dict,
    rng: np.random.Generator,
    xy_m: float,
    objects: list[str],
    min_separation_m: float | None = None,
    table_half_extent_m=None,
    footprint_radius_m: float = 0.0,
) -> dict:
    """One draw of a legacy G2 task's layout, as its collection drew it: each of ``objects`` offset along x and y by a
    uniform draw within ``xy_m``; with ``min_separation_m``, centres at least that far apart and footprints (circles of
    ``footprint_radius_m``) inside ``table_half_extent_m``, else drawn again.

    Args:
        positions: Per object its nominal centre ``[x, y]``.
        rng: The episode's random generator.
        xy_m: Half-range of the offset along x and along y.
        objects: The objects that move; the others stay.

    Returns:
        Per object ``dxy_m`` and ``dyaw_deg`` (always 0).
    """
    for _ in range(MAX_TRIES):
        drawn = {name: {"dxy_m": [0.0, 0.0], "dyaw_deg": 0.0} for name in positions}
        for name in objects:
            drawn[name]["dxy_m"] = rng.uniform(-xy_m, xy_m, 2).round(6).tolist()
        moved = [np.asarray(positions[n], float) + drawn[n]["dxy_m"] for n in objects]
        if table_half_extent_m is not None:
            half = np.asarray(table_half_extent_m, float)
            if not all(np.all(np.abs(p) + footprint_radius_m < half) for p in moved):
                continue
        if min_separation_m is not None:
            pairs = [(a, b) for i, a in enumerate(moved) for b in moved[i + 1 :]]
            if any(np.linalg.norm(a - b) < min_separation_m for a, b in pairs):
                continue
        return drawn
    raise AssertionError(f"no layout within {xy_m} m keeps the objects {min_separation_m} m apart on the table")
