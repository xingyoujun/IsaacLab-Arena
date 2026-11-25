# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The success check of a ``g2.eval.task.v2`` task (``success_check.v1``), judged every control step (NumPy + SciPy).

A copy of the data engine's judge (SimEngine ``data_engine/harness/success_check.py``, ``Judge``), which judged the
task's pilot. A check spec names spatial predicates over labelled objects and gives each object's geometry in its own
(asset) frame:

- ``checks``: one AND list, or ``segments`` (AND lists that must hold in order) and ``queries`` (AND lists counted every
  step; counting past ``aim`` fails the episode). A check is ``{"fn": <predicate>, ...arguments}``; ``any`` / ``all``
  nest an OR / AND list.
- ``objects``: label -> ``box`` (``min_m``, ``size_m``: the bounding box in the asset frame; its centre is where the
  object is), ``axes`` (3x3, columns: the checks' x, y, z axes in the asset frame), ``points`` (tag -> points, asset
  frame), ``circles`` (tag -> ``centers_m``, ``radii_m``), and for an object the simulation does not move, its fixed
  ``pose`` (environment frame).
- ``ee_bodies``: per arm, the body whose pose ``all_robot_back_to_origin`` compares with the episode's start.

``Judge.update`` runs once per control step: queries counted, then the first pending segment checked and, when it
holds, popped; the step the last segment is popped with every query at its aim the episode succeeds. Lists
short-circuit, so a stateful check changes the state only where it is reached.

Differences from the data engine's copy: no Shapely. ``is_pointA_in_B_functional_bbox`` measures the quad with NumPy
(the same cover test, 1e-6 m tolerance, and distances). The predicates on box footprints (``UNSUPPORTED``), on cloth
(``is_cloth_folded``) and on grasp contacts (``is_grasped``) are not evaluated in G2-Eval: a spec naming one is
refused when its ``Judge`` is made.
"""

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = "success_check.v1"
QUAD_TOLERANCE_M = 1e-6
"""A point this close to a quad counts as inside it (the data engine widens the quad by this)."""


class _Missing(Exception):
    """A check reads an object, point or circle the spec or the state does not have (the check does not hold)."""


def centre(entry: dict) -> np.ndarray:
    """The centre of an object's box (``objects[label]``), in its asset frame."""
    box = entry["box"]
    return np.asarray(box["min_m"], float) + np.asarray(box["size_m"], float) / 2


def _angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _rotation_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees((Rotation.from_matrix(Ra).inv() * Rotation.from_matrix(Rb)).magnitude()))


def polygon_area(polygon: np.ndarray) -> float:
    """The area of a polygon (N x 2, vertices in order)."""
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2)


def polygon_distance(polygon: np.ndarray, point) -> float:
    """The distance from ``point`` (xy) to a simple polygon (N x 2, vertices in order): 0 inside or on it."""
    p = np.asarray(point, float)[:2]
    a, b = polygon, np.roll(polygon, -1, axis=0)
    inside = False
    for (x0, y0), (x1, y1) in zip(a, b):
        if (y0 > p[1]) != (y1 > p[1]) and p[0] < x0 + (p[1] - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    if inside:
        return 0.0
    edge = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, edge) / np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-300), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * edge - p, axis=1)))


PREDICATES = {
    "is_axis_up": "_is_axis_up",
    "is_A_up_B": "_is_a_up_b",
    "is_axis_aligned": "_is_axis_aligned",
    "is_A_in_B_support_circle": "_is_a_in_b_support_circle",
    "is_all_A_in_B_support_circle": "_is_all_a_in_b_support_circle",
    "is_pointA_in_B_functional_bbox": "_is_pointa_in_b_functional_bbox",
    "is_all_pointA_in_B_functional_bbox": "_is_all_pointa_in_b_functional_bbox",
    "is_A_point_above_B_point_by_z_range": "_is_a_point_above_b_point_by_z_range",
    "is_AB_xy_distance_within_threshold": "_is_ab_xy_distance_within_threshold",
    "is_lift": "_is_lift",
    "is_not_lift": "_is_lift",
    "is_moved": "_is_moved",
    "is_not_moved": "_is_moved",
    "update_object_state": "_update_object_state",
    "is_stacked": "_is_stacked",
    "is_labels_axis_difference_in_range": "_is_labels_axis_difference_in_range",
    "is_A_on_B_left": "_is_a_on_b_left",
    "is_A_before_B_on_axis": "_is_a_before_b_on_axis",
    "is_A_xy_distance_close_to_pos": "_is_a_xy_distance_close_to_pos",
    "is_qpos_close": "_is_qpos_close",
    "is_all_gripper_open": "_is_all_gripper_open",
    "all_robot_back_to_origin": "_all_robot_back_to_origin",
}
"""The predicates G2-Eval evaluates: name -> the ``Judge`` method that evaluates it."""

UNSUPPORTED = {
    "is_A_in_B": "a box footprint test",
    "is_A_cover_B": "a box footprint test",
    "is_A_on_B_bottom": "a box footprint test",
    "is_A_bbox_cover_rect_region": "a box footprint test",
    "is_cloth_folded": "cloth",
    "is_grasped": "grasp contacts",
}
"""Predicates of the data engine's judge that G2-Eval does not evaluate, and what they would need."""


def calls(value, fn: str) -> bool:
    """Whether a spec (or any part of it) names predicate ``fn``."""
    if isinstance(value, dict):
        return value.get("fn") == fn or any(calls(v, fn) for v in value.values())
    if isinstance(value, list):
        return any(calls(v, fn) for v in value)
    return False


def segments(spec: dict) -> list[list]:
    """The spec's AND lists in order (``checks`` is one)."""
    return spec.get("segments") or [spec.get("checks") or []]


class Judge:
    """A spec's judgement of one episode: ``update`` once per control step, then ``result`` (module docstring)."""

    def __init__(self, spec: dict):
        assert spec.get("schema", SCHEMA) == SCHEMA, f"not a {SCHEMA} spec: {spec.get('schema')}"
        for fn, needs in UNSUPPORTED.items():
            assert not calls(spec, fn), f"G2-Eval does not evaluate {fn} ({needs}); the task's success check names it"
        self.spec = spec
        self.segments = segments(spec)
        self.queries = [{"checks": q["checks"], "aim": int(q["aim"]), "count": 0} for q in spec.get("queries") or []]
        self.objects = spec["objects"]
        self._centre = {label: centre(entry) for label, entry in self.objects.items()}
        self._axes = {label: np.asarray(entry.get("axes", np.eye(3)), float) for label, entry in self.objects.items()}
        self._fixed = {label: np.asarray(e["pose"], float) for label, e in self.objects.items() if e.get("pose")}
        self.step = 0
        self.popped_at = []
        """The step each popped segment held."""
        self.outcome = None
        """``success`` or ``failed`` once the episode is decided, else None."""
        self.pre_state = None
        """label (or ``label:tag``) -> the reference position of ``is_lift`` / ``is_moved``: the start's centre (or
        point), then where an updating check left it."""
        self.checks = []
        """The last evaluation of a segment."""
        self.segment = 0
        """The segment ``checks`` evaluated."""

    def _poses(self, poses: dict) -> dict:
        result = {label: np.asarray(T, float) for label, T in poses.items() if label in self.objects}
        for label, T in self._fixed.items():
            result.setdefault(label, T)
        return result

    def update(self, poses: dict, ee_now: dict, ee_start: dict, initial: dict, grippers: dict | None = None):
        """Judge one control step; returns ``outcome``.

        Args:
            poses: label -> the object's 4x4 asset pose now (environment frame); a fixed object without one is at
                its spec ``pose``.
            ee_now: arm -> its end body's 4x4 pose now; ``ee_start`` the same at the episode's start.
            initial: label -> the object's 4x4 pose at the episode's start.
            grippers: arm -> gripper opening normalised to 1 open, 0 closed (``is_all_gripper_open``).
        """
        if self.outcome is not None:
            return self.outcome
        self.step += 1
        self._now = self._poses(poses)
        self._start = self._poses(initial)
        self._ee_now, self._ee_start, self._grippers = ee_now, ee_start, grippers
        if self.pre_state is None:
            self.pre_state = {label: self._point(self._start, label) for label in self._start}
        for query in self.queries:
            holds, query["last"] = self._all(query["checks"], live=True)
            query["count"] += int(holds)
            if query["count"] > query["aim"]:
                self.outcome = "failed"
        if self.outcome is None and len(self.popped_at) < len(self.segments):
            self.segment = len(self.popped_at)
            holds, self.checks = self._all(self.segments[self.segment], live=True)
            if holds:
                self.popped_at.append(self.step)
        if (
            self.outcome is None
            and len(self.popped_at) == len(self.segments)
            and all(q["count"] == q["aim"] for q in self.queries)
        ):
            self.outcome = "success"
        return self.outcome

    def result(self) -> dict:
        """The judgement so far: ``success``, the last evaluated segment's ``checks``, segment progress and queries."""
        return {
            "source": self.spec.get("source"),
            "success": self.outcome == "success",
            "outcome": self.outcome,
            "steps": self.step,
            "checks": self.checks,
            "segments": {
                "total": len(self.segments),
                "done": len(self.popped_at),
                "done_at_step": list(self.popped_at),
                "checked": self.segment,
            },
            "queries": [{"aim": q["aim"], "count": q["count"], "checks": q.get("last", [])} for q in self.queries],
        }

    def _all(self, checks: list, live: bool) -> tuple[bool, list]:
        """An AND list: evaluated live up to the first check that does not hold, the rest reported (``skipped``)."""
        holds, parts = True, []
        for check in checks:
            part = self._judge(check, live and holds)
            if live and not holds:
                part["skipped"] = True
            parts.append(part)
            holds = holds and part["holds"]
        return holds, parts

    def _any(self, checks: list, live: bool) -> tuple[bool, list]:
        """An OR list: evaluated live up to the first check that holds."""
        holds, parts = False, []
        for check in checks:
            part = self._judge(check, live and not holds)
            if live and holds:
                part["skipped"] = True
            parts.append(part)
            holds = holds or part["holds"]
        return holds, parts

    def _judge(self, check: dict, live: bool) -> dict:
        fn = check["fn"]
        if fn in ("any", "all"):
            ok, parts = (self._any if fn == "any" else self._all)(check["of"], live)
            return {"check": fn, "holds": bool(ok), "of": parts}
        method = PREDICATES.get(fn)
        assert method is not None, f"no predicate {fn}"
        try:
            ok, measured = getattr(self, method)(fn, check, live)
        except _Missing as missing:
            ok, measured = False, {"problem": str(missing)}
        return {"check": fn, "args": {k: v for k, v in check.items() if k != "fn"}, "holds": bool(ok), **measured}

    # Geometry of the state --------------------------------------------------------------------------------------

    def _pose(self, label, state=None) -> np.ndarray:
        state = self._now if state is None else state
        if label not in state:
            raise _Missing(f"no pose of {label}")
        return state[label]

    def _point(self, state: dict, label) -> np.ndarray:
        """Where the object is in ``state``: its box centre."""
        T = self._pose(label, state)
        return T[:3, :3] @ self._centre[label] + T[:3, 3]

    def _here(self, label) -> np.ndarray:
        return self._point(self._now, label)

    def _frame(self, label, state=None) -> np.ndarray:
        """The object's check axes (``axes``) in the world, as columns."""
        return self._pose(label, state)[:3, :3] @ self._axes[label]

    def _points(self, label, tag) -> list[np.ndarray]:
        """The object's points ``tag`` now (world)."""
        listed = (self.objects.get(label, {}).get("points") or {}).get(tag)
        if not listed:
            raise _Missing(f"{label} has no points {tag}")
        T = self._pose(label)
        return [T[:3, :3] @ np.asarray(p, float) + T[:3, 3] for p in listed]

    def _at(self, label, tag, state=None) -> np.ndarray:
        """Where a check measures the object in ``state``: its centre, or its first point ``tag``."""
        if tag is None:
            return self._point(self._now if state is None else state, label)
        listed = (self.objects.get(label, {}).get("points") or {}).get(tag)
        if not listed:
            raise _Missing(f"{label} has no points {tag}")
        T = self._pose(label, state)
        return T[:3, :3] @ np.asarray(listed[0], float) + T[:3, 3]

    def _reference(self, label, tag) -> tuple[str, np.ndarray]:
        """The reference of ``is_lift`` / ``is_moved`` for the object (at its point ``tag``): key and position."""
        key = label if tag is None else f"{label}:{tag}"
        if key not in self.pre_state:
            if label not in self._start:
                raise _Missing(f"no start pose of {label}")
            self.pre_state[key] = self._at(label, tag, self._start)
        return key, self.pre_state[key]

    def _a_points(self, label, tag) -> list[np.ndarray]:
        """The object's points ``tag``, else its centre."""
        return self._points(label, tag) if tag is not None else [self._here(label)]

    def _in_quad(self, a, b, a_tag, b_tag) -> tuple[bool, dict]:
        """A point of A (its points ``a_tag``, else its centre) in the xy quad of B's four points ``b_tag`` (sorted
        around their centroid, widened by ``QUAD_TOLERANCE_M``)."""
        A = self._a_points(a, a_tag)
        B = np.asarray([p[:2] for p in self._points(b, b_tag)])
        if len(B) != 4:
            return False, {"problem": f"{b} has {len(B)} {b_tag} points, not 4"}
        middle = B.mean(axis=0)
        quad = B[np.argsort(np.arctan2(B[:, 1] - middle[1], B[:, 0] - middle[0]))]
        if polygon_area(quad) < 1e-12:
            return False, {"problem": f"{b}'s {b_tag} quad is degenerate"}
        gaps = [polygon_distance(quad, p) for p in A]
        distances = [round(max(gap - QUAD_TOLERANCE_M, 0.0), 4) for gap in gaps]
        return any(gap <= QUAD_TOLERANCE_M for gap in gaps), {"outside_m": distances}

    def _in_circle(self, a, b, tag, a_tag=None, radius=None) -> tuple[bool, dict]:
        """A point of A (its points ``a_tag``, else its centre) within one of B's support circles ``tag`` in xy
        (``radius`` replaces the circles' own)."""
        A = self._a_points(a, a_tag)
        circle = (self.objects.get(b, {}).get("circles") or {}).get(tag) or {}
        if not circle.get("centers_m"):
            raise _Missing(f"{b} has no support circle {tag}")
        T_B = self._pose(b)
        ok, distances = False, []
        for point in A:
            for local, r in zip(circle["centers_m"], circle["radii_m"]):
                c = T_B[:3, :3] @ np.asarray(local, float) + T_B[:3, 3]
                d = float(np.linalg.norm(point[:2] - c[:2]))
                distances.append(round(d, 4))
                ok |= d < (radius if radius is not None else r)
        return ok, {"xy_m": distances}

    # Predicates -------------------------------------------------------------------------------------------------

    def _is_axis_up(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        """The object's ``axis`` within ``threshold`` deg of world up; ``select`` picks the lowest of ``labels``."""
        label = check.get("label")
        if check.get("select"):
            label = min(check["select"]["labels"], key=lambda n: self._here(n)[2])
        angle = _angle_deg(self._frame(label) @ np.asarray(check["axis"], float), [0, 0, 1])
        return angle < check["threshold"], {"label": label, "angle_deg": round(angle, 2)}

    def _is_a_up_b(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        """A's centre above B's by more than ``z_threshold_min`` (and less than ``z_threshold_max``)."""
        dz = float(self._here(check["label_A"])[2] - self._here(check["label_B"])[2])
        z_max = check.get("z_threshold_max")
        ok = dz > check["z_threshold_min"] and (z_max is None or dz < z_max)
        return ok, {"dz_m": round(dz, 4)}

    def _is_axis_aligned(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        a = self._frame(check["label_A"]) @ np.asarray(check["axis_A"], float)
        if check.get("world_axis") is not None:
            b = np.asarray(check["world_axis"], float)
        else:
            b = self._frame(check["label_B"]) @ np.asarray(check["axis_B"], float)
        angle = _angle_deg(a, b)
        return angle < check["threshold"], {"angle_deg": round(angle, 2)}

    def _is_a_in_b_support_circle(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        return self._in_circle(
            check["label_A"], check["label_B"], check["support_tag"], check.get("A_functional_tag"), check.get("radius")
        )

    def _is_all_a_in_b_support_circle(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        ok, each = True, {}
        for label in check["label_A"]:
            holds, each[label] = self._in_circle(
                label, check["label_B"], check["support_tag"], check.get("A_functional_tag")
            )
            ok &= holds
        return ok, {"each": each}

    def _is_pointa_in_b_functional_bbox(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        return self._in_quad(
            check["label_A"], check["label_B"], check.get("A_functional_tag"), check["B_functional_tag"]
        )

    def _is_all_pointa_in_b_functional_bbox(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        ok, each = True, {}
        for label in check["label_A"]:
            holds, each[label] = self._in_quad(
                label, check["label_B"], check.get("A_functional_tag"), check["B_functional_tag"]
            )
            ok &= holds
        return ok, {"each": each}

    def _is_a_point_above_b_point_by_z_range(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        A = self._a_points(check["label_A"], check.get("A_functional_tag"))
        B = self._a_points(check["label_B"], check.get("B_functional_tag"))
        dz = [float(p[2] - q[2]) for p in A for q in B]
        ok = any(check["z_lower"] < d < check["z_upper"] for d in dz)
        return ok, {"dz_m": sorted({round(d, 4) for d in dz})}

    def _is_ab_xy_distance_within_threshold(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        """A's centre (or its points ``A_functional_point``) within ``threshold`` of B's centre in xy (in B's own
        axes with ``axis="object"``)."""
        a, b = check["label_A"], check["label_B"]
        A = self._a_points(a, check.get("A_functional_point"))
        here_b = self._here(b)
        distances = []
        for p in A:
            delta = p - here_b
            if check.get("axis", "world") == "object":
                delta = self._frame(b).T @ delta
            distances.append(float(np.linalg.norm(delta[:2])))
        return any(d < check["threshold"] for d in distances), {"xy_m": [round(d, 4) for d in distances]}

    def _is_lift(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        """The object (its point ``point``) more than ``z_threshold`` above its reference (``pre_state``)."""
        label, tag = check["label"], check.get("point")
        _, reference = self._reference(label, tag)
        dz = float(self._at(label, tag)[2] - reference[2])
        lifted = dz > check["z_threshold"]
        return lifted if fn == "is_lift" else not lifted, {"dz_m": round(dz, 4)}

    def _is_moved(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        label, tag = check["label"], check.get("point")
        position = self._at(label, tag).copy()
        key, reference = self._reference(label, tag)
        moved = float(np.linalg.norm(position - reference))
        if live and check.get("update"):
            self.pre_state[key] = position
        ok = moved > check["dis_threshold"]
        return ok if fn == "is_moved" else not ok, {"moved_m": round(moved, 4)}

    def _update_object_state(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        label, tag = check["label"], check.get("point")
        position = self._at(label, tag).copy()
        if live:
            self.pre_state[label if tag is None else f"{label}:{tag}"] = position
        return True, {"z_m": round(float(position[2]), 4)}

    def _is_stacked(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        """``labels`` ordered by height, each next one within ``xy_threshold`` of the one under it in xy and at
        least ``z_min`` (and at most ``z_max``) higher (centres)."""
        here = {label: self._here(label) for label in check["labels"]}
        ordered = sorted(check["labels"], key=lambda n: here[n][2])
        gaps, ok = [], True
        for a, b in zip(ordered, ordered[1:]):
            xy = float(np.linalg.norm(here[a][:2] - here[b][:2]))
            dz = float(here[b][2] - here[a][2])
            gaps.append({"pair": [a, b], "xy_m": round(xy, 4), "dz_m": round(dz, 4)})
            ok &= xy <= check["xy_threshold"] and dz >= check["z_min"]
            ok &= check.get("z_max") is None or dz <= check["z_max"]
        return ok, {"order": ordered, "gaps": gaps}

    def _is_labels_axis_difference_in_range(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        axis = {"x": 0, "y": 1, "z": 2}[check["axis"]]
        values = [self._here(n)[axis] for n in check["labels"]]
        spread = float(max(values) - min(values))
        ok = (check.get("min_threshold") is None or spread >= check["min_threshold"]) and (
            check.get("max_threshold") is None or spread <= check["max_threshold"]
        )
        return ok, {"spread_m": round(spread, 4)}

    def _is_a_on_b_left(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        gap = float(self._here(check["label_B"])[0] - self._here(check["label_A"])[0])
        return gap > check["x_threshold"], {"x_gap_m": round(gap, 4)}

    def _is_a_before_b_on_axis(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        """B's centre more than ``min_gap`` further along ``axis`` (``x``/``y``/``z``, world frame) than A's."""
        axis = {"x": 0, "y": 1, "z": 2}[check["axis"]]
        gap = float(self._here(check["label_B"])[axis] - self._here(check["label_A"])[axis])
        return gap > check["min_gap"], {"gap_m": round(gap, 4)}

    def _is_a_xy_distance_close_to_pos(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        start = self._point(self._start, check["initial_of"])
        xy = float(np.linalg.norm(self._here(check["label"])[:2] - start[:2]))
        return xy < check["dis_threshold"], {"xy_m": round(xy, 4)}

    def _is_qpos_close(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        """A's axes within ``dis_threshold`` deg of B's now (``label_B``) or at the start (``initial_of``)."""
        if check.get("label_B"):
            other = self._frame(check["label_B"])
        else:
            other = self._frame(check["initial_of"], self._start)
        angle = _rotation_deg(self._frame(check["label_A"]), other)
        return angle < check["dis_threshold"], {"angle_deg": round(angle, 2)}

    def _is_all_gripper_open(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        if self._grippers is None:
            raise _Missing("no gripper openings")
        opening = {arm: round(float(v), 3) for arm, v in self._grippers.items()}
        return all(v >= check["open_threshold"] for v in opening.values()), {"opening": opening}

    def _all_robot_back_to_origin(self, fn: str, check: dict, live: bool) -> tuple[bool, dict]:
        ok, arms = True, {}
        for arm in self._ee_start:
            now_T, start_T = np.asarray(self._ee_now[arm]), np.asarray(self._ee_start[arm])
            offset = np.abs(now_T[:3, 3] - start_T[:3, 3])
            turn = _rotation_deg(now_T[:3, :3], start_T[:3, :3])
            arms[arm] = {"max_offset_m": round(float(offset.max()), 4), "turn_deg": round(turn, 2)}
            ok &= bool(np.all(offset <= check["pos_threshold"])) and turn <= check["rot_threshold"]
        return ok, arms
