# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G2-Eval's pack check, success relations and layout draws, without Isaac Sim."""

import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

import pytest

from isaaclab_arena.g2_eval import layout, pack
from isaaclab_arena.g2_eval.success import evaluate


def _box(size) -> list[list[float]]:
    """The corners of a box standing on z = 0, centred in x and y."""
    sx, sy, sz = (v / 2 for v in size)
    return [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (0.0, 2 * sz)]


def _pose(x, y, z, roll_deg=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("x", roll_deg, degrees=True).as_matrix()
    T[:3, 3] = (x, y, z)
    return T


GEOMETRY = {
    "block": {"hull_m": _box((0.04, 0.04, 0.05)), "receive_m": None},
    "bin": {"hull_m": _box((0.16, 0.12, 0.08)), "receive_m": [0.0, 0.0, 0.08]},
}


@pytest.mark.parametrize(
    "relation, block, holds",
    [
        ("in", _pose(0.0, 0.0, 0.01), True),  # bottom 70 mm below the opening, on its axis
        ("in", _pose(0.05, 0.0, 0.01), True),  # 50 mm off the axis, within the 60 mm half-width
        ("in", _pose(0.07, 0.0, 0.01), False),
        ("in", _pose(0.0, 0.0, 0.09), False),  # above the opening
        ("on", _pose(0.0, 0.0, 0.085), True),  # 5 mm above the top
        ("on", _pose(0.0, 0.0, 0.1), False),
        ("on", _pose(0.0, 0.0, 0.085, roll_deg=10), False),  # tilted
        ("upright", _pose(0.3, 0.0, 0.0, roll_deg=4), True),
        ("upright", _pose(0.3, 0.0, 0.0, roll_deg=30), False),
    ],
)
def test_success_relations(relation, block, holds):
    entry = {"relation": relation, "object": "block", "target": None if relation == "upright" else "bin"}
    (result,) = evaluate([entry], GEOMETRY, {"block": block, "bin": _pose(0.0, 0.0, 0.0)})
    assert result["holds"] == holds, result


def test_success_counts_nesting_and_refuses_unknown_relations():
    geometry = {
        "block": GEOMETRY["block"],
        "cup": {"hull_m": _box((0.07, 0.07, 0.06)), "receive_m": [0.0, 0.0, 0.06]},
        "bin": {"hull_m": _box((0.2, 0.2, 0.08)), "receive_m": [0.0, 0.0, 0.08]},
    }
    poses = {"block": _pose(0.06, 0.0, 0.02), "cup": _pose(0.06, 0.0, 0.01), "bin": _pose(0.0, 0.0, 0.0)}
    (result,) = evaluate([{"relation": "in", "object": "block", "target": "bin"}], geometry, poses)
    assert result["holds"]
    with pytest.raises(AssertionError, match="no G2-Eval evaluator"):
        evaluate([{"relation": "seated", "object": "block", "target": "bin"}], geometry, poses)


OBJECTS = {
    "apple": {"xy_m": [-0.08, 0.22], "radius_m": 0.046, "symmetric": True, "yaw_deg": 0.0},
    "box": {"xy_m": [0.04, 0.02], "radius_m": 0.08, "symmetric": False, "yaw_deg": 0.0},
}


def test_layout_draws_are_reproducible_and_accepted():
    workspace = [-0.22, 0.18, -0.4, 0.4]
    turns = []
    for seed in range(20):
        drawn = layout.sample(OBJECTS, np.random.default_rng([seed, 0, 0]), 0.05, 30.0, workspace)
        assert drawn == layout.sample(OBJECTS, np.random.default_rng([seed, 0, 0]), 0.05, 30.0, workspace)
        assert drawn["apple"]["dyaw_deg"] == 0.0  # symmetric
        assert all(abs(v) <= 0.05 for d in drawn.values() for v in d["dxy_m"])
        centre = {n: np.asarray(s["xy_m"], float) for n, s in OBJECTS.items()}
        assert layout.accepted(OBJECTS, drawn, centre, workspace, 0.02)
        turns.append(drawn["box"]["dyaw_deg"])
    assert max(abs(t) for t in turns) <= 30.0 and len(set(turns)) > 1
    fixed = {**OBJECTS, "box": {**OBJECTS["box"], "fixed": True}}
    assert layout.sample(fixed, np.random.default_rng(0), 0.05, 30.0)["box"] == {"dxy_m": [0.0, 0.0], "dyaw_deg": 0.0}
    reach = {"step_m": 0.02, "cells_xy_m": [[-0.08, 0.22]]}
    for seed in range(5):
        drawn = layout.sample(
            OBJECTS,
            np.random.default_rng(seed),
            0.05,
            30.0,
            region={"reach": reach},
            per_object={"box": {"fixed": True}},
        )
        assert layout.reached(reach, np.asarray(OBJECTS["apple"]["xy_m"]) + drawn["apple"]["dxy_m"])
    with pytest.raises(AssertionError, match="no layout"):
        layout.sample(OBJECTS, np.random.default_rng(0), 0.05, 30.0, workspace_m=[5.0, 6.0, 5.0, 6.0])


def _write_pack(root, lock_path):
    (root / "tasks/pick_apple").mkdir(parents=True)
    (root / "common").mkdir()
    files = {
        "tasks/pick_apple/task.json": {
            "schema": pack.TASK_SCHEMA,
            "name": "pick_apple",
            "platform": "common/platform.json",
            "geometry": "geometry.json",
        },
        "tasks/pick_apple/geometry.json": {"objects": GEOMETRY},
        "common/platform.json": {"table_top_z_m": 0.0},
    }
    records = {}
    for relative, value in files.items():
        (root / relative).write_text(json.dumps(value))
        data = (root / relative).read_bytes()
        records[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"schema_version": 1, "pack": pack.PACK_SCHEMA, "tasks": ["pick_apple"], "files": records}
    (root / "manifest.json").write_text(json.dumps(manifest))
    lock = {"repo_id": None, "revision": None, "manifest_sha256": pack.digest(root / "manifest.json")}
    lock_path.write_text(json.dumps({**lock, "tasks": ["pick_apple"]}))


def test_pack_is_checked_against_the_lock(tmp_path, monkeypatch):
    root, lock_path = tmp_path / "G2-Eval", tmp_path / "lock.json"
    _write_pack(root, lock_path)
    monkeypatch.setattr(pack, "LOCK", lock_path)
    monkeypatch.setenv(pack.ROOT_ENV, str(root))
    pack.verified_root.cache_clear()
    pack.load_task.cache_clear()
    loaded = pack.load_task("pick_apple")
    assert loaded["geometry"]["bin"]["receive_m"] == [0.0, 0.0, 0.08] and loaded["platform"]["table_top_z_m"] == 0.0
    with pytest.raises(AssertionError, match="not a task of the locked"):
        pack.load_task("pick_orange")
    with pytest.raises(AssertionError, match="no published G2-Eval revision"):
        pack.download(tmp_path / "download")

    (root / "common/platform.json").write_text("{}")
    with pytest.raises(AssertionError, match="changed file"):
        pack.verify(root)
    (root / "manifest.json").write_text("{}")
    with pytest.raises(AssertionError, match="not the locked G2-Eval pack"):
        pack.verify(root)
    pack.verified_root.cache_clear()
    pack.load_task.cache_clear()


def test_legacy_relations():
    still = {"lin_vel": {"block": [0.0] * 3, "bin": [0.0] * 3}, "ang_vel": {"block": [0.0] * 3, "bin": [0.0] * 3}}
    poses = {"block": _pose(0.01, 0.0, 0.05), "bin": _pose(0.0, 0.0, 0.0)}
    placed = {
        "relation": "placed",
        "object": "block",
        "target": "bin",
        "params": {"force_threshold_n": 0.1, "speed_max_m_s": 0.1, "max_separation_m": [0.05, 0.05, 0.1]},
    }
    touching = {**still, "contact_n": {"block>bin": 2.0}}
    assert evaluate([placed], GEOMETRY, poses, touching)[0]["holds"]
    assert not evaluate([placed], GEOMETRY, poses, {**still, "contact_n": {"block>bin": 0.0}})[0]["holds"]
    fast = {**touching, "lin_vel": {**still["lin_vel"], "block": [0.2, 0.0, 0.0]}}
    assert not evaluate([placed], GEOMETRY, poses, fast)[0]["holds"]
    assert not evaluate([placed], GEOMETRY, {**poses, "block": _pose(0.06, 0.0, 0.05)}, touching)[0]["holds"]
    above = {"relation": "above", "object": "block", "target": "bin"}
    assert evaluate([above], GEOMETRY, poses, still)[0]["holds"]

    inserted = {
        "relation": "inserted",
        "object": "block",
        "target": "bin",
        "params": {"radial_tolerance_m": 0.001, "seating_tolerance_m": 0.003, "axis_tolerance_deg": 1.0},
    }
    for block, holds in (
        (_pose(0.0005, 0.0, 0.002), True),
        (_pose(0.002, 0.0, 0.0), False),
        (_pose(0, 0, 0, 2), False),
    ):
        assert evaluate([inserted], GEOMETRY, {**poses, "block": block}, still)[0]["holds"] == holds

    geometry = {**GEOMETRY, "block": {**GEOMETRY["block"], "box_m": [[-0.02, -0.02, 0.0], [0.02, 0.02, 0.05]]}}
    in_bin = {
        "relation": "in_bin",
        "object": "block",
        "target": "bin",
        "params": {
            "inner_xy_bounds_m": [[-0.07, -0.05], [0.07, 0.05]],
            "floor_z_m": 0.002,
            "rim_max_z_m": 0.08,
            "planar_margin_m": 0.003,
            "floor_tolerance_m": 0.012,
            "rim_tolerance_m": 0.003,
            "max_linear_speed_m_s": 0.01,
            "max_angular_speed_rad_s": 0.1,
        },
    }
    for block, holds in ((_pose(0.0, 0.0, 0.003), True), (_pose(0.06, 0.0, 0.003), False), (_pose(0, 0, 0.03), False)):
        assert evaluate([in_bin], geometry, {**poses, "block": block}, still)[0]["holds"] == holds
    turning = {**still, "ang_vel": {**still["ang_vel"], "block": [0.0, 0.0, 0.5]}}
    assert not evaluate([in_bin], geometry, {**poses, "block": _pose(0.0, 0.0, 0.003)}, turning)[0]["holds"]


def test_uniform_layout_keeps_the_legacy_constraints():
    nominal = {"bowl_1": [-0.12, -0.24], "bowl_2": [-0.10, 0.0], "bowl_3": [-0.12, 0.24]}
    for seed in range(50):
        drawn = layout.sample_uniform(
            nominal, np.random.default_rng(seed), 0.02, list(nominal), 0.18, [0.30, 0.60], 0.079
        )
        moved = [np.asarray(nominal[n]) + drawn[n]["dxy_m"] for n in nominal]
        assert all(np.all(np.abs(np.asarray(d["dxy_m"])) <= 0.02) and d["dyaw_deg"] == 0.0 for d in drawn.values())
        assert all(np.linalg.norm(a - b) >= 0.18 for i, a in enumerate(moved) for b in moved[i + 1 :])
    one = layout.sample_uniform({"peg": [0.05, -0.22], "sleeve": [-0.1, 0.0]}, np.random.default_rng(0), 0.02, ["peg"])
    assert one["sleeve"]["dxy_m"] == [0.0, 0.0] and one["peg"]["dxy_m"] != [0.0, 0.0]
    with pytest.raises(AssertionError, match="no layout"):
        layout.sample_uniform(nominal, np.random.default_rng(0), 0.02, list(nominal), 0.5)
