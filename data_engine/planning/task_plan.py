# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read task preparation decisions without starting a simulator or importing a planner."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_plan(family, task_id):
    """Return the declarative task plan, or explicitly identify an unmigrated recipe."""
    assert family in {"g2", "pine_wm", "interactions"}
    assert task_id and all(c.isalnum() or c == "_" for c in task_id)
    if family == "interactions":
        task = json.loads((ROOT / "data_engine/interactions/tasks.json").read_text())["tasks"][task_id]
        return dict(
            schema="arena.interaction_plan.v1",
            task_id=task_id,
            source="USD body-local interaction annotations",
            sequence=task["sequence"],
            mechanism="physics-step conditional joint limits with authored springs",
            success=task["success"],
            qualification="single preview; runtime evidence required",
        )
    path = ROOT / "data_engine" / family / "task_plans" / f"{task_id}.json"
    if not path.exists():
        return dict(
            schema="arena.task_plan.unmigrated.v1",
            task_id=task_id,
            status="legacy_recipe",
            reason="Task stages and grasp selection are still defined by the family recipe; no declarative plan yet.",
        )
    plan = json.loads(path.read_text())
    assert plan["schema"] == "arena.task_plan.v1" and plan["task_id"] == task_id
    assert plan["grasp"]["strategy"] == "priority_first_feasible"
    assert 0 < plan["contact_path"]["step_m"] <= 0.02
    assert 0 < plan["contact_path"]["tolerance_m"] <= 0.005
    assert 0 < plan["contact_path"]["angle_deg"] <= 10
    for step in plan["sequence"]:
        assert step["skill"] == "pick_place_rim" and step["arm"] in ("right", "left")
        assert plan["grasp"]["candidate_yaw_deg"][step["arm"]]
    return plan
