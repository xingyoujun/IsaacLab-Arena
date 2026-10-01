# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Builtin collection references; extend here or add a family catalog without changing the runner."""

from pathlib import Path

from data_engine.harness.storage import read

ROOT = Path(__file__).resolve().parents[3]


def scenario_definitions():
    """Describe collection capabilities without importing Isaac, robot factories or a GPU library."""
    from data_engine.planning.task_plan import load_plan

    entries = {}
    for task in read(ROOT / "data_engine/pine_wm/tasks.json")["tasks"]:
        key = "pine_wm/" + task["task_id"]
        entries[key] = dict(
            id=key,
            adapter="pine_wm",
            environment="pine_wm_first20",
            embodiment="pine_wm_ur7e",
            scene="pine_wm",
            task_id=task["task_id"],
            instruction=task["name"],
            required_assets=["pine_wm_ur7e", "pine_wm", *task["asset_bindings"]],
            annotation_supported=task["task_id"] in {"T001", "T041"},
            preview_supported=True,
            registration_source="data_engine/pine_wm/tasks.json",
            layout=read(ROOT / "data_engine/pine_wm/review_layouts.json")["tasks"][task["task_id"]],
            action_profile={"width": 7, "semantic": "absolute_joint_position", "gripper": "finger_joint_rad"},
            capture_profile={"alignment": "post_step", "purpose": "preview", "fps": 15},
        )
    for name, task in read(ROOT / "data_engine/g2/tasks.json")["tasks"].items():
        key = "g2/" + name
        entries[key] = dict(
            id=key,
            adapter="g2",
            environment=task["environment"],
            embodiment="g2",
            scene="g2_workcell" if task["collector"] == "workcell" else "g2_tabletop",
            task_id=name,
            instruction=task["instruction"],
            required_assets=["g2", "g2_room", "g2_table", *task["assets"]],
            annotation_supported=False,
            preview_supported=True,
            registration_source="data_engine/g2/tasks.json",
            layout=task,
            action_profile={
                "width": 16,
                "semantic": "absolute_arm_joints_binary_grippers",
                "arm_order": ["right", "left"],
            },
            capture_profile={"alignment": "pre_step", "purpose": "bounded_replay", "fps": 15},
            task_plan=load_plan("g2", name),
        )
    for name, task in read(ROOT / "data_engine/interactions/tasks.json")["tasks"].items():
        key = "interactions/" + name
        entries[key] = dict(
            id=key,
            adapter="interactions",
            environment=task["environment"],
            embodiment=task["embodiment"],
            scene=task["scene"],
            task_id=name,
            instruction=task["instruction"],
            required_assets=["pine_wm_ur7e", "pine_wm", task["asset_id"]],
            annotation_supported=True,
            preview_supported=True,
            registration_source="data_engine/interactions/tasks.json",
            layout=task,
            action_profile={"width": 7, "semantic": "absolute_joint_position", "gripper": "finger_joint_rad"},
            capture_profile={"alignment": "post_step", "purpose": "preview", "fps": 15},
        )
    return entries


def source_patterns(adapter):
    """Declare family executable/config dependencies independently of pipeline internals."""
    shared = ["data_engine/motion/**/*.py", f"data_engine/{adapter}/**/*"]
    if adapter in {"pine_wm", "interactions"}:
        return shared + [
            "data_engine/pine_wm/collection/**/*",
            "isaaclab_arena/embodiments/ur7e/**/*",
            "isaaclab_arena_environments/*pine_wm*.py",
            "isaaclab_arena_environments/ur7e_workcell_environment.py",
        ]
    if adapter == "g2":
        return shared + [
            "isaaclab_arena/embodiments/g2/**/*",
            "data_engine/g2/collection/**/*",
            "isaaclab_arena_environments/g2*.py",
        ]
    raise ValueError(f"Unknown collection adapter: {adapter}")


def preflight_family(assets, scenario):
    """Run additional family contracts; common manifest validation is owned by the pipeline."""
    from data_engine.assets.interactions import audit_asset

    manifest = read(assets / "manifest.json")
    for name in scenario["required_assets"]:
        path = assets / manifest["entries"][name]["path"]
        if path.with_name("interaction_annotations.json").exists():
            report = audit_asset(path)
            assert not report["errors"], report
    if scenario["adapter"] == "g2":
        from data_engine.g2.cli import preflight

        preflight(assets, scenario["layout"])
