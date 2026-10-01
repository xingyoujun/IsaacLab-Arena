# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Protect one implementation, lazy robot loading and complete provenance after relocation."""

import ast
import importlib
import subprocess
import sys

from data_engine.harness.catalog import ROOT, scenarios, source_files


def test_legacy_imports_share_module_identity():
    for old, new in (
        ("isaaclab_arena.collection.catalog", "data_engine.harness.catalog"),
        ("isaaclab_arena.annotations.skill_trace", "data_engine.annotations.skill_trace"),
        ("isaaclab_arena_cumotion.cumotion_embodiment_cfg", "data_engine.motion.cumotion.cumotion_embodiment_cfg"),
        (
            "isaaclab_arena_cumotion.embodiment_cumotion_registry",
            "data_engine.motion.cumotion.embodiment_cumotion_registry",
        ),
        ("tools.data_collection.g2", "data_engine.g2.cli"),
        ("tools.usdcraft_scene.manage", "data_engine.assets.manage"),
    ):
        assert importlib.import_module(old) is importlib.import_module(new)


def test_hardware_registry_does_not_load_robots_or_simulator():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from data_engine.motion.cumotion.embodiment_cumotion_registry import"
                " get_cumotion_cfg_by_name; assert not any(n.startswith(('isaaclab.', 'omni.',"
                " 'data_engine.motion.embodiments.')) for n in sys.modules)"
            ),
        ],
        cwd=ROOT,
        check=True,
    )


def test_canonical_modules_do_not_import_compatibility_packages():
    for path in (ROOT / "data_engine").rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(
                name.startswith((
                    "isaaclab_arena_cumotion",
                    "isaaclab_arena.collection",
                    "isaaclab_arena.annotations",
                    "tools.pine_wm.first20",
                    "tools.data_collection",
                    "tools.usdcraft_scene",
                ))
                for name in names
            ), str(path)


def test_source_identity_covers_shared_motion_and_family_recipe():
    for family in ("g2", "pine_wm"):
        files = source_files(family)
        assert "data_engine/motion/cumotion/planner.py" in files
        assert "data_engine/motion/cumotion/executor.py" in files
        assert f"data_engine/{family}/tasks.json" in files
        assert any(name.startswith(f"data_engine/{family}/collection/") for name in files)
    for spec in scenarios().values():
        assert spec["registration_source"].startswith("data_engine/")
        assert (ROOT / spec["registration_source"]).is_file()


def test_legacy_cli_and_module_cli_list_the_same_scenarios():
    canonical = subprocess.check_output([sys.executable, "-m", "data_engine.cli.harness", "list"], cwd=ROOT)
    legacy = subprocess.check_output([sys.executable, "tools/data_collection/harness.py", "list"], cwd=ROOT)
    assert canonical == legacy
