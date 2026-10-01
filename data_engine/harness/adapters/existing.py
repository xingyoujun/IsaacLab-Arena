# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Invoke qualified Arena entrypoints without replacing their controllers or environment factories."""

import sys

from data_engine.harness.catalog import ROOT


def preview_command(spec, run):
    """Return a serial single-episode command; never grant stability or collection approval."""
    scenario = spec["scenario"]
    assert scenario["preview_supported"], "Scenario is listed but not yet adapted for harness preview"
    if scenario["adapter"] == "interactions":
        return [
            sys.executable,
            "-m",
            "data_engine.interactions.collect",
            "--task",
            scenario["task_id"],
            "--output",
            str(run / "payload"),
            "--device",
            spec["device"],
            "--seed",
            str(spec["seed"]),
            "--harness-run",
            str(run),
            "--enable_cameras",
        ]
    if scenario["adapter"] == "pine_wm":
        return [
            sys.executable,
            str(ROOT / "data_engine/pine_wm/collection/qualify.py"),
            "--task",
            scenario["task_id"],
            "--trials",
            "1",
            *(["--annotate-skills"] if scenario["annotation_supported"] else []),
            "--enable_cameras",
            "--device",
            spec["device"],
            "--seed",
            str(spec["seed"]),
            "--output",
            str(run / "payload"),
            "--harness-run",
            str(run),
        ]
    if scenario["adapter"] == "g2":
        return [
            sys.executable,
            str(ROOT / "data_engine/g2/cli.py"),
            "run",
            "--task",
            scenario["task_id"],
            "--run-dir",
            str(run / "payload"),
            "--assets",
            spec["asset_root"],
            "--device",
            spec["device"],
            "--seed",
            str(spec["seed"]),
        ]
    raise ValueError(f"No execution adapter: {scenario['adapter']}")
