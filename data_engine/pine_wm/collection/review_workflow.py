# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep preview, user review, stability qualification and collection separate."""

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[3] / "data_engine/pine_wm/review_layouts.json"


def validate_run(stage, task_ids, annotation_preview=False, harness_run=None):
    """Reject bulk runs until the user approves each revised task and its coverage."""
    assert stage == "preview", (
        "Tasks are awaiting layout review. Stability and collection are disabled until "
        "the user approves layouts, asset dimensions and randomization coverage."
    )
    assert len(task_ids) == 1, "Preview one task at a time, then wait for user feedback."
    config = json.loads(CONFIG.read_text())
    if harness_run is not None:
        from data_engine.harness.catalog import validate_spec

        manifest = json.loads((Path(harness_run) / "run.json").read_text())
        assert manifest["schema"] == "arena.collection.run.v1" and manifest["state"] == "collecting"
        spec = manifest["spec"]
        validate_spec(spec)
        assert spec["scenario"]["id"] == "pine_wm/" + task_ids[0]
        assert spec["scenario"]["preview_supported"]
        assert annotation_preview == spec["scenario"]["annotation_supported"]
    elif annotation_preview:
        assert config.get("annotation_preview_authorized"), "Annotation pilot is not authorized"
        assert set(task_ids) <= set(config.get("annotation_preview_tasks", [])), "Task outside annotation pilot"
    else:
        assert config["launch_authorized"], "Execution paused by user for layout review; do not launch new trials."
    task = config["tasks"][task_ids[0]]
    assert not task["asset_revision_required"], f"Asset/layout revision required before preview: {task['notes']}"
