# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep preview, user review, stability qualification and collection separate."""

import json
from pathlib import Path

CONFIG = Path(__file__).with_name("review_layouts.json")


def validate_run(stage, task_ids):
    """Reject bulk runs until the user approves each revised task and its coverage."""
    assert stage == "preview", (
        "Tasks are awaiting layout review. Stability and collection are disabled until "
        "the user approves layouts, asset dimensions and randomization coverage."
    )
    assert len(task_ids) == 1, "Preview one task at a time, then wait for user feedback."
    config = json.loads(CONFIG.read_text())
    assert config["launch_authorized"], "Execution paused by user for layout review; do not launch new trials."
    task = config["tasks"][task_ids[0]]
    assert not task["asset_revision_required"], f"Asset/layout revision required before preview: {task['notes']}"
