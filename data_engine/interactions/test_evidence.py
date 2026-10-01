# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A terminal pose alone must not pass a multi-joint interaction task."""

import json
from pathlib import Path

from data_engine.interactions.evidence import evaluate


def test_toaster_requires_hold_then_release():
    task = json.loads(Path(__file__).with_name("tasks.json").read_text())["tasks"]["toaster_cancel"]
    samples = [dict(q=[0.06], engaged=[True])] + [dict(q=[0], engaged=[False])] * 20
    assert not evaluate(task, ["carriage_slide"], samples, [])["success"]
    wrong_order = [dict(event="released", step=1), dict(event="engaged", step=2)]
    assert not evaluate(task, ["carriage_slide"], samples, wrong_order)["success"]
    correct_order = [dict(event="engaged", step=1), dict(event="released", step=2)]
    assert evaluate(task, ["carriage_slide"], samples, correct_order)["success"]
    samples[-1] = dict(q=[0.03], engaged=[False])
    assert not evaluate(task, ["carriage_slide"], samples, correct_order)["success"]


def test_initially_open_kettle_is_not_success():
    task = json.loads(Path(__file__).with_name("tasks.json").read_text())["tasks"]["kettle_release"]
    samples = [dict(q=[1.39], engaged=[False])] * 20
    assert not evaluate(task, ["lid_hinge"], samples, [dict(event="released", step=1)])["success"]
