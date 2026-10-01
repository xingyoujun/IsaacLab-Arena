# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordered, measured mechanism success independent of the motion recipe."""


def evaluate(task, joint_names, samples, events):
    """Require a held latch, a measured release, and sustained passive endpoint motion."""
    import numpy as np

    q = np.asarray([sample["q"] for sample in samples])
    assert len(q) >= task["hold_steps"] and np.isfinite(q).all()
    goal = task["success"]
    index = joint_names.index(goal["final_joint"])
    tail = q[-task["hold_steps"] :, index]
    releases = [event for event in events if event["event"] == "released"]
    engaged = [event for event in events if event["event"] == "engaged"]
    held_samples = [sample for sample in samples if sample["engaged"][0]]
    prior_hold = bool(held_samples)
    if task["kind"] == "toaster":
        prior_hold = prior_hold and any(event["step"] < release["step"] for event in engaged for release in releases)
    metrics = dict(
        release_count=len(releases),
        engagement_count=len(engaged),
        prior_hold=prior_hold,
        final_hold_min=float(tail.min()),
        final_hold_max=float(tail.max()),
        final_joint=goal["final_joint"],
        final_latch_released=not samples[-1]["engaged"][0],
    )
    metrics["success"] = bool(
        releases
        and prior_hold
        and metrics["final_latch_released"]
        and np.all(tail >= goal["min"])
        and np.all(tail <= goal["max"])
    )
    return metrics
