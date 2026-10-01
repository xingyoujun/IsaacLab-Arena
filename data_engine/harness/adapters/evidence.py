# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Family-specific evidence checks for existing Arena collectors."""

import math
from pathlib import Path

from data_engine.harness.audit import check_raw
from data_engine.harness.storage import inside, read


def audit_payload(spec, payload):
    """Audit family evidence with explicit image timing; do not rewrite or relabel source files."""
    from data_engine.recording.alignment import validate_video

    payload = Path(payload).resolve()
    scenario = spec["scenario"]
    fps = scenario["capture_profile"]["fps"]
    if scenario["adapter"] == "pine_wm":
        result = read(payload / "results.json")
        assert len(result) == 1 and result[0]["success"]
        result = result[0]
        assert result["task"] == scenario["task_id"]
        assert result["seed"] == spec["seed"]
        assert read(payload / "layout_snapshot.json") == scenario["layout"], "Layout differs from resolved scenario"
        assert not result["forbidden_contacts"], "Forbidden robot/tool contact"
        raw = check_raw(payload / "demos.hdf5", scenario["action_profile"]["width"])
        replay = read(payload / "live_preview/trial_0.json")
        assert replay["validation_passed"] and replay["frames"] == raw["steps"]
        assert replay["mode"] == "live_sensor_post_step_preview" and replay["fps"] == fps
        assert set(replay["files"]) == {"realsense_d435", "wrist_a", "wrist_b", "scene_cam"}
        videos = {}
        for name, relative in replay["files"].items():
            path = inside(payload / "live_preview", relative)
            shape = (720, 1280) if name == "scene_cam" else (480, 640)
            videos[name] = dict(
                path=path.relative_to(payload).as_posix(), **validate_video(path, raw["steps"], fps, shape)
            )
        if scenario["annotation_supported"]:
            assert result.get("annotation_alignment_checked") and raw["annotations"]["status"] == "pass"
        task_evidence = result.get("metrics", {})
        training_alignment = {"status": "fail", "reason": "Post-step review videos are not pre-action training images"}
    elif scenario["adapter"] == "interactions":
        import h5py
        import numpy as np

        from data_engine.interactions.evidence import evaluate

        result = read(payload / "results.json")
        assert result["success"] and result["mode"] == "robot_contact"
        assert result["task"] == scenario["task_id"] and result["seed"] == spec["seed"]
        assert read(payload / "layout_snapshot.json") == scenario["layout"]
        assert not result["forbidden_contacts"] and result["contact_events"] > 0
        raw = check_raw(payload / "demos.hdf5", scenario["action_profile"]["width"])
        assert raw["annotations"]["status"] == "pass"
        assert len(result["samples"]) == raw["steps"]
        with h5py.File(payload / "demos.hdf5") as file:
            demo = file["data/demo_0"]
            kind = scenario["layout"]["kind"]
            joint_states = demo[f"states/articulation/{kind}/joint_position"][:]
            np.testing.assert_allclose(joint_states, [row["q"] for row in result["samples"]], atol=1e-6)
            mechanism = demo["mechanism"]
            assert mechanism.attrs["asset"] == kind
            post = mechanism["latch_engaged_post"][:]
            pre = mechanism["latch_engaged_pre"][:]
            np.testing.assert_array_equal(post, [row["engaged"] for row in result["samples"]])
            np.testing.assert_array_equal(pre[1:], post[:-1])
            assert pre.shape == post.shape == (raw["steps"], 1)
            assert bool(pre[0, 0]) == (kind == "kettle")
            if kind == "kettle":
                assert "kettle_power_base" in demo["states/rigid_object"], "Disconnected base was not recorded"
        task_evidence = evaluate(scenario["layout"], result["joint_names"], result["samples"], result["events"])
        assert task_evidence["success"]
        replay = read(payload / "live_preview/trial_0.json")
        assert replay["validation_passed"] and replay["frames"] == raw["steps"]
        assert replay["mode"] == "live_sensor_post_step_preview" and replay["fps"] == fps
        assert set(replay["files"]) == {"realsense_d435", "wrist_a", "wrist_b", "scene_cam"}
        videos = {}
        for name, relative in replay["files"].items():
            path = inside(payload / "live_preview", relative)
            shape = (720, 1280) if name == "scene_cam" else (480, 640)
            videos[name] = dict(
                path=path.relative_to(payload).as_posix(), **validate_video(path, raw["steps"], fps, shape)
            )
        training_alignment = {
            "status": "fail",
            "reason": "Post-step mechanism preview is not pre-action training imagery",
        }
    elif scenario["adapter"] == "g2":
        from data_engine.g2.collection.validation import audit_run

        family = audit_run(payload)
        assert family["task_id"] == scenario["task_id"] and family["replay_passed"] and family["export_passed"]
        raw = check_raw(payload / "raw/episodes.hdf5", scenario["action_profile"]["width"])
        replay = read(payload / "render/render_report.json")
        assert replay["source_hdf5_sha256"] == raw["sha256"]
        videos = {}
        for name, shape in {"head": (400, 640), "left_wrist": (528, 640), "right_wrist": (528, 640)}.items():
            relative = f"render/{name}_camera.mp4"
            videos[name] = dict(path=relative, **validate_video(payload / relative, raw["steps"], fps, shape))
        task_evidence = family
        training_alignment = {
            "status": "unknown",
            "reason": "Bounded physics-microstep replay needs task-specific acceptance",
            "drift": replay.get("limits"),
        }
    else:
        raise ValueError("No audit adapter")
    assert math.isfinite(float(fps)) and fps > 0
    return dict(
        schema="arena.collection.quality.v1",
        spec_id=spec["spec_id"],
        task_success=True,
        preview_valid=True,
        raw=raw,
        videos=videos,
        task_evidence=task_evidence,
        training_alignment=training_alignment,
        gpu={"status": "unknown", "reason": "Requested CUDA is not independent proof of planner and collision device"},
        review={"status": "pending", "spec_id": spec["spec_id"]},
        training_eligible=False,
        stability_qualified=False,
        batch_enabled=False,
    )
