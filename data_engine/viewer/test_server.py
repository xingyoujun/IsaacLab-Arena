# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep supplementary diagnostics separate from immutable run results."""

import json

import pytest

from data_engine.viewer.presentation import group_tasks, timeline
from data_engine.viewer.server import page, rename_attempt, snapshot


def test_interaction_report_dict_keeps_failed_attempt_visible(tmp_path):
    run = tmp_path / "kettle"
    (run / "payload").mkdir(parents=True)
    manifest = dict(
        schema="arena.collection.run.v1",
        state="collection_failed",
        spec={"spec_id": "one", "scenario": {"id": "interactions/kettle_release"}},
        created_at="2026-09-30",
        events=[],
        artifacts={},
    )
    (run / "run.json").write_text(json.dumps(manifest))
    (run / "payload/results.json").write_text(
        json.dumps({"error": "contact blocked", "decisions": [{"body": "lid_release"}]})
    )
    records, _ = snapshot(tmp_path)
    assert records[0]["failure"] == "contact blocked"
    assert records[0]["decisions"] == [{"body": "lid_release"}]
    assert not records[0]["videos"]


def test_diagnostic_review_is_bound_to_spec_and_escaped(tmp_path):
    run = tmp_path / "one"
    run.mkdir()
    manifest = dict(
        schema="arena.collection.run.v1",
        state="preview_validated",
        spec={"spec_id": "current", "scenario": {"id": "g2/stack_bowls"}},
        created_at="2026-09-27",
        events=[],
        artifacts={},
    )
    source = json.dumps(manifest)
    (run / "run.json").write_text(source)
    review = dict(spec_id="stale", status="blocked", summary="<unsafe>")
    (run / "diagnostic_review.json").write_text(json.dumps(review))
    records, _ = snapshot(tmp_path)
    assert records[0]["diagnostic_review"] == {}
    review["spec_id"] = "current"
    (run / "diagnostic_review.json").write_text(json.dumps(review))
    records, allowed = snapshot(tmp_path)
    assert records[0]["state"] == "preview_validated"
    assert records[0]["diagnostic_review"]["status"] == "blocked"
    assert "/one/diagnostic_review.json" in allowed
    html = page(records).decode()
    assert "&lt;unsafe&gt;" in html and "<unsafe>" not in html
    assert (run / "run.json").read_text() == source


def test_collector_phases_bind_to_video_control_clock(tmp_path):
    run = tmp_path / "one"
    (run / "payload/raw").mkdir(parents=True)
    manifest = dict(
        schema="arena.collection.run.v1",
        state="preview_validated",
        spec={"spec_id": "current", "scenario": {"id": "g2/stack_bowls", "capture_profile": {"fps": 15}}},
        created_at="2026-09-27",
        events=[],
        artifacts={},
    )
    (run / "run.json").write_text(json.dumps(manifest))
    (run / "quality.json").write_text(json.dumps(dict(spec_id="current", raw=dict(steps=60))))
    report = dict(
        frames=60, phases=[dict(name="select", frame=0), dict(name="approach", frame=0), dict(name="close", frame=30)]
    )
    (run / "payload/raw/report.json").write_text(json.dumps(report))
    records, _ = snapshot(tmp_path)
    assert records[0]["stage_source"] == "collector_phases_preview_only"
    assert records[0]["step_dt"] == 1 / 15
    assert records[0]["stages"][-1] == dict(name="close", start_step=30, end_step=60)
    report["frames"] = 61
    (run / "payload/raw/report.json").write_text(json.dumps(report))
    assert snapshot(tmp_path)[0][0]["stages"] == []


def test_rename_is_persistent_spec_bound_and_does_not_edit_evidence(tmp_path):
    run = tmp_path / "attempt"
    (run / "payload").mkdir(parents=True)
    manifest = dict(
        schema="arena.collection.run.v1",
        state="collection_failed",
        spec={"spec_id": "current", "scenario": {"id": "pine_wm/T041"}},
        created_at="2026-09-27",
        events=[],
        artifacts={},
    )
    original = json.dumps(manifest)
    (run / "run.json").write_text(original)
    (run / "payload/proof").write_text("immutable")
    request = dict(run="attempt", spec_id="current", display_name="正面布局 </script><img src=x>")
    rename_attempt(tmp_path, request)
    records, _ = snapshot(tmp_path)
    assert records[0]["display_name"] == request["display_name"]
    assert "</script><img src=x>" not in page(records).decode()
    assert (run / "run.json").read_text() == original
    assert (run / "payload/proof").read_text() == "immutable"
    for invalid in [
        dict(request, run="../attempt"),
        dict(request, spec_id="stale"),
        dict(request, display_name="\n"),
        dict(request, display_name="x" * 81),
    ]:
        with pytest.raises(AssertionError):
            rename_attempt(tmp_path, invalid)


def test_tasks_prefer_usable_preview_and_keep_failures_and_blocked_attempts():
    records = []
    for index, (videos, status) in enumerate([(True, "reviewed"), (True, "blocked"), (False, "")]):
        records.append(
            dict(
                name=str(index),
                created_at=f"2026-09-27T0{index}:00:00",
                scenario="pine_wm/T041",
                task_title="抽屉",
                videos=[{}] if videos else [],
                diagnostic_review={"status": status},
            )
        )
    tasks = group_tasks(records)
    assert len(tasks) == 1
    assert tasks[0]["default_attempt"] == "0"
    assert [r["attempt_number"] for r in tasks[0]["attempts"]] == [3, 2, 1]


def test_playback_intervals_reject_overlap_and_exclude_planning_events():
    record = dict(
        quality={"raw": {"steps": 60}},
        step_dt=1 / 15,
        task_title="叠碗",
        scenario="g2/stack_bowls",
        stages=[
            dict(name="right: select_grasp bowl_1", start_step=0, end_step=0),
            dict(name="right: approach bowl_1", start_step=0, end_step=30),
            dict(name="right: close bowl_1", start_step=30, end_step=60),
        ],
    )
    result = timeline(record)
    assert len(result) == 2
    assert result[0]["end"] == result[1]["start"] == 2
    assert result[1]["label"] == "闭合夹爪"
    assert result[0]["group"] == "右手：右碗叠到中间碗"
    record["stages"][-1]["start_step"] = 29
    assert timeline(record) == []


def test_review_downloads_require_sealed_manifest_and_contained_path(tmp_path):
    from data_engine.viewer.server import review_files

    name = "kettle_release_v2_20260930_12"
    directory = tmp_path / name
    (directory / "payload").mkdir(parents=True)
    (directory / "payload/results.json").write_text("{}")
    outside = tmp_path / "private.hdf5"
    outside.write_text("private")
    (directory / "payload/demos.hdf5").symlink_to(outside)
    (directory / "run.json").write_text(json.dumps({"artifacts": {"results.json": {}, "demos.hdf5": {}}}))
    record = dict(name=name, state="preview_validated", spec_id="one", quality={"spec_id": "one"})
    allowed = review_files(tmp_path, [record])
    assert f"/{name}/payload/results.json" in allowed
    assert f"/{name}/payload/demos.hdf5" not in allowed
    record["quality"]["spec_id"] = "stale"
    assert f"/{name}/payload/results.json" not in review_files(tmp_path, [record])
    record.update(state="collection_failed", quality={"spec_id": "one"})
    assert f"/{name}/payload/results.json" not in review_files(tmp_path, [record])


def test_inventory_does_not_hide_new_asset_failures_behind_old_success():
    from data_engine.viewer.inventory import inventory, page

    sha = inventory([])["asset_manifest_sha256"]
    if sha is None:
        pytest.skip("Local asset bundle is external")
    records = [
        dict(
            scenario="pine_wm/T031",
            created_at="2026-09-28",
            name="old",
            state="preview_validated",
            asset_manifest_sha256="old",
            videos=[{}, {}, {}, {}],
            events=[],
        ),
        dict(
            scenario="pine_wm/T031",
            created_at="2026-09-30",
            name="new",
            state="collection_failed",
            asset_manifest_sha256=sha,
            videos=[],
            failure="<contact>",
            events=[],
        ),
    ]
    data = inventory(records)
    assert len(data["tasks"]) == 22
    task = next(t for t in data["tasks"] if t["id"] == "pine_wm/T031")
    assert task["run"] == "new" and task["cameras"] == 0 and task["historical_attempts"] == 1
    assert b"&lt;contact&gt;" in page(records)
    assert b"<contact>" not in page(records)
