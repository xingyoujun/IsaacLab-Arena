# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Guard the unified entrypoint against asset fallback and accidental episode replacement."""

import json

import pytest

from data_engine.assets.manage import digest
from data_engine.g2.cli import TASKS, main, preflight


def test_missing_g2_bundle_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("GENIESIM_ASSETS_DIR", "/old/path")
    monkeypatch.setenv("ARENA_USDCRAFT_SCENE_ROOT", "")
    manifest = dict(schema_version=1, files={}, entries={}, repository="xingyoujun/USDCraft-Scene")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="Asset ID absent"):
        preflight(tmp_path, TASKS["stack_bowls"])


def test_modified_asset_fails_before_simulation(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_USDCRAFT_SCENE_ROOT", "")
    (tmp_path / "robot.usd").write_text("original")
    manifest = dict(
        schema_version=1, files={"robot.usd": dict(bytes=8, sha256=digest(tmp_path / "robot.usd"))}, entries={}
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "robot.usd").write_text("modified")
    with pytest.raises(AssertionError, match="Changed file"):
        preflight(tmp_path, TASKS["stack_bowls"])


def test_cpu_collection_rejected_before_launch(tmp_path):
    with pytest.raises(AssertionError, match="CUDA"):
        main(["collect", "--task", "stack_bowls", "--run-dir", str(tmp_path), "--device", "cpu"])


def test_existing_run_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr("data_engine.g2.cli.preflight", lambda *args: {})
    with pytest.raises(FileExistsError):
        main(["collect", "--task", "stack_bowls", "--run-dir", str(tmp_path)])


@pytest.mark.parametrize("warning", ["", "PhysX: collision detection will fall back to CPU. Prim /blue_bin"])
def test_task_success_does_not_claim_cuda_only_execution(tmp_path, monkeypatch, warning):
    from data_engine.g2.collection import validation

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "episodes.hdf5").write_bytes(b"fixture: raw audit is mocked for the device-report test")
    (tmp_path / "run.json").write_text(
        json.dumps(dict(task_id="clean_workcell_table", device="cuda:0", assets=dict(manifest_sha256="a" * 64)))
    )
    (tmp_path / "collect.log").write_text(warning)
    monkeypatch.setattr(validation, "audit_raw", lambda *args: dict(raw_audit_pass=True, task_success=True, steps=1))
    result = validation.audit_run(tmp_path)
    assert result["task_success"]
    assert result["task_id"] == "clean_workcell_table"
    device = result["runtime_device_audit"]
    assert device["cpu_collision_fallback_detected"] == bool(warning)
    assert not device["native_planner_cuda_verified"] and not device["all_cuda_verified"]
    if warning:
        assert device["cpu_collision_fallback_warnings"] == [dict(stage="collect", message=warning)]


def test_raw_audit_task_identity_cannot_be_relabelled(tmp_path, monkeypatch):
    from data_engine.g2.collection import validation

    (tmp_path / "run.json").write_text(json.dumps(dict(task_id="clean_workcell_table")))
    monkeypatch.setattr(validation, "audit_raw", lambda *args: dict(task_id="stack_bowls"))
    with pytest.raises(AssertionError, match="task identity differs"):
        validation.audit_run(tmp_path)
