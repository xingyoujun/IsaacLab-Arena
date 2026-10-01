# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only evidence audit; successful previews do not imply training or CUDA qualification."""

import json
from pathlib import Path

from data_engine.harness.storage import digest, read, verify_snapshot, write


def check_trace(demo, count):
    """Validate saved metadata plus per-transition labels, when this recording has annotations."""
    from data_engine.annotations.skill_trace import validate_trace

    if "annotations" not in demo:
        return {"status": "unknown", "reason": "Collector has no skill annotation adapter"}
    doc = json.loads(demo["annotations/metadata_json"][()])
    for key in ("step_skill_ids", "step_stage_ids"):
        doc[key] = demo["annotations/" + key][:].tolist()
    validate_trace(doc, count)
    return {"status": "pass", "skills": len(doc["skills"]), "stages": len(doc["stages"])}


def check_raw(path, width):
    """Check complete finite transitions, pre-step observations and any embedded labels."""
    import h5py
    import numpy as np

    from data_engine.recording.alignment import pre_step_states

    with h5py.File(path) as file:
        assert list(file["data"]) == ["demo_0"], "Pilot adapter expects exactly one episode"
        demo = file["data/demo_0"]
        count = len(demo["actions"])
        assert count > 0 and demo["actions"].shape == (count, width)
        assert bool(demo.attrs.get("success", False)), "Episode is not a successful preview"
        pre_step_states(demo)
        if "joint_pos_target" in demo:
            assert np.allclose(
                demo["actions"][:], demo["joint_pos_target"][:], atol=1e-6, rtol=0
            ), "Action/target mismatch"

        def finite(name, value):
            if isinstance(value, h5py.Dataset) and np.issubdtype(value.dtype, np.number):
                assert np.isfinite(value[()]).all(), name

        demo.visititems(finite)
        labels = check_trace(demo, count)
    return {"status": "pass", "steps": count, "sha256": digest(path), "annotations": labels}


def audit_run(run):
    """Audit sealed bytes, recording a quality result without granting user approval."""
    run = Path(run).resolve()
    manifest = read(run / "run.json")
    assert manifest["state"] in {"raw_sealed", "preview_validated", "audit_failed"}
    verify_snapshot(run / "payload", manifest["artifacts"])
    from data_engine.harness.adapters.evidence import audit_payload

    report = audit_payload(manifest["spec"], run / "payload")
    write(run / "quality.json", report)
    return report
