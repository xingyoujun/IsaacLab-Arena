# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare a complete HF release candidate and review plan without network writes."""

import argparse
import json
import shutil
from pathlib import Path

from tools.usdcraft_scene.manage import LOCK, REPO, ROOT, digest, verify, write_json


def stage(source, output, lock_path=LOCK):
    """Stage an additive, immutable local candidate; the published lock stays unchanged."""
    manifest = verify(source)
    lock = json.loads(lock_path.read_text())
    assert not output.exists(), f"Refusing to replace release candidate: {output}"
    assert manifest["repository"] == lock["repo_id"] == REPO
    if "local_extension" in manifest:
        assert manifest["local_extension"]["base_manifest_sha256"] == lock["manifest_sha256"]
    else:
        assert digest(source / "manifest.json") == lock["manifest_sha256"], "Source must match the published release"
    output.mkdir(parents=True)
    payload = output / "payload"
    payload.mkdir()
    for relative in manifest["files"]:
        target = payload / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    entries = manifest["entries"]
    tasks = json.loads((ROOT / "isaaclab_arena_cumotion/g2_collection/tasks.json").read_text())
    snapshots = {
        "scenes/g2/tasks.json": ROOT / "isaaclab_arena_cumotion/g2_collection/tasks.json",
        "scenes/g2/motion_config_cumotion.yaml": (
            ROOT / "isaaclab_arena_cumotion/g2_collection/workcell/motion_config_cumotion.yaml"
        ),
        "embodiments/g2/calibration/g2.py": ROOT / "isaaclab_arena/embodiments/g2/g2.py",
    }
    for relative, path in snapshots.items():
        target = payload / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    entries["g2_task_registry"] = dict(path="scenes/g2/tasks.json", kind="task_registry")
    entries["g2_camera_calibration"] = dict(path="embodiments/g2/calibration/g2.py", kind="calibration_snapshot")
    entries["g2_workcell_motion"] = dict(path="scenes/g2/motion_config_cumotion.yaml", kind="planning_config")
    for name, task in tasks["tasks"].items():
        if name == "clean_workcell_table":
            continue  # Preserve the existing geometry/configuration descriptor.
        relative = f"scenes/g2/{name}.json"
        write_json(payload / relative, dict(schema_version=1, embodiment="g2", **task))
        entries[f"g2_{name}_scene"] = dict(path=relative, kind="scene")
    if "## G2" not in (payload / "README.md").read_text():
        with (payload / "README.md").open("a") as stream:
            stream.write(
                "\n## G2\n\n- `embodiments/g2/`: robot USD closure, planning URDFs and vendor collision spheres.\n-"
                " `embodiments/g2/calibration/`: Git calibration snapshot; never execute code downloaded from HF.\n-"
                " `scenes/g2/`: task registry, scene configuration and native cuMotion workcell parameters.\n- G2"
                " shared objects are resolved by manifest IDs; internal USD references are relative.\n\nUse Arena"
                " `tools/data_collection/g2.py`; local episode outputs are not part of this asset repository. Preserve"
                " upstream source attribution and licensing. This bundle does not grant new redistribution rights.\n"
            )
    # Register binary texture types up front so HF does not silently append LFS
    # rules after the manifest has been computed.
    attributes = payload / ".gitattributes"
    patterns = attributes.read_text() if attributes.exists() else ""
    for suffix in ("jpg", "jpeg"):
        rule = f"*.{suffix} filter=lfs diff=lfs merge=lfs -text"
        if rule not in patterns.splitlines():
            patterns = patterns.rstrip("\n") + "\n" + rule + "\n"
    attributes.write_text(patterns)
    manifest.pop("local_extension", None)
    manifest["release_lineage"] = dict(base_revision=lock["revision"], base_manifest_sha256=lock["manifest_sha256"])
    for path in sorted(payload.rglob("*")):
        if path.is_file():
            manifest["files"][path.relative_to(payload).as_posix()] = dict(
                bytes=path.stat().st_size, sha256=digest(path)
            )
    write_json(payload / "manifest.json", manifest)
    verify(payload)
    plan = dict(
        schema_version=1,
        status="local_candidate_not_uploaded",
        repo_id=REPO,
        repo_type="dataset",
        expected_parent_revision=lock["revision"],
        base_manifest_sha256=lock["manifest_sha256"],
        manifest_sha256=digest(payload / "manifest.json"),
        file_count=len(manifest["files"]),
        total_bytes=sum(value["bytes"] for value in manifest["files"].values()),
        task_ids=sorted(tasks["tasks"]),
        instructions=(
            "After explicit publication authorization, upload with --release-plan; "
            "only the successful remote commit may update release.json. No raw episodes belong in this release."
        ),
    )
    write_json(output / "release-plan.json", plan)
    print(json.dumps(plan, indent=2))
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage(args.source, args.output)
