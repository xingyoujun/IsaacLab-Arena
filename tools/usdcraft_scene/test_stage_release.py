# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Release staging must preserve the published lock and reject concurrent remote changes."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tools.usdcraft_scene.manage import REPO, digest, upload, verify, write_json
from tools.usdcraft_scene.stage_release import stage


def test_candidate_does_not_modify_source_or_lock(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("original\n")
    manifest = dict(
        schema_version=1,
        repository=REPO,
        entries={"g2": dict(path="README.md")},
        files={"README.md": dict(bytes=9, sha256=digest(source / "README.md"))},
        local_extension=dict(base_manifest_sha256="a" * 64, published=False),
    )
    write_json(source / "manifest.json", manifest)
    lock = tmp_path / "release.json"
    write_json(lock, dict(repo_id=REPO, repo_type="dataset", revision="b" * 40, manifest_sha256="a" * 64))
    old_lock, old_manifest = lock.read_bytes(), (source / "manifest.json").read_bytes()
    plan = stage(source, tmp_path / "candidate", lock)
    assert lock.read_bytes() == old_lock and (source / "manifest.json").read_bytes() == old_manifest
    assert (source / "README.md").read_text() == "original\n"
    assert plan["expected_parent_revision"] == "b" * 40 and "revision" not in plan
    candidate = verify(tmp_path / "candidate/payload")
    assert "g2" in candidate["entries"] and "g2_task_registry" in candidate["entries"]
    with pytest.raises(AssertionError, match="replace release"):
        stage(source, tmp_path / "candidate", lock)
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(private=True, sha="c" * 40)
    with patch("huggingface_hub.HfApi", return_value=api):
        with pytest.raises(AssertionError, match="Remote advanced"):
            upload(
                SimpleNamespace(
                    directory=tmp_path / "candidate/payload", release_plan=tmp_path / "candidate/release-plan.json"
                )
            )
    api.upload_folder.assert_not_called()
    assert lock.read_bytes() == old_lock


def test_published_release_can_be_reused_without_duplicate_readme(tmp_path):
    source = tmp_path / "published"
    source.mkdir()
    (source / "README.md").write_text("# Assets\n\n## G2\n\nPublished assets.\n")
    write_json(
        source / "manifest.json",
        dict(
            schema_version=1,
            repository=REPO,
            entries={"g2": dict(path="README.md")},
            files={"README.md": dict(bytes=(source / "README.md").stat().st_size, sha256=digest(source / "README.md"))},
        ),
    )
    lock = tmp_path / "release.json"
    write_json(
        lock,
        dict(repo_id=REPO, repo_type="dataset", revision="b" * 40, manifest_sha256=digest(source / "manifest.json")),
    )
    stage(source, tmp_path / "candidate", lock)
    assert (tmp_path / "candidate/payload/README.md").read_bytes() == (source / "README.md").read_bytes()
    assert verify(tmp_path / "candidate/payload")["release_lineage"]["base_revision"] == "b" * 40


def test_upload_verifies_remote_bytes_before_changing_lock(tmp_path, monkeypatch):
    import shutil

    from tools.usdcraft_scene import manage

    source = tmp_path / "source"
    source.mkdir()
    (source / ".gitattributes").write_text("*.usd filter=lfs\n")
    write_json(
        source / "manifest.json",
        dict(
            schema_version=1,
            repository=REPO,
            entries={},
            files={
                ".gitattributes": dict(
                    bytes=(source / ".gitattributes").stat().st_size, sha256=digest(source / ".gitattributes")
                )
            },
        ),
    )
    lock = tmp_path / "release.json"
    lock.write_text('"previous verified revision"\n')
    monkeypatch.setattr(manage, "LOCK", lock)
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(private=True, sha="a" * 40)
    api.list_repo_files.return_value = [".gitattributes", "manifest.json"]
    api.upload_folder.return_value = SimpleNamespace(oid="b" * 40)

    def download(**kwargs):
        destination = Path(kwargs["local_dir"])
        for name in ("manifest.json", ".gitattributes"):
            shutil.copy2(source / name, destination / name)
        (destination / ".gitattributes").write_text("server changed this file\n")

    with (
        patch("huggingface_hub.HfApi", return_value=api),
        patch("huggingface_hub.snapshot_download", side_effect=download),
    ):
        with pytest.raises(AssertionError, match="Changed file: .gitattributes"):
            upload(SimpleNamespace(directory=source, release_plan=None))
    assert lock.read_text() == '"previous verified revision"\n'
