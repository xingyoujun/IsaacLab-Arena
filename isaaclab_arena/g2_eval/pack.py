# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The G2-Eval asset pack: the G2 platform and the evaluation tasks, pinned by ``lock.json``.

The pack is produced by the data engine (SimEngine ``data_engine/assets/g2_eval.py``) and published as an HF dataset.
``lock.json`` names the revision this checkout evaluates with, its manifest digest and its tasks; each task registers
an environment of the same name (``isaaclab_arena_environments/g2_eval_environments.py``). The pack lives in
``local_assets/G2-Eval/`` unless ``G2_EVAL_ASSETS_DIR`` names another directory. Nothing is downloaded implicitly and
no code is executed from the pack.

    python -m isaaclab_arena.g2_eval.pack download   # the locked revision, every file checked
    python -m isaaclab_arena.g2_eval.pack verify
"""

import argparse
import functools
import hashlib
import json
import os
import re
from pathlib import Path

LOCK = Path(__file__).with_name("lock.json")
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "local_assets" / "G2-Eval"
ROOT_ENV = "G2_EVAL_ASSETS_DIR"
PACK_SCHEMA = "g2.eval.pack.v1"
TASK_SCHEMA = "g2.eval.task.v1"


def lock() -> dict:
    """The pinned pack: ``repo_id``, ``revision``, ``manifest_sha256`` and ``tasks``."""
    return json.loads(LOCK.read_text())


def pack_root() -> Path:
    """Where the pack is: ``G2_EVAL_ASSETS_DIR``, else ``local_assets/G2-Eval`` of this checkout."""
    return Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT)


def digest(path: Path) -> str:
    """A file's SHA-256 digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify(root: Path) -> dict:
    """Check that ``root`` holds the locked pack, every file with its size and digest; returns its manifest."""
    manifest_path = root / "manifest.json"
    assert (
        manifest_path.is_file()
    ), f"no G2-Eval pack at {root}: run python -m isaaclab_arena.g2_eval.pack download, or set {ROOT_ENV}"
    expected = lock()["manifest_sha256"]
    assert digest(manifest_path) == expected, f"{root} is not the locked G2-Eval pack (manifest is not {expected})"
    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("pack") == PACK_SCHEMA, f"{root} is not a {PACK_SCHEMA} pack"
    for relative, record in manifest["files"].items():
        path = _inside(root, relative)
        assert path.is_file(), f"missing file in the G2-Eval pack: {relative}"
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], f"changed file: {relative}"
    return manifest


@functools.cache
def verified_root() -> Path:
    """The pack's root, verified once per process."""
    root = pack_root().resolve()
    verify(root)
    return root


@functools.cache
def load_task(name: str) -> dict:
    """A task of the locked pack: ``task`` (``task.json``), ``geometry`` (``geometry.json``'s objects), ``platform``
    (``common/platform.json``) and ``root`` (the task's directory); asset paths stay relative to their file."""
    assert name in lock()["tasks"], f"{name} is not a task of the locked G2-Eval pack ({lock()['tasks']})"
    root = verified_root()
    directory = root / "tasks" / name
    task = json.loads((directory / "task.json").read_text())
    assert task["schema"] == TASK_SCHEMA and task["name"] == name, f"{directory / 'task.json'} is not {name}"
    return {
        "task": task,
        "geometry": json.loads((directory / task["geometry"]).read_text())["objects"],
        "platform": json.loads((root / task["platform"]).read_text()),
        "root": directory,
        "pack_root": root,
    }


def _inside(root: Path, relative: str) -> Path:
    """A manifest path under ``root``, refusing one that leaves it."""
    path = root / relative
    assert not Path(relative).is_absolute() and path.resolve().is_relative_to(root.resolve()), relative
    return path


def download(directory: Path) -> None:
    """Download the locked revision into ``directory`` and verify every file."""
    from huggingface_hub import hf_hub_download, snapshot_download

    pinned = lock()
    assert pinned["repo_id"] and re.fullmatch(
        r"[0-9a-f]{40}", pinned["revision"] or ""
    ), "lock.json pins no published G2-Eval revision yet; point G2_EVAL_ASSETS_DIR at a local candidate"
    existing = directory / "manifest.json"
    if existing.exists():
        assert (
            digest(existing) == pinned["manifest_sha256"]
        ), f"{directory} holds another G2-Eval version; download into a new directory and set {ROOT_ENV}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(
        hf_hub_download(
            pinned["repo_id"], "manifest.json", repo_type="dataset", revision=pinned["revision"], local_dir=directory
        )
    )
    assert digest(manifest_path) == pinned["manifest_sha256"], "unexpected G2-Eval manifest"
    manifest = json.loads(manifest_path.read_text())
    for relative in manifest["files"]:
        _inside(directory, relative)
    snapshot_download(
        pinned["repo_id"],
        repo_type="dataset",
        revision=pinned["revision"],
        local_dir=directory,
        allow_patterns=[*manifest["files"], "manifest.json"],
    )
    verify(directory)
    print(f"G2-Eval {pinned['revision']} in {directory.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("download", "verify"))
    parser.add_argument("--directory", type=Path, help=f"default: {ROOT_ENV} or {DEFAULT_ROOT}")
    args = parser.parse_args()
    directory = args.directory or pack_root()
    if args.command == "download":
        download(directory)
    else:
        manifest = verify(directory)
        print(f"Verified {len(manifest['files'])} files; tasks {manifest['tasks']}")


if __name__ == "__main__":
    main()
