# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Atomic run metadata and cross-process leases, independent of the simulator."""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def digest(path):
    """Hash file bytes without retaining a large recording in memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    """Identify a finite JSON configuration independently of formatting."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
    """Atomically replace metadata after flushing bytes; callers own the run lease."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inside(root, relative):
    """Resolve only files inside a run/bundle, including through symlinks."""
    root = Path(root).resolve()
    path = root / relative
    assert not Path(relative).is_absolute() and path.resolve().is_relative_to(root), relative
    return path


@contextmanager
def lease(path):
    """Hold a nonblocking advisory lock; process exit releases it without stale PID cleanup."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Resource is busy: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def snapshot(root):
    """Hash immutable child outputs, excluding transient locks and Python caches."""
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".lock":
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def verify_snapshot(root, expected):
    """Reject changed, removed or additional artifacts before deriving another product."""
    actual = snapshot(root)
    assert actual == expected, "Sealed recording artifacts changed; do not reuse derived data"
