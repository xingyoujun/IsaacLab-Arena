# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve stable asset IDs from a locally downloaded USDCraft-Scene release."""

import json
import os
from pathlib import Path

DEFAULT_BUNDLE_ROOT = Path(__file__).resolve().parents[2] / "local_assets/USDCraft-Scene"


def bundle_root(root: str | Path | None = None) -> Path:
    """Return the configured asset bundle root; never download implicitly during simulation."""
    return Path(root or os.environ.get("ARENA_USDCRAFT_SCENE_ROOT", DEFAULT_BUNDLE_ROOT)).expanduser()


def resolve_asset(asset_id: str, root: str | Path | None = None) -> Path:
    """Resolve an existing manifest entry within its bundle root."""
    directory = bundle_root(root)
    manifest_path = directory / "manifest.json"
    assert manifest_path.is_file(), (
        f"Missing USDCraft-Scene manifest: {manifest_path}. Run data_engine/assets/manage.py download "
        "DESTINATION and set ARENA_USDCRAFT_SCENE_ROOT=DESTINATION."
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert asset_id in manifest["entries"], f"Asset ID absent from release: {asset_id}"
    relative = manifest["entries"][asset_id]["path"]
    path = directory / relative
    assert not Path(relative).is_absolute() and path.resolve().is_relative_to(directory.resolve()), relative
    assert relative in manifest["files"] and path.is_file(), f"Missing asset payload: {path}"
    return path


def asset_or_legacy(asset_id: str, legacy: str | Path) -> str:
    """Prefer a bundled entry, retaining old paths when no explicit bundle was selected."""
    root = bundle_root()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if asset_id in manifest["entries"]:
            return str(resolve_asset(asset_id, root))
    assert not os.environ.get("ARENA_USDCRAFT_SCENE_ROOT"), f"Explicit bundle is missing {asset_id}: {root}"
    return str(legacy)
