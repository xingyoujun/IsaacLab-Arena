# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve G2 asset and planner inputs without importing the simulator."""

from pathlib import Path

from isaaclab_arena.assets.usdcraft_scene import resolve_asset


def planning_path(asset_id: str, explicit: str | Path | None = None) -> Path:
    """Resolve an explicit planner path or a required HF manifest ID."""
    path = Path(explicit or resolve_asset(asset_id)).expanduser()
    assert path.is_file(), f"Missing G2 planner asset {asset_id}: {path}"
    return path


def transfer_urdf(config: dict) -> Path | None:
    """Resolve the optional transfer model from its HF manifest ID."""
    if config.get("transfer_robot_asset"):
        return planning_path(config["transfer_robot_asset"])
    assert not config.get("transfer_robot_urdf"), "Use transfer_robot_asset with a manifest ID"
    return None
