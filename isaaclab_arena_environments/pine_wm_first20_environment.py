# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asset-preserving composition for the first twenty pine_wm qualification tasks."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.assets.usdcraft_scene import bundle_root, resolve_asset
from isaaclab_arena.embodiments.ur7e.ur7e import UR7E_READY_JOINT_POS
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.pine_wm_environment import PineWmEnvironment, PineWmEnvironmentCfg

DEFAULT_PACKAGE = str(bundle_root())


@dataclass
class PineWmFirst20EnvironmentCfg(PineWmEnvironmentCfg):
    """Load one task with its exact supplied asset bindings."""

    task_id: str = "T001"
    package_root: str | None = None
    episode_length_s: float = 600.0
    review_asset_sizes_mm: dict | None = None


def task_instances(task):
    """Return instance names, asset IDs and fixture flags for a task."""
    ids = list(task["asset_bindings"])
    result = [
        (a.removeprefix("P20_"), a, a in {"P20_wall", "P20_v_support", "P20_coaster", "P20_push_mat"}) for a in ids
    ]
    tid = task["task_id"]
    if tid == "T016":
        result = [
            ("cube_40", "P20_cube_40", False),
            ("landmark_a", "P20_cylinder", True),
            ("landmark_b", "P20_cylinder", True),
        ]
    if tid == "T017":
        result = [("cube_40", "P20_cube_40", False), ("box_a", "P20_box", True), ("box_b", "P20_box", True)]
    if tid == "T142":
        result = [(a.removeprefix("P20_"), a, False) for a in ids if a != "P20_tray"]
        result += [(f"tray_{i}", "P20_tray", False) for i in range(3)]
    if tid == "T145":
        result = [(f"cube_{i}", "P20_cube_40", False) for i in range(6)] + [("box", "P20_box", False)]
    return result


@register_environment
class PineWmFirst20Environment(ArenaEnvironmentFactory[PineWmFirst20EnvironmentCfg]):
    """Construct task objects; the qualification runner owns reset sampling and scoring."""

    name = "pine_wm_first20"
    _legacy_argparse_cfg_type = PineWmFirst20EnvironmentCfg

    def build(self, cfg):
        package = bundle_root(os.environ.get("ARENA_PINE_WM_FIRST20_ROOT") or cfg.package_root or cfg.asset_root)
        is_bundle = (package / "manifest.json").is_file() and "entries" in json.loads(
            (package / "manifest.json").read_text()
        )
        if is_bundle:
            catalog = Path(__file__).resolve().parents[1] / "tools/pine_wm/first20/tasks.json"
        else:
            catalog = package / "configs/tasks.json"
        assert (
            catalog.is_file()
        ), f"Missing task catalog: {catalog}; download USDCraft-Scene and set ARENA_USDCRAFT_SCENE_ROOT"
        tasks = json.loads(catalog.read_text())["tasks"]
        task = next(t for t in tasks if t["task_id"] == cfg.task_id)
        arena = PineWmEnvironment().build(cfg)
        arena.name = self.name
        arena.first20_task = task
        arena.embodiment.set_joint_initial_pos(UR7E_READY_JOINT_POS)
        arena.first20_instances = task_instances(task)
        arena.first20_bounds = {}
        for i, (name, asset_id, fixed) in enumerate(arena.first20_instances):
            path = (
                resolve_asset(asset_id, package) if is_bundle else package / task["asset_bindings"][asset_id]["entry"]
            )
            stage = Usd.Stage.Open(str(path))
            bounds = (
                UsdGeom.BBoxCache(0, ["default", "render"])
                .ComputeWorldBound(stage.GetDefaultPrim())
                .ComputeAlignedRange()
            )
            scale = (1.0, 1.0, 1.0)
            requested_size = (cfg.review_asset_sizes_mm or {}).get(name)
            if requested_size is not None:
                assert asset_id in {"P20_box", "P20_tray"}, "Only review containers can be resized"
                extent = bounds.GetSize()
                scale = tuple(requested_size[axis] / 1000 / extent[axis] for axis in range(3))
            arena.first20_bounds[name] = (
                [bounds.GetMin()[axis] * scale[axis] for axis in range(3)],
                [bounds.GetMax()[axis] * scale[axis] for axis in range(3)],
            )
            articulated = asset_id in {"P20_drawer", "P20_button"}
            addons = {"collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)}
            if asset_id in {"P20_cylinder", "P20_sphere"} and not fixed:
                # Finite rolling resistance prevents ideal PhysX rollers drifting
                # off the slotted worktable during a long sorting episode.
                addons["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(linear_damping=0.1, angular_damping=2.0)
            if cfg.task_id == "T142" and name in {"cube_40", "cylinder", "sphere"}:
                # Equal appearance prevents the shape task being solved by fixed asset colours.
                addons["visual_material"] = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.32, 0.42, 0.55), roughness=0.6, metallic=0.0
                )
            if fixed:
                addons["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
            if articulated:
                addons["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)
            obj = Object(
                name=name,
                usd_path=str(path),
                scale=scale,
                object_type=ObjectType.ARTICULATION if articulated else ObjectType.RIGID,
                initial_pose=Pose(position_xyz=(-0.3 + i * 0.12, 0.1, 0.742 - float(bounds.GetMin()[2]))),
                spawn_cfg_addon=addons,
            )
            arena.scene.add_asset(obj)
        from isaaclab_arena_environments.pine_wm_first20_markers import add_markers

        add_markers(arena, cfg.task_id)
        return arena
