# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Extend a verified Pine WM bundle locally with portable G2 dependencies; never publish."""

import argparse
import copy
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from data_engine.assets.manage import ROOT, digest, verify, write_json


def collect_usd(source, destination):
    """Copy a USD dependency closure and rewrite references relative to its new layers."""
    from pxr import Sdf, UsdUtils

    source = Path(source).resolve()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source))
    assert set(unresolved) <= {"OmniPBR.mdl"}, (source, unresolved)
    sources = {Path(layer.realPath).resolve() for layer in layers if layer.realPath}
    sources.update(Path(asset).resolve() for asset in assets)
    targets = {}
    for path in sorted(sources):
        assert path.is_file(), path
        targets[path] = destination.parent / "dependencies" / f"{digest(path)[:16]}_{path.name}"
    targets[source] = destination
    for path, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for layer in layers:
        old = Path(layer.realPath).resolve()
        new = targets[old]
        copied = Sdf.Layer.FindOrOpen(str(new))

        def relocate(value):
            if not value or value == "OmniPBR.mdl":
                return value
            resolved = Path(Sdf.ComputeAssetPathRelativeToLayer(layer, value)).resolve()
            assert resolved in targets, (old, value, resolved)
            return os.path.relpath(targets[resolved], new.parent)

        UsdUtils.ModifyAssetPaths(copied, relocate)
        copied.Save()
    return {"source_name": source.name, "source_sha256": digest(source)}


def planning_urdf(source, target):
    """Export unchanged planning joints without unused vendor visual/collision mesh URIs."""
    tree = ET.parse(source)
    # Both supported planners use the recorded sphere model, never these mesh tags.
    # The full visual/collision robot remains in the USD entry; retain the source hash.
    for link in tree.getroot().findall("link"):
        for element in list(link):
            if element.tag in {"visual", "collision"}:
                link.remove(element)
    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=True)


def prepare(args):
    """Create a new additive bundle; refuse to overwrite either input or destination."""
    import yaml

    original = verify(args.base)
    assert not args.output.exists(), f"Destination already exists: {args.output}"
    shutil.copytree(args.base, args.output, ignore=shutil.ignore_patterns(".cache"))
    manifest = copy.deepcopy(original)
    entries = manifest["entries"]
    source = args.geniesim
    arena = args.arena_assets
    models = {
        "g2": (source / "robot/G2_omnipicker/robot_fix.usda", "embodiments/g2/robot/model.usda"),
        "g2_table": (args.wrappers / "benchmark_table_000_collision.usda", "assets/g2_table/model.usda"),
        "g2_bowl": (args.wrappers / "benchmark_bowl_025_rigid_root.usda", "assets/g2_bowl/model.usda"),
        "g2_peg": (arena / "Factory/factory_peg_8mm.usd", "assets/g2_peg/model.usd"),
        "g2_hole": (arena / "Factory/factory_hole_8mm.usd", "assets/g2_hole/model.usd"),
        "g2_small_gear": (arena / "Factory/factory_gear_small.usd", "assets/g2_small_gear/model.usd"),
        "g2_room": (args.room, "assets/g2_room/model.usda"),
        "g2_aluminum_stock": (
            args.aluminum_stock,
            "assets/g2_aluminum_stock/model.usda",
        ),
    }
    objects = arena / "Arena/assets/object_library/srl_robolab_assets/objects"
    models["g2_cordless_drill_ycb_robolab"] = (objects / "ycb/cordless_drill.usd", "assets/g2_drill/model.usd")
    models["g2_bin_b04_vomp_robolab"] = (objects / "vomp/bin_b04/bin_b04.usd", "assets/g2_bin/model.usd")
    for asset_id, (path, relative) in models.items():
        assert asset_id not in entries, f"Existing entry must not be replaced: {asset_id}"
        provenance = collect_usd(path, args.output / relative)
        entries[asset_id] = {"path": relative, "kind": "embodiment" if asset_id == "g2" else "object", **provenance}
    planner = args.output / "embodiments/g2/planning"
    for asset_id, path in {
        "g2_tcp_urdf": args.planning / "g2_tcp_aligned.urdf",
        "g2_margin_urdf": args.planning / "g2_joint_margin.urdf",
        "g2_original_urdf": args.robot_yaml.with_name("robot.urdf"),
    }.items():
        target = planner / f"{asset_id}.urdf"
        planning_urdf(path, target)
        entries[asset_id] = {
            "path": target.relative_to(args.output).as_posix(),
            "kind": "planning",
            "source_sha256": digest(path),
            "adaptation": (
                "Remove unused visual/collision mesh tags; unchanged joints and inertials; use vendor spheres"
            ),
        }
    config = yaml.safe_load(args.robot_yaml.read_text())
    kin = config["robot_cfg"]["kinematics"]
    kin.update(use_usd_kinematics=False, urdf_path="g2_tcp_urdf.urdf", asset_root_path=".")
    for key in ("isaac_usd_path", "usd_path", "usd_robot_root"):
        kin.pop(key, None)
    (planner / "robot.yaml").write_text(yaml.safe_dump(config))
    entries["g2_robot_yaml"] = {
        "path": "embodiments/g2/planning/robot.yaml",
        "kind": "planning",
        "source_sha256": digest(args.robot_yaml),
    }
    scene_source = ROOT / "data_engine/g2/clean_workcell_table.yaml"
    scene = {
        "schema_version": 1,
        "environment": "g2_clean_workcell_table",
        "embodiment": "g2",
        "implementation": "isaaclab_arena_environments/g2_clean_workcell_environment.py",
        "configuration_source": scene_source.relative_to(ROOT).as_posix(),
        "configuration_sha256": digest(scene_source),
        "configuration": yaml.safe_load(scene_source.read_text()),
        "cameras": {"head_camera": [640, 400], "left_wrist_camera": [640, 528], "right_wrist_camera": [640, 528]},
        "calibration_source": "Authored camera prims in the pinned G2 USD; runtime G2CameraCfg selects resolution",
        "table_top_z_m": 0.0,
        "robot_position_m": [-0.65, 0.0, -0.75],
        "asset_ids": list(models),
    }
    scene_path = "scenes/g2_clean_workcell/scene.json"
    write_json(args.output / scene_path, scene)
    entries["g2_clean_workcell"] = {"path": scene_path, "kind": "scene"}
    manifest["local_extension"] = {
        "published": False,
        "base_manifest_sha256": digest(args.base / "manifest.json"),
        "description": (
            "G2 local integration. Preserve GenieSim/Arena upstream provenance and licenses; no redistribution grant."
        ),
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path != args.output / "manifest.json":
            name = path.relative_to(args.output).as_posix()
            record = {"sha256": digest(path), "bytes": path.stat().st_size}
            if name in original["files"]:
                assert record == original["files"][name], f"Base asset changed: {name}"
            manifest["files"][name] = record
    write_json(args.output / "manifest.json", manifest)
    verify(args.output)
    print(f"Local only. Set ARENA_USDCRAFT_SCENE_ROOT={args.output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base",
        "output",
        "geniesim",
        "arena-assets",
        "wrappers",
        "planning",
        "robot-yaml",
        "room",
        "aluminum-stock",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    prepare(parser.parse_args())
