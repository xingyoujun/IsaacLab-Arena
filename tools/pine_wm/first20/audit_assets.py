# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect the external package without changing asset geometry or physics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from isaaclab_arena.assets.usdcraft_scene import bundle_root  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, default=bundle_root())
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows = []
manifest = json.loads((args.package / "manifest.json").read_text())
if "entries" in manifest:
    items = [
        {"asset_id": key, "entry": value["path"]}
        for key, value in manifest["entries"].items()
        if key.startswith("P20_")
    ]
else:
    items = manifest["assets"]
for item in items:
    path = args.package / item["entry"]
    stage = Usd.Stage.Open(str(path))
    box = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    bodies = []
    joints = []
    colliders = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders += 1
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.PrismaticJoint(prim)
            joints.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "lower": j.GetLowerLimitAttr().Get(),
                    "upper": j.GetUpperLimitAttr().Get(),
                }
                if j
                else {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
            )
    rows.append({
        "asset_id": item["asset_id"],
        "entry": item["entry"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bounds_min_m": list(box.GetMin()),
        "bounds_max_m": list(box.GetMax()),
        "rigid_bodies": bodies,
        "joints": joints,
        "colliders": colliders,
        "runtime_validated": False,
    })
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(rows, indent=2) + "\n")
print(f"Audited {len(rows)} assets; static checks do not establish runtime success: {args.output}")
