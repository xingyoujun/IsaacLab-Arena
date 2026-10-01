# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sample printed-bracket convex surfaces against cuMotion's tool collision spheres."""

import argparse
import importlib.util
import json
import numpy as np
import yaml
from pathlib import Path
from scipy.spatial import ConvexHull

from pxr import Usd, UsdGeom

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
robot = root / "isaaclab_arena/embodiments/ur7e"
spec = importlib.util.spec_from_file_location("mounts", robot / "pine_wm_mounts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
stage = Usd.Stage.CreateInMemory()
UsdGeom.Xform.Define(stage, "/Tool")
UsdGeom.Cylinder.Define(stage, "/Tool/GripperSpacer")
calibration = json.loads((robot / "calibration/pine_wm/wrist_cams.json").read_text())
module.build_camera_mounts(stage, "/Tool", calibration)
geometry = yaml.safe_load((robot / "rmpflow/pine_wm_ur7e.yaml").read_text())["collision_spheres"]
spheres = next(item["tool0"] for item in geometry if "tool0" in item)
centers = np.array([sphere["center"] for sphere in spheres])
radii = np.array([sphere["radius"] for sphere in spheres])
cache = UsdGeom.XformCache()
reports = []
for prim in Usd.PrimRange(stage.GetPrimAtPath("/Tool/PrintedCameraMount")):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
    T_T_M = np.array(cache.GetLocalToWorldTransform(prim)).T
    points = points @ T_T_M[:3, :3].T + T_T_M[:3, 3]
    hull = ConvexHull(points)
    samples = []
    for triangle in hull.simplices:
        a, b, c = points[triangle]
        for i in np.linspace(0, 1, 41):
            for j in np.linspace(0, 1 - i, 41):
                samples.append(a * i + b * j + c * (1 - i - j))
    samples = np.array(samples)
    gaps = (np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2) - radii).min(axis=1)
    reports.append({"part": prim.GetName(), "samples": len(samples), "max_uncovered_gap_m": float(gaps.max())})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(reports, indent=2) + "\n")
assert all(report["max_uncovered_gap_m"] <= 0 for report in reports), reports
print(args.output)
