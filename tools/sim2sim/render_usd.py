# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render a few USD assets from four views in Isaac Sim (review images only; no physics).

.venv/bin/python tools/sim2sim/render_usd.py --out DIR NAME=/abs/asset.usd[z] [NAME=...]
"""

import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--spp", type=int, default=48)
parser.add_argument("items", nargs="+")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 640, "height": 480, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

VIEWS = {"a": (35.0, 25.0), "b": (125.0, 25.0), "c": (-145.0, 25.0), "d": (-55.0, 25.0)}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
report = {}


def look_at(eye, target):
    matrix = Gf.Matrix4d()
    matrix.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    return matrix.GetInverse()


for item in args.items:
    name, path = item.split("=", 1)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset = UsdGeom.Xform.Define(stage, "/World/asset")
    asset.GetPrim().GetReferences().AddReference(path)
    source = Usd.Stage.Open(path)
    mpu = UsdGeom.GetStageMetersPerUnit(source)
    if abs(mpu - 1.0) > 1e-9:
        asset.AddScaleOp().Set(Gf.Vec3f(mpu, mpu, mpu))
    if UsdGeom.GetStageUpAxis(source) == UsdGeom.Tokens.y:
        asset.AddRotateXOp().Set(90.0)
    app.update()
    prim = stage.GetPrimAtPath("/World/asset")
    rng = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )
    if rng.IsEmpty():
        report[name] = dict(path=path, renders=[], note="empty bounds")
        continue
    center = np.array(rng.GetMidpoint())
    size = [round(float(v), 4) for v in rng.GetSize()]
    radius = max(float(np.linalg.norm(np.array(rng.GetSize()))) * 1.5, 0.3)
    UsdLux.DomeLight.Define(stage, "/World/lights/dome").CreateIntensityAttr(700.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/lights/sun")
    sun.CreateIntensityAttr(1800.0)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))
    camera = UsdGeom.Camera.Define(stage, "/World/camera")
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    op = UsdGeom.Xformable(camera.GetPrim()).AddTransformOp()
    product = rep.create.render_product("/World/camera", (640, 480))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    renders = []
    for view, (azimuth, elevation) in VIEWS.items():
        az, el = math.radians(azimuth), math.radians(elevation)
        eye = center + radius * np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
        op.Set(look_at(eye, center))
        rep.orchestrator.step(rt_subframes=args.spp, delta_time=0.0)
        file = f"{name}_{view}.png"
        Image.fromarray(np.asarray(annotator.get_data())[..., :3]).save(out / file)
        renders.append(file)
    annotator.detach()
    product.destroy()
    report[name] = dict(path=path, renders=renders, size_m=size)
    print("[render]", name, size, flush=True)

(out / "renders.json").write_text(json.dumps(report, indent=1))
app.close()
