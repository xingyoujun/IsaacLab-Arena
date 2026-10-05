# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Open every test_assets_v0 asset in Isaac Sim, render two views and record the result in the manifest.

This is a load-and-look check only: the USD composes on an Isaac Sim stage and renders. It does not
start physics, so it says nothing about articulation parsing, stability or joint behaviour.

    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
    export PYTHONPATH=/home/ubuntu/code/IsaacLab-Arena-tasks:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab
    .venv/bin/python tools/asset_baselines/render_assets.py [--only <asset_id> ...]
"""

import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--only", nargs="*", default=None)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--spp", type=int, default=48)
parser.add_argument("--force", action="store_true", help="re-render assets that already have renders")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.width, "height": args.height, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "outputs/asset_baselines/test_assets_v0"
VIEWS = {"a": (35.0, 25.0), "b": (125.0, 25.0), "c": (-145.0, 25.0), "d": (-55.0, 25.0)}


def look_at(eye, target):
    matrix = Gf.Matrix4d()
    matrix.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    return matrix.GetInverse()


def render(entry):
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    asset = UsdGeom.Xform.Define(stage, "/World/asset")
    # Render what the data engine would ingest: the recorded rewrite copy when one exists.
    asset.GetPrim().GetReferences().AddReference((entry.get("ingest") or {}).get("usd") or entry["usd"])
    scale = float(entry["static"]["meters_per_unit"] or 1.0)
    if abs(scale - 1.0) > 1e-9:
        asset.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    app.update()
    prim = stage.GetPrimAtPath("/World/asset")
    children = list(prim.GetChildren()) if prim and prim.IsValid() else []
    rng = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )
    if not children or rng.IsEmpty():
        return dict(isaac_load="empty_stage", renders=[])
    center = np.array(rng.GetMidpoint())
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
    product = rep.create.render_product("/World/camera", (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    (ROOT / "renders").mkdir(parents=True, exist_ok=True)

    def capture(suffix=""):
        images, blank = [], 0
        for view, (azimuth, elevation) in VIEWS.items():
            az, el = math.radians(azimuth), math.radians(elevation)
            eye = center + radius * np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
            op.Set(look_at(eye, center))
            rep.orchestrator.step(rt_subframes=args.spp, delta_time=0.0)
            pixels = np.asarray(annotator.get_data())[..., :3]
            blank += int(pixels.std() < 2.0)
            relative = f"renders/{entry['asset_id'].replace('/', '__')}_{view}{suffix}.png"
            Image.fromarray(pixels).save(ROOT / relative)
            images.append(relative)
        return images, blank == len(VIEWS)

    renders, blank = capture()
    note = None
    if blank:
        # The authored materials rendered nothing; bind one plain material above the asset to tell an
        # invisible-material problem apart from missing geometry. The asset file is not modified.
        material = UsdShade.Material.Define(stage, "/World/Looks/review_gray")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/review_gray/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.6, 0.62))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        app.update()
        renders, still_blank = capture("_gray")
        note = "geometry_not_visible" if still_blank else "authored_materials_not_rendered_in_isaac_rtx"
    annotator.detach()
    product.destroy()
    return dict(isaac_load="ok", renders=renders, render_note=note)


manifest_path = ROOT / "manifest.json"
manifest = json.loads(manifest_path.read_text())
version = None
try:
    from isaacsim.core.version import get_version

    version = ".".join(str(v) for v in get_version()[:3])
except Exception:  # noqa: BLE001 - version string is informational only
    pass
for entry in manifest["entries"]:
    if args.only and entry["asset_id"] not in args.only:
        continue
    if not args.force and (entry.get("runtime") or {}).get("renders"):
        continue
    try:
        result = render(entry)
    except Exception as error:  # noqa: BLE001 - one bad asset must not hide the rest
        result = dict(isaac_load=f"error: {type(error).__name__}: {error}"[:200], renders=[])
    result.update(isaac_check="usd_compose_and_render_only", isaac_version=version)
    entry["runtime"] = result
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
    print("[asset]", entry["asset_id"], result["isaac_load"], flush=True)

app.close()
