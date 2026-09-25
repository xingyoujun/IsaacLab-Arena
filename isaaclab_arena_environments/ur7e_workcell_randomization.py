# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visual domain randomization of the UR7e workcell for camera re-rendering.

Everything the calibrated D435 sees except the robot and the task object is randomized per
demonstration with stock Isaac Sim content: the three workcell lights (intensity, colour, direction
and, for the dome, one of Isaac Sim's HDRI skies), the materials of the slotted table, its legs and
the ground plane (either the original preview surfaces with jittered parameters or one of the
``NVIDIA/Materials/Base`` MDL materials shipped with Isaac Sim), and a handful of small collision-free
distractor primitives scattered on the table away from the task object. Physics is untouched, so
recorded states replay unchanged.

Usage::

    randomizer = WorkcellVisualRandomizer(stage, seed=0, skies_dir="/path/to/skies")
    randomizer.randomize(demo_index, keep_clear=[(x, y, radius), ...])
"""

from __future__ import annotations

import math
import numpy as np
import pathlib

import isaaclab.sim as sim_utils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from isaaclab_arena_environments.ur7e_workcell_environment import (
    DOME_INTENSITY,
    FILL_INTENSITY,
    KEY_INTENSITY,
    LEG_COLOR,
    PROFILE_COLOR,
    ROBOT_BASE_XY,
    SLOT_FLOOR_COLOR,
    TABLE_SIZE_M,
    TABLE_TOP_HEIGHT_M,
)

try:
    from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
except ImportError:  # pragma: no cover - older Isaac Lab
    NVIDIA_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/NVIDIA"

# Isaac Sim's MDL library, grouped by where the material is plausible on the workcell.
TABLE_MDLS = [
    "Metals/Aluminum_Anodized.mdl",
    "Metals/Aluminum_Anodized_Charcoal.mdl",
    "Metals/Aluminum_Cast.mdl",
    "Metals/Aluminum_Polished.mdl",
    "Metals/Brushed_Antique_Copper.mdl",
    "Metals/Steel_Stainless.mdl",
    "Metals/Steel_Carbon.mdl",
    "Metals/Iron.mdl",
    "Wood/Ash.mdl",
    "Wood/Birch.mdl",
    "Wood/Oak.mdl",
    "Wood/Plywood.mdl",
    "Wood/Bamboo.mdl",
    "Wood/Walnut.mdl",
    "Plastics/Plastic.mdl",
    "Plastics/Plastic_ABS.mdl",
    "Plastics/Rubber_Textured.mdl",
    "Plastics/Vinyl.mdl",
    "Stone/Marble_Smooth.mdl",
    "Stone/Granite_Light.mdl",
    "Stone/Granite_Dark.mdl",
    "Stone/Ceramic_Smooth_Fired.mdl",
    "Stone/Porcelain_Smooth.mdl",
    "Miscellaneous/Paint_Matte.mdl",
    "Miscellaneous/Paint_Satin.mdl",
]
"""Materials that may replace the table surface (profiles and slot floor)."""
FLOOR_MDLS = [
    "Stone/Ceramic_Tile_12.mdl",
    "Stone/Terrazzo.mdl",
    "Stone/Slate.mdl",
    "Stone/Granite_Dark.mdl",
    "Wood/Oak_Planks.mdl",
    "Wood/Parquet_Floor.mdl",
    "Textiles/Cloth_Gray.mdl",
    "Plastics/Rubber_Textured.mdl",
    "Miscellaneous/Paint_Matte.mdl",
    "Metals/CorrugatedMetal.mdl",
]
"""Materials for the ground plane (visible beyond the table edge)."""

TABLE_MATERIAL_MDL_PROBABILITY = 0.5
"""Chance the table gets an MDL material instead of the jittered original preview surface."""
DOME_TEXTURE_PROBABILITY = 0.6
"""Chance the dome gets an HDRI sky instead of a uniform colour."""
DISTRACTOR_MAX_COUNT = 6
DISTRACTOR_SIZE_M = (0.015, 0.06)
"""Edge length / diameter range of the distractor primitives."""
ROBOT_KEEP_CLEAR_M = 0.20
"""No distractor within this radius of the robot base."""


def _light_scale(rng: np.random.Generator, low: float, high: float) -> float:
    """Log-uniform multiplier so halving and doubling are equally likely."""
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _light_color(rng: np.random.Generator) -> Gf.Vec3f:
    """A near-white light colour from warm (fluorescent lab) to cool (daylight)."""
    warmth = rng.uniform(-1.0, 1.0)  # <0 cool, >0 warm
    r = 1.0 - 0.10 * max(0.0, -warmth)
    g = 1.0 - 0.04 * abs(warmth)
    b = 1.0 - 0.22 * max(0.0, warmth)
    jitter = rng.uniform(-0.03, 0.03, size=3)
    return Gf.Vec3f(*(np.clip(np.array([r, g, b]) + jitter, 0.6, 1.0).tolist()))


def _quat_from_direction(elevation_deg: float, azimuth_deg: float) -> Gf.Quatf:
    """Orientation whose local -Z (the distant light's emission axis) points down at the given angles."""
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    # Emission direction (unit vector pointing from the light towards the scene).
    d = Gf.Vec3d(-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))
    rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), d)
    q = rot.GetQuat()
    return Gf.Quatf(q.GetReal(), Gf.Vec3f(*q.GetImaginary()))


def _xform_ops(prim: Usd.Prim) -> dict:
    """Return the prim's translate/orient/scale xform ops, creating the missing ones."""
    xformable = UsdGeom.Xformable(prim)
    ops = {op.GetOpType(): op for op in xformable.GetOrderedXformOps()}
    if UsdGeom.XformOp.TypeTranslate not in ops:
        ops[UsdGeom.XformOp.TypeTranslate] = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if UsdGeom.XformOp.TypeOrient not in ops:
        ops[UsdGeom.XformOp.TypeOrient] = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    if UsdGeom.XformOp.TypeScale not in ops:
        ops[UsdGeom.XformOp.TypeScale] = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    xformable.SetXformOpOrder(
        [ops[UsdGeom.XformOp.TypeTranslate], ops[UsdGeom.XformOp.TypeOrient], ops[UsdGeom.XformOp.TypeScale]]
    )
    return ops


def _set_orient(op: UsdGeom.XformOp, q: Gf.Quatf) -> None:
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Quatd(q.GetReal(), Gf.Vec3d(*q.GetImaginary())))
    else:
        op.Set(q)


def _set_vec3(op: UsdGeom.XformOp, v) -> None:
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Vec3d(*v))
    else:
        op.Set(Gf.Vec3f(*v))


def _yaw_quat(yaw_rad: float) -> Gf.Quatf:
    return Gf.Quatf(math.cos(yaw_rad / 2), Gf.Vec3f(0.0, 0.0, math.sin(yaw_rad / 2)))


class WorkcellVisualRandomizer:
    """Per-demonstration visual randomization of the UR7e workcell stage.

    Args:
        stage: The USD stage of the built environment.
        seed: Base seed; ``randomize(index)`` derives an independent stream per index, so a demo renders
            identically on every worker and relaunch.
        skies_dir: Directory of ``.hdr`` HDRI files for the dome light; ``None`` disables sky textures.
        env_ns: Prim path of the environment instance that holds the table.
        max_distractors: Upper bound on the number of distractor primitives per demo.
    """

    def __init__(
        self,
        stage: Usd.Stage,
        seed: int,
        skies_dir: str | pathlib.Path | None = None,
        env_ns: str = "/World/envs/env_0",
        max_distractors: int = DISTRACTOR_MAX_COUNT,
    ):
        self.stage = stage
        self.seed = seed
        self.max_distractors = max_distractors
        self.skies = sorted(str(p) for p in pathlib.Path(skies_dir).glob("*.hdr")) if skies_dir else []

        # Lights.
        self.dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Light"))
        assert self.dome, "dome light /World/Light not found"
        self.key = UsdLux.DistantLight(stage.GetPrimAtPath("/World/KeyLight"))
        self.fill = UsdLux.DistantLight(stage.GetPrimAtPath("/World/FillLight"))
        assert self.key and self.fill, "distant lights /World/KeyLight and /World/FillLight not found"

        # Table meshes, grouped by role. Each Arena cuboid is <name>/geometry/mesh with <name>/geometry/material.
        table = stage.GetPrimAtPath(f"{env_ns}/Table")
        assert table.IsValid(), f"table prim {env_ns}/Table not found"
        self.profiles, self.slot_floor, self.legs = [], [], []
        for block in table.GetChildren():
            mesh = stage.GetPrimAtPath(f"{block.GetPath()}/geometry/mesh")
            assert mesh.IsValid(), f"unexpected table block layout at {block.GetPath()}"
            name = block.GetName()
            target = self.profiles if name.startswith("Profile") else self.legs if name.startswith("Leg") else None
            if name == "SlotFloor":
                target = self.slot_floor
            assert target is not None, f"unknown table block {name}"
            target.append(mesh)
        self.original_material = {
            mesh.GetPath(): UsdShade.MaterialBindingAPI(mesh).GetDirectBinding().GetMaterialPath()
            for mesh in self.profiles + self.slot_floor + self.legs
        }
        self.ground = stage.GetPrimAtPath("/World/GroundPlane")
        assert self.ground.IsValid(), "ground plane /World/GroundPlane not found"

        # MDL materials are created once (the first use downloads them) and only re-bound per demo.
        self.table_materials = [self._mdl_material("Table", m) for m in TABLE_MDLS]
        self.floor_materials = [self._mdl_material("Floor", m) for m in FLOOR_MDLS]

        # Distractor primitives: created once, re-posed/re-coloured/hidden per demo.
        self.distractors = []
        for i in range(max_distractors):
            path = f"/World/Distractors/d{i}"
            shape = ("cube", "cylinder", "sphere")[i % 3]
            material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.0, roughness=0.5)
            if shape == "cube":
                cfg = sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=material)
            elif shape == "cylinder":
                cfg = sim_utils.CylinderCfg(radius=0.5, height=1.0, visual_material=material)
            else:
                cfg = sim_utils.SphereCfg(radius=0.5, visual_material=material)
            prim = cfg.func(path, cfg, translation=(0.0, 0.0, -1.0))
            self.distractors.append(prim)
            UsdGeom.Imageable(prim).MakeInvisible()

    # ------------------------------------------------------------------------------ helpers ---
    def _mdl_material(self, group: str, relative_mdl: str) -> Sdf.Path:
        stem = pathlib.Path(relative_mdl).stem
        path = f"/World/Looks/DR_{group}_{stem}"
        if not self.stage.GetPrimAtPath(path).IsValid():
            cfg = sim_utils.MdlFileCfg(mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/{relative_mdl}", project_uvw=True)
            cfg.func(path, cfg)
        return Sdf.Path(path)

    @staticmethod
    def _shader(material_path: Sdf.Path, stage: Usd.Stage) -> UsdShade.Shader:
        shader = UsdShade.Shader(stage.GetPrimAtPath(material_path.AppendChild("Shader")))
        assert shader, f"no Shader under {material_path}"
        return shader

    def _bind(self, meshes: list[Usd.Prim], material_path: Sdf.Path) -> None:
        for mesh in meshes:
            sim_utils.bind_visual_material(mesh.GetPath(), material_path, stage=self.stage)

    def _preview_surface(self, meshes: list[Usd.Prim], color, metallic: float, roughness: float) -> None:
        """Rebind the original preview surface of each mesh and set its parameters."""
        for mesh in meshes:
            material_path = self.original_material[mesh.GetPath()]
            sim_utils.bind_visual_material(mesh.GetPath(), material_path, stage=self.stage)
            shader = self._shader(material_path, self.stage)
            shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color))
            shader.GetInput("metallic").Set(float(metallic))
            shader.GetInput("roughness").Set(float(roughness))

    @staticmethod
    def _jitter_color(rng: np.random.Generator, base, brightness=(0.45, 1.2), hue=0.08) -> tuple[float, ...]:
        scale = rng.uniform(*brightness)
        color = np.array(base) * scale + rng.uniform(-hue, hue, size=3)
        return tuple(np.clip(color, 0.02, 1.0).tolist())

    # -------------------------------------------------------------------------------- public ---
    def randomize(self, index: int, keep_clear: list[tuple[float, float, float]] = ()) -> dict:
        """Randomize lights, materials and distractors for demo ``index``; returns a summary dict.

        Args:
            index: Demo index; the same index always yields the same randomization for a given seed.
            keep_clear: ``(x, y, radius)`` discs in the world frame that distractors must avoid,
                typically the task object's footprint. The robot base is always avoided.
        """
        rng = np.random.default_rng([self.seed, index])
        summary = {}

        # --- lights
        dome_texture = ""
        if self.skies and rng.random() < DOME_TEXTURE_PROBABILITY:
            dome_texture = str(rng.choice(self.skies))
        self.dome.CreateTextureFileAttr().Set(Sdf.AssetPath(dome_texture))
        self.dome.CreateIntensityAttr().Set(DOME_INTENSITY * _light_scale(rng, 0.4, 1.6))
        self.dome.CreateColorAttr().Set(_light_color(rng) if not dome_texture else Gf.Vec3f(1.0, 1.0, 1.0))
        dome_ops = _xform_ops(self.dome.GetPrim())
        _set_orient(dome_ops[UsdGeom.XformOp.TypeOrient], _yaw_quat(rng.uniform(0.0, 2 * math.pi)))
        summary["dome"] = pathlib.Path(dome_texture).stem if dome_texture else "uniform"

        for name, light, base, low, high, el_range in (
            ("key", self.key, KEY_INTENSITY, 0.2, 1.8, (30.0, 80.0)),
            ("fill", self.fill, FILL_INTENSITY, 0.0, 1.6, (15.0, 70.0)),
        ):
            light.CreateIntensityAttr().Set(
                base * (_light_scale(rng, max(low, 0.05), high) if rng.random() > 0.1 else 0.0)
            )
            light.CreateColorAttr().Set(_light_color(rng))
            ops = _xform_ops(light.GetPrim())
            _set_orient(
                ops[UsdGeom.XformOp.TypeOrient], _quat_from_direction(rng.uniform(*el_range), rng.uniform(0.0, 360.0))
            )
            summary[name] = round(float(light.GetIntensityAttr().Get()), 1)

        # --- table
        if rng.random() < TABLE_MATERIAL_MDL_PROBABILITY:
            material = self.table_materials[rng.integers(len(self.table_materials))]
            self._bind(self.profiles, material)
            # The groove floor is either the same material (a solid slab) or a darker plain surface.
            if rng.random() < 0.5:
                self._bind(self.slot_floor, material)
            else:
                self._preview_surface(
                    self.slot_floor,
                    self._jitter_color(rng, SLOT_FLOOR_COLOR),
                    rng.uniform(0.0, 0.6),
                    rng.uniform(0.3, 0.9),
                )
            summary["table"] = material.name
        else:
            self._preview_surface(
                self.profiles, self._jitter_color(rng, PROFILE_COLOR), rng.uniform(0.1, 0.9), rng.uniform(0.15, 0.8)
            )
            self._preview_surface(
                self.slot_floor, self._jitter_color(rng, SLOT_FLOOR_COLOR), rng.uniform(0.0, 0.6), rng.uniform(0.3, 0.9)
            )
            summary["table"] = "preview_surface"
        self._preview_surface(
            self.legs, self._jitter_color(rng, LEG_COLOR), rng.uniform(0.0, 0.8), rng.uniform(0.2, 0.8)
        )

        # --- ground plane
        material = self.floor_materials[rng.integers(len(self.floor_materials))]
        sim_utils.bind_visual_material(self.ground.GetPath(), material, stage=self.stage)
        summary["floor"] = material.name

        # --- distractors
        count = int(rng.integers(0, self.max_distractors + 1))
        clear = [(*ROBOT_BASE_XY, ROBOT_KEEP_CLEAR_M), *keep_clear]
        placed = 0
        for prim in self.distractors:
            imageable = UsdGeom.Imageable(prim)
            if placed >= count:
                imageable.MakeInvisible()
                continue
            position = None
            for _ in range(50):
                candidate = rng.uniform(-TABLE_SIZE_M / 2 + 0.05, TABLE_SIZE_M / 2 - 0.05, size=2)
                if all(math.hypot(candidate[0] - cx, candidate[1] - cy) > r for cx, cy, r in clear):
                    position = candidate
                    break
            if position is None:
                imageable.MakeInvisible()
                continue
            size = rng.uniform(*DISTRACTOR_SIZE_M, size=3)
            ops = _xform_ops(prim)
            _set_vec3(ops[UsdGeom.XformOp.TypeTranslate], (position[0], position[1], TABLE_TOP_HEIGHT_M + size[2] / 2))
            _set_orient(ops[UsdGeom.XformOp.TypeOrient], _yaw_quat(rng.uniform(0.0, 2 * math.pi)))
            _set_vec3(ops[UsdGeom.XformOp.TypeScale], size)
            shader = self._shader(Sdf.Path(f"{prim.GetPath()}/geometry/material"), self.stage)
            shader.GetInput("diffuseColor").Set(Gf.Vec3f(*rng.uniform(0.05, 1.0, size=3).tolist()))
            shader.GetInput("metallic").Set(float(rng.uniform(0.0, 0.9)))
            shader.GetInput("roughness").Set(float(rng.uniform(0.2, 0.9)))
            imageable.MakeVisible()
            placed += 1
        summary["distractors"] = placed
        return summary
