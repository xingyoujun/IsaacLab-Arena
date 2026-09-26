# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Genie Sim scene assets (tables, benchmark objects) registered for use in Arena scenes.

Every Genie Sim asset ships as an ``Aligned.usda`` wrapper that re-orients the Y-up mesh payload to Z-up, so
these classes point at the wrapper rather than the raw ``Aligned.usd``. Objects carry their own rigid-body,
collision and physics-material setup; furniture ships without collision, which is added here at spawn time.
"""

import os
import tempfile

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.geniesim import GENIESIM_ASSETS_DIR
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.usdcraft_scene import asset_or_legacy
from isaaclab_arena.utils.pose import Pose

GENIE_TABLE_HEIGHT_M: float = 0.74
"""Height of ``benchmark_table_000`` (0.6 m deep, 1.2 m wide); its origin sits at mid-height."""

GENIE_TABLE_SOURCE_USD: str = f"{GENIESIM_ASSETS_DIR}/background/common/table/benchmark_table_000/Aligned.usda"
"""Vendor table layer (visuals only)."""


def with_static_mesh_collision(source_usd: str, output_basename: str) -> str:
    """Return a cached USD that references ``source_usd`` and adds triangle-mesh colliders to every mesh.

    Genie Sim furniture ships without collision. Isaac Lab's ``collision_props`` only modifies prims that already
    carry ``CollisionAPI``, so the schema is authored here as ``over`` opinions in a thin wrapper layer written to
    ``~/.cache/isaaclab_arena/usd/geniesim/`` (once per basename).

    Args:
        source_usd: The vendor ``Aligned.usda`` to wrap.
        output_basename: File name (without extension) of the wrapper in the cache directory.

    Returns:
        Path of the wrapper USD.
    """
    # Import locally because USD/pxr is available only after simulation initialization.
    from pxr import Usd, UsdGeom, UsdPhysics

    cache_root = get_arena_asset_cache_dir().parent / "usd" / "geniesim"
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = cache_root / f"{output_basename}.usda"
    if out_path.exists():
        return str(out_path)

    with tempfile.NamedTemporaryFile(suffix=".usda", dir=cache_root, delete=False) as tmp_file:
        tmp_path = tmp_file.name
    stage = Usd.Stage.CreateNew(tmp_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim("/World", "Xform")
    root.GetReferences().AddReference(source_usd, "/World")
    stage.SetDefaultPrim(root)
    num_meshes = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
            num_meshes += 1
    assert num_meshes > 0, f"no meshes found in {source_usd}"
    stage.GetRootLayer().Save()
    os.replace(tmp_path, out_path)
    return str(out_path)


@register_asset
class GenieBenchmarkTable(LibraryBackground):
    """Genie Sim's ``benchmark_table_000`` dining table, placed so its top surface is at z = 0.

    This is the table of the ``table_task_*_g2_op`` benchmark scenes (e.g. ``stack_bowls``). The floor is at
    z = -height_m (default -0.74); place the robot chassis there.
    """

    name = "genie_benchmark_table"
    tags = ["background", "geniesim"]
    usd_path = None  # resolved lazily: the vendor layer wrapped with static mesh colliders
    initial_pose = Pose(position_xyz=(0.0, 0.0, -GENIE_TABLE_HEIGHT_M / 2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    object_min_z = -0.5

    def __init__(self, height_m: float = GENIE_TABLE_HEIGHT_M):
        assert height_m > 0.0, "Table height must be positive"
        source = asset_or_legacy("g2_table", GENIE_TABLE_SOURCE_USD)
        self.usd_path = (
            source
            if source != GENIE_TABLE_SOURCE_USD
            else with_static_mesh_collision(source, "benchmark_table_000_collision")
        )
        self.initial_pose = Pose(position_xyz=(0.0, 0.0, -height_m / 2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        super().__init__(scale=(1.0, 1.0, height_m / GENIE_TABLE_HEIGHT_M))


def with_rigid_root_above_entity(source_usd: str, output_basename: str) -> str:
    """Return a cached USD whose rigid body is a new root prim wrapping the vendor ``/World/entity``.

    Genie Sim object wrappers author the Y-up-to-Z-up rotation on the same prim that carries ``RigidBodyAPI``.
    Isaac Lab writes an object's pose to its rigid-body prim, which would discard that rotation and lay the object
    on its side. The wrapper moves ``RigidBodyAPI``/``MassAPI`` to a parent ``/World/object`` prim and keeps the
    rotated vendor prim (with its collision and physics material) as a child.

    Args:
        source_usd: The vendor ``Aligned.usda`` to wrap.
        output_basename: File name (without extension) of the wrapper in the cache directory.

    Returns:
        Path of the wrapper USD.
    """
    # Import locally because USD/pxr is available only after simulation initialization.
    from pxr import Usd, UsdGeom, UsdPhysics

    cache_root = get_arena_asset_cache_dir().parent / "usd" / "geniesim"
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = cache_root / f"{output_basename}.usda"
    if out_path.exists():
        return str(out_path)

    source_stage = Usd.Stage.Open(source_usd)
    source_entity = source_stage.GetPrimAtPath("/World/entity")
    assert source_entity.IsValid(), f"{source_usd} has no /World/entity prim"
    mass_attr = source_entity.GetAttribute("physics:mass")
    mass = mass_attr.Get() if mass_attr and mass_attr.HasAuthoredValue() else None

    with tempfile.NamedTemporaryFile(suffix=".usda", dir=cache_root, delete=False) as tmp_file:
        tmp_path = tmp_file.name
    stage = Usd.Stage.CreateNew(tmp_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    root = stage.DefinePrim("/World/object", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    if mass is not None:
        mass_api.CreateMassAttr(float(mass))
    geometry = stage.DefinePrim("/World/object/entity", "Xform")
    geometry.GetReferences().AddReference(source_usd, "/World/entity")
    geometry.RemoveAPI(UsdPhysics.RigidBodyAPI)
    geometry.RemoveAPI(UsdPhysics.MassAPI)
    stage.GetRootLayer().Save()
    os.replace(tmp_path, out_path)
    return str(out_path)


@register_asset
class GenieBenchmarkBowl(LibraryObject):
    """Genie Sim's ``benchmark_bowl_025``: a 15.6 cm light-green ceramic bowl used by the ``stack_bowls`` task."""

    name = "genie_benchmark_bowl"
    tags = ["object", "graspable", "container", "geniesim"]
    usd_path = None  # resolved lazily: the vendor layer wrapped with a Z-up rigid root

    def __init__(self, instance_name: str | None = None, **kwargs):
        legacy = f"{GENIESIM_ASSETS_DIR}/objects/benchmark/bowl/benchmark_bowl_025/Aligned.usda"
        source = asset_or_legacy("g2_bowl", legacy)
        self.usd_path = (
            source if source != legacy else with_rigid_root_above_entity(source, "benchmark_bowl_025_rigid_root")
        )
        super().__init__(instance_name=instance_name, **kwargs)
