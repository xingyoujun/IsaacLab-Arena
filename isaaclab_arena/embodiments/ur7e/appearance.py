# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared visual-only black Robotiq material for all UR7e tasks."""

from pxr import Sdf, Usd, UsdGeom, UsdShade


def apply_black_gripper(stage: Usd.Stage, robot_path: str) -> None:
    """Override only the Robotiq subtree, including instanced finger materials."""
    candidates = [f"{robot_path}/{suffix}" for suffix in ("ee_link/Robotiq_2F_85", "Gripper/Robotiq_2F_85")]
    roots = [stage.GetPrimAtPath(path) for path in candidates if stage.GetPrimAtPath(path).IsValid()]
    assert len(roots) == 1, f"Expected one Robotiq subtree under {robot_path}"
    root = roots[0]
    assert not root.IsInstance(), "Author the material on a non-instance gripper root"
    material_path = root.GetPath().AppendPath("ArenaBlackGripperMaterial")
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.015, 0.015, 0.015))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(root).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    # RTX retains the instanced finger's per-face materials despite the root
    # binding. Make only its visual branches editable and bind every mesh/subset
    # for rendering. All-purpose/physics bindings and collision branches stay intact.
    for side in ("left", "right"):
        visuals = stage.GetPrimAtPath(root.GetPath().AppendPath(f"{side}_inner_finger/visuals"))
        if not visuals.IsValid():
            continue
        visuals.SetInstanceable(False)
        for child in Usd.PrimRange(visuals):
            if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Subset):
                binding = UsdShade.MaterialBindingAPI.Apply(child)
                for purpose in (UsdShade.Tokens.full, UsdShade.Tokens.preview):
                    binding.Bind(
                        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose=purpose
                    )


def spawn_ur7e_with_black_gripper(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the standard robot and override gripper appearance on every spawned root."""
    import isaaclab.sim as sim_utils

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = prim.GetStage()
    for path in sim_utils.find_matching_prim_paths(prim_path, stage=stage):
        apply_black_gripper(stage, path)
    return prim
