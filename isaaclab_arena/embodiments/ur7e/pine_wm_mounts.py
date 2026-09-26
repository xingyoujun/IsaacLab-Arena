# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Photo-derived visual camera brackets, anchored to the calibrated pine_wm rig."""

import numpy as np

from pxr import Gf, Sdf, UsdGeom, UsdShade


def _material(stage, path, color, metallic, roughness):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _rounded_rectangle(width, length, radius):
    points = []
    for cx, cy, first_angle in (
        (width / 2 - radius, length / 2 - radius, 0),
        (-width / 2 + radius, length / 2 - radius, 90),
        (-width / 2 + radius, -length / 2 + radius, 180),
        (width / 2 - radius, -length / 2 + radius, 270),
    ):
        for angle in np.linspace(first_angle, first_angle + 90, 9):
            radians = np.deg2rad(angle)
            points.append((cx + radius * np.cos(radians), cy + radius * np.sin(radians)))
    return np.asarray(points)


def _extruded_ring(stage, path, outer, inner, thickness, transform, material):
    """Create a closed mesh with a real through opening, in the supplied local frame."""
    count = len(outer)
    points = []
    for z in (-thickness / 2, thickness / 2):
        for contour in (outer, inner):
            for x, y in contour:
                points.append((transform @ np.array([x, y, z, 1.0]))[:3].tolist())
    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.extend([
            (i, count + i, count + j, j),
            (2 * count + i, 2 * count + j, 3 * count + j, 3 * count + i),
            (i, j, 2 * count + j, 2 * count + i),
            (count + i, 3 * count + i, 3 * count + j, count + j),
        ])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    indices = []
    for face in faces:
        indices.extend(face)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def build_camera_mounts(stage, tool_path, calibration):
    """Add an annular collar, two windowed support arms and rear camera mounting plates.

    Dimensions are approximate from the user's photograph. The robot spawner adds
    conservative collisions; optical transforms and the 13 mm spacer stay unchanged.
    """
    root = f"{tool_path}/PrintedCameraMount"
    UsdGeom.Xform.Define(stage, root)
    plastic = _material(stage, f"{root}/BlackNylon", (0.018, 0.020, 0.022), 0.0, 0.65)
    steel = _material(stage, f"{root}/FastenerSteel", (0.15, 0.16, 0.17), 0.8, 0.3)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    T_T_M = np.eye(4)
    T_T_M[2, 3] = 0.0065
    _extruded_ring(stage, f"{root}/Collar", circle * 0.041, circle * 0.032, 0.009, T_T_M, plastic)
    for name, camera in calibration["cameras"].items():
        T_T_C = np.asarray(camera["T_tool0_cam"])
        # Rear of D405: the optical plane is z=0; housing extends to z=-24 mm.
        end = (T_T_C @ np.array([0.009, 0.0, -0.027, 1.0]))[:3]
        radial = end[:2] / np.linalg.norm(end[:2])
        start = np.array([radial[0] * 0.032, radial[1] * 0.032, 0.0065])
        along = end - start
        distance = np.linalg.norm(along)
        along /= distance
        across = np.cross(along, [0.0, 0.0, 1.0])
        across /= np.linalg.norm(across)
        normal = np.cross(across, along)
        T_T_M = np.eye(4)
        T_T_M[:3, :3] = np.column_stack((across, along, normal))
        T_T_M[:3, 3] = (start + end) / 2
        length = distance + 0.020
        _extruded_ring(
            stage,
            f"{root}/{name}_Arm",
            _rounded_rectangle(0.032, length, 0.006),
            _rounded_rectangle(0.016, length - 0.023, 0.004),
            0.006,
            T_T_M,
            plastic,
        )
        plate = UsdGeom.Cube.Define(stage, f"{root}/{name}_RearPlate")
        plate.CreateSizeAttr(1.0)
        T_T_P = T_T_C.copy()
        T_T_P[:3, 3] = end
        transform = UsdGeom.Xformable(plate)
        transform.AddTransformOp().Set(Gf.Matrix4d(T_T_P.T.tolist()))
        transform.AddScaleOp().Set(Gf.Vec3f(0.038, 0.032, 0.006))
        UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(plastic)
        for index, x in enumerate((-0.012, 0.012)):
            screw = UsdGeom.Cylinder.Define(stage, f"{root}/{name}_Screw{index}")
            screw.CreateRadiusAttr(0.0028)
            screw.CreateHeightAttr(0.002)
            screw.CreateAxisAttr("Z")
            T_T_S = T_T_C.copy()
            T_T_S[:3, 3] = (T_T_C @ np.array([0.009 + x, 0.008, -0.031, 1.0]))[:3]
            UsdGeom.Xformable(screw).AddTransformOp().Set(Gf.Matrix4d(T_T_S.T.tolist()))
            UsdShade.MaterialBindingAPI.Apply(screw.GetPrim()).Bind(steel)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{tool_path}/GripperSpacer")).Bind(plastic)
