# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Extract triangulated source collision surfaces in the asset frame."""


def mesh_parts(stage, root):
    """Return triangulated meshes in the root prim frame, retaining collision provenance."""
    import numpy as np

    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(root).GetInverse()
    parts = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        if not len(vertices):
            continue
        transform = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse)
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ transform)[:, :3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        triangles, offset = [], 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[offset : offset + count]
            triangles.extend([[face[0], face[i], face[i + 1]] for i in range(1, count - 1)])
            offset += count
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        approximation = (
            str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            else None
        )
        parts.append({
            "path": str(prim.GetPath()),
            "vertices": vertices,
            "faces": np.asarray(triangles, dtype=int),
            "collision": collision,
            "approximation": approximation,
        })
    return parts


def selected_mesh(parts):
    """Prefer authored collision meshes, falling back to visual geometry explicitly."""
    import numpy as np

    selected = [p for p in parts if p["collision"]] or parts
    vertices, faces, offset = [], [], 0
    for part in selected:
        vertices.append(part["vertices"])
        faces.append(part["faces"] + offset)
        offset += len(part["vertices"])
    return (
        np.concatenate(vertices),
        np.concatenate(faces),
        bool(any(p["collision"] for p in parts)),
    )
