# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read PhysX convex pieces for concave USD colliders without changing their geometry."""

import numpy as np


def convex_pieces(prim):
    """Return cooked local vertices for a convex-decomposition mesh; None for other shapes."""
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, UsdGeom, UsdPhysics, UsdUtils

    if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        return None
    if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "convexDecomposition":
        return None
    pieces, statuses = [], []

    def receive(status, hulls):
        statuses.append(status)
        for hull in hulls:
            pieces.append(np.array([[v.x, v.y, v.z] for v in hull.vertices], dtype=np.float64))

    stage_id = UsdUtils.StageCache.Get().Insert(prim.GetStage()).ToLongInt()
    get_physx_cooking_interface().request_convex_collision_representation(
        stage_id=stage_id,
        collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
        run_asynchronously=False,
        on_result=receive,
    )
    assert (
        statuses == [PhysxCollisionRepresentationResult.RESULT_VALID] and pieces
    ), f"No valid convex decomposition for {prim.GetPath()}: {statuses}"
    assert all(vertices.shape[0] >= 4 and np.isfinite(vertices).all() for vertices in pieces)
    # The cooking API does not promise hull order. Keep obstacle indices reproducible.
    return sorted(pieces, key=lambda vertices: tuple(np.r_[vertices.min(axis=0), vertices.max(axis=0)]))
