# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Selected convex-decomposition cooking overrides without modifying source USD files."""

from isaaclab.utils.configclass import configclass
from pxr import PhysxSchema, UsdPhysics

from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg


@configclass
class ConvexDecompositionCfg(UsdPrimSpawnPhysicsCfg):
    """Preserve decomposition while avoiding shrink-wrapped sliver hulls."""

    min_thickness: float = 0.001
    """Minimum hull thickness in asset units; scene scale still applies."""
    shrink_wrap: bool = False
    """Whether to project hull points back onto the original visual mesh."""

    def validate_target(self, prim, root):
        """Require an existing convex-decomposition collider and positive thickness."""
        assert self.min_thickness > 0
        assert prim.HasAPI(UsdPhysics.CollisionAPI), f"Missing collider: {prim.GetPath()}"
        assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexDecomposition"

    def apply(self, prim, root):
        """Override cooking properties before PhysX imports this collider."""
        api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        api.CreateMinThicknessAttr(self.min_thickness)
        api.CreateShrinkWrapAttr(self.shrink_wrap)
