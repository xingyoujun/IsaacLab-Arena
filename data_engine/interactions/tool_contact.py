# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure a fingertip contact frame from the current Robotiq configuration."""

import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def fingertip_offset(env, planner, side="left"):
    """Return the selected finger's distal collision-face centre in the measured tool frame."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from data_engine.pine_wm.collection.geometry import collider_transform

    robot = planner.robot
    poses = robot.data.body_pose_w.torch[0].cpu().numpy()
    tool = poses[planner.tool_body_index]
    R_W_T = Rotation.from_quat(tool[3:])
    points = []
    cache = UsdGeom.XformCache()
    for name in (side + "_inner_finger", side + "_inner_finger_pad"):
        if name not in robot.body_names:
            continue
        body_pose = poses[robot.body_names.index(name)]
        root = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/Robot/ee_link/Robotiq_2F_85/{name}")
        if not root:
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            T_L_C = collider_transform(cache, root, prim)
            if prim.IsA(UsdGeom.Mesh):
                vertices = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            else:
                bounds = (
                    UsdGeom.BBoxCache(0, ["default", "render", "proxy", "guide"], False, True)
                    .ComputeUntransformedBound(prim)
                    .ComputeAlignedRange()
                )
                vertices = np.array(list(itertools.product(*zip(bounds.GetMin(), bounds.GetMax()))))
            local = vertices @ T_L_C[:3, :3].T + T_L_C[:3, 3]
            world = Rotation.from_quat(body_pose[3:]).apply(local) + body_pose[:3]
            points.extend(R_W_T.inv().apply(world - tool[:3]))
    assert points, "No fingertip collision surface"
    points = np.asarray(points)
    face = points[points[:, 2] > points[:, 2].max() - 0.0005]
    center = (face.min(axis=0) + face.max(axis=0)) / 2
    center[2] = points[:, 2].max()
    return center
