# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Link-aware appliance clearance for contact motions excluded from the planner world."""

import numpy as np
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation


class ApplianceGuard:
    """Check arm/camera spheres against each measured appliance collider and the tabletop."""

    def __init__(self, env, planner, kind):
        from pxr import Usd, UsdGeom, UsdPhysics

        from data_engine.pine_wm.collection.geometry import collider_transform

        self.env, self.planner, self.kind = env, planner, kind
        self.geometry = []
        for entry in yaml.safe_load(Path(planner.cfg.lula_robot_description).read_text())["collision_spheres"]:
            for link, spheres in entry.items():
                self.geometry.append(
                    (link, np.array([s["center"] for s in spheres]), np.array([s["radius"] for s in spheres]))
                )
        self.shapes = []
        root = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{kind}")
        cache = UsdGeom.XformCache()
        for body in Usd.PrimRange(root):
            if not body.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            for prim in Usd.PrimRange(body):
                if (
                    prim.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                ):
                    bounds = (
                        UsdGeom.BBoxCache(0, ["default", "render", "proxy", "guide"], False, True)
                        .ComputeUntransformedBound(prim)
                        .ComputeAlignedRange()
                    )
                    transform = collider_transform(cache, body, prim)
                    self.shapes.append(
                        (body.GetName(), transform, np.array(bounds.GetMin()), np.array(bounds.GetMax()))
                    )
        self.checks, self.minimum = 0, 1e6
        self.cached_step = None

    def refresh(self):
        """Freeze obstacle poses for planning until simulation advances another substep."""
        step = self.env._sim_step_counter
        if step == self.cached_step:
            return
        asset = self.env.scene[self.kind]
        poses = asset.data.body_pose_w.torch[0].cpu().numpy()
        centers, rotations, half_extents = [], [], []
        for name, transform, low, high in self.shapes:
            if name in asset.body_names:
                pose = poses[asset.body_names.index(name)]
            else:
                pose = self.env.scene["kettle_power_base"].data.root_pose_w.torch[0].cpu().numpy()
            R_W_L = Rotation.from_quat(pose[3:]).as_matrix()
            basis = R_W_L @ transform[:3, :3]
            scale = np.linalg.norm(basis, axis=0)
            centers.append(pose[:3] + R_W_L @ transform[:3, 3] + basis @ ((low + high) / 2))
            rotations.append(basis / scale)
            half_extents.append((high - low) / 2 * scale)
        self.boxes = np.array(centers), np.array(rotations), np.array(half_extents)
        self.cached_step = step

    def check(self, q, stage="planned"):
        from data_engine.pine_wm.collection.geometry import sphere_box_distances

        self.refresh()
        for link, local_centers, radii in self.geometry:
            T_B_L = self.planner.kinematics.pose(np.asarray(q, dtype=np.float64), link)
            points = (
                local_centers @ np.array(T_B_L.rotation.matrix()).T
                + np.array(T_B_L.translation)
                + self.planner.base_pos
            )
            if link not in {"base_link_inertia", "shoulder_link"}:
                gap = float(np.min(points[:, 2] - radii) - 0.74)
                if gap < -0.001:
                    self.planner.safety.reject(stage=stage, reason="table_clearance", link=link, gap_m=gap)
            # Actual finger-to-body forces are audited per physics step. This
            # coarse gripper sphere cannot resolve a 22 mm button beside a handle.
            if link == "pine_wm_gripper_base":
                continue
            distances = sphere_box_distances(points, radii, *self.boxes)
            gap = float(distances.min())
            self.minimum = min(self.minimum, gap)
            if gap < 0.002:
                self.planner.safety.reject(stage=stage, reason="appliance_clearance", link=link, gap_m=gap)
        self.checks += 1
