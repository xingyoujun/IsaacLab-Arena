# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Audit GPU PhysX contact forces, including contacts against unfiltered table geometry."""

import numpy as np

import warp as wp
from isaaclab_physx.physics import PhysxManager
from pxr import Usd, UsdPhysics


class ContactAudit:
    """Allow only intended finger/object forces; arm and wrist contacts are forbidden."""

    def __init__(self, env, current, names):
        self.env = env
        self.current = current
        self.names = names
        self.events = 0
        self.forbidden = []
        self.max_force = 0.0
        root = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")
        self.paths = [str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert self.paths, "Robot contact audit found no rigid bodies"
        filters = []
        self.filter_names = []
        self.filter_paths = []
        for name in names:
            root = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    filters.append(str(prim.GetPath()))
                    self.filter_names.append(name)
                    self.filter_paths.append(str(prim.GetPath()))
        view = PhysxManager.get_physics_sim_view()
        self.view = view.create_rigid_contact_view(
            self.paths, filter_patterns=[filters] * len(self.paths), max_contact_data_count=8192
        )
        assert self.view.sensor_count == len(self.paths)
        self.paths = list(self.view.sensor_paths)

    def reset(self):
        self.events = 0
        self.forbidden.clear()
        self.max_force = 0.0

    def check(self):
        if self.current["phase"] == "reset":
            return
        dt = self.env.sim.get_physics_dt()
        net = wp.to_torch(self.view.get_net_contact_forces(dt=dt)).cpu().numpy()
        matrix = wp.to_torch(self.view.get_contact_force_matrix(dt=dt)).cpu().numpy()
        for index, path in enumerate(self.paths):
            relative = path.split("/Robot/", 1)[1]
            if relative in {"base_link", "base_link_inertia"}:
                continue
            force = float(np.linalg.norm(net[index]))
            self.max_force = max(self.max_force, force)
            finger = relative.rsplit("/", 1)[-1] in {
                "left_inner_finger",
                "right_inner_finger",
                "left_inner_finger_pad",
                "right_inner_finger_pad",
            }
            pair_forces = np.linalg.norm(matrix[index], axis=1)
            unfiltered = net[index] - matrix[index].sum(axis=0)
            residual = float(np.linalg.norm(unfiltered))
            for j, name in enumerate(self.filter_names):
                allowed = (
                    finger
                    and name == self.current.get("manipulated")
                    and not self.filter_paths[j].endswith(("/carcass", "/base"))
                )
                if "contact_bodies" in self.current:
                    allowed = allowed and self.filter_paths[j].rsplit("/", 1)[-1] in self.current["contact_bodies"]
                if not allowed:
                    residual += float(pair_forces[j])
            if max(force, float(pair_forces.max(initial=0)), residual) < 0.02:
                continue
            self.events += 1
            if residual > 0.05:
                event = {
                    "robot_body": path,
                    "net_force_n": force,
                    "unallowed_force_n": residual,
                    "object_forces_n": {
                        path: float(np.linalg.norm(matrix[index, j])) for j, path in enumerate(self.filter_paths)
                    },
                }
                if self.current.get("contact_details"):
                    _, points, normals, distances, counts, starts = self.view.get_contact_data(dt=dt)
                    counts, starts = wp.to_torch(counts).cpu().numpy(), wp.to_torch(starts).cpu().numpy()
                    points, normals = wp.to_torch(points).cpu().numpy(), wp.to_torch(normals).cpu().numpy()
                    distances = wp.to_torch(distances).cpu().numpy()
                    details = []
                    for j, name in enumerate(self.filter_paths):
                        start, count = int(starts[index, j]), int(counts[index, j])
                        if count:
                            details.append(
                                dict(
                                    body=name,
                                    points=points[start : start + count].tolist(),
                                    normals=normals[start : start + count].tolist(),
                                    distances=distances[start : start + count].tolist(),
                                )
                            )
                    event["contact_details"] = details
                self.forbidden.append(event)
                raise RuntimeError(f"forbidden_physx_contact:{event}")
