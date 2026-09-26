# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check conservative payload coverage and placement false positives."""

import copy
import itertools
import numpy as np
import trimesh
import unittest
from scipy.spatial.transform import Rotation
from types import SimpleNamespace

from isaaclab_arena_examples.g2_workcell.placement import evaluate_placement, payload_cover, placement_candidates


class PlacementTests(unittest.TestCase):
    def test_sphere_cover_contains_boundary_and_interior_in_tcp_frame(self):
        low, high = np.array([-0.09, -0.095, -0.03]), np.array([0.09, 0.095, 0.03])
        vertices = np.array(list(itertools.product(*zip(low, high))))
        rotation = Rotation.from_euler("xyz", [18, 31, 75], degrees=True)
        offset = np.array([0.04, -0.02, 0.013])
        spheres, _ = payload_cover(vertices, offset, rotation)
        spheres = spheres[spheres[:, 3] > 0]
        rng = np.random.default_rng(4)
        points = np.vstack([vertices, rng.uniform(low, high, (2000, 3))])
        tcp_points = rotation.apply(points) + offset
        distance = np.linalg.norm(tcp_points[:, None] - spheres[None, :, :3], axis=-1) - spheres[None, :, 3]
        self.assertTrue(np.all(distance.min(1) <= 0))

    def test_long_tool_is_centred_in_rotated_bin_and_wrong_yaw_is_rejected(self):
        vertices = np.array(list(itertools.product([-0.15, 0.17], [-0.03, 0.07], [-0.015, 0.015])))
        session = SimpleNamespace(meshes={"metal_stock": {"vertices": vertices}}, geometry={"bin": self.geometry()})
        bp = np.array([0.15, -0.29, 0.0])
        br = Rotation.from_euler("z", 90, degrees=True)
        candidates = list(placement_candidates(session, "metal_stock", bp, br))
        self.assertEqual([c[0] for c in candidates], [90, 270])
        for _, _, rotation, position in candidates:
            points = br.inv().apply(rotation.apply(vertices) + position - bp)
            sample = self.sample()
            sample["minimum_bin"] = [*points.min(0)[:2], 0.002]
            sample["maximum_bin"] = [*points.max(0)[:2], 0.032]
            self.assertTrue(evaluate_placement([sample], self.geometry())["passed"])

    def test_default_drill_candidates_preserve_validated_offsets(self):
        session = SimpleNamespace(meshes={"drill": {"vertices": np.zeros((1, 3))}}, geometry={"bin": self.geometry()})
        bp = np.array([0.15, 0.29, 0.0])
        poses = list(placement_candidates(session, "drill", bp, Rotation.identity()))
        self.assertEqual(len(poses), 12)
        self.assertEqual(poses[0][0], 270)
        np.testing.assert_allclose(poses[0][3], [0.14, 0.23, 0.0])

    def test_mesh_slab_cover_contains_head_and_handle_interiors(self):
        handle = trimesh.creation.box(extents=[0.30, 0.025, 0.025])
        head = trimesh.creation.box(extents=[0.05, 0.12, 0.038])
        head.apply_translation([0.15, 0.0, 0.0])
        mesh = trimesh.util.concatenate([handle, head])
        rotation = Rotation.from_euler("xyz", [20, 15, 295], degrees=True)
        offset = np.array([0.01, 0.03, -0.02])
        spheres, _ = payload_cover(mesh.vertices, offset, rotation, faces=mesh.faces)
        spheres = spheres[spheres[:, 3] > 0]
        rng = np.random.default_rng(7)
        points = np.vstack([
            mesh.vertices,
            rng.uniform(-0.5, 0.5, (2000, 3)) * [0.30, 0.025, 0.025],
            rng.uniform(-0.5, 0.5, (2000, 3)) * [0.05, 0.12, 0.038] + [0.15, 0, 0],
        ])
        world = rotation.apply(points) + offset
        distance = np.linalg.norm(world[:, None] - spheres[None, :, :3], axis=-1) - spheres[None, :, 3]
        self.assertTrue(np.all(distance.min(1) <= 0))

    def test_stable_but_suspended_object_is_rejected(self):
        sample = self.sample()
        sample["minimum_bin"][2] = 0.10
        self.assertFalse(evaluate_placement([sample, sample], self.geometry())["passed"])

    def test_center_inside_but_mesh_overhang_is_rejected(self):
        sample = self.sample()
        sample["maximum_bin"][0] = 0.21
        self.assertFalse(evaluate_placement([sample, sample], self.geometry())["passed"])

    def test_stable_supported_object_passes_but_drift_fails(self):
        sample = self.sample()
        self.assertTrue(evaluate_placement([sample, sample], self.geometry())["passed"])
        moved = copy.deepcopy(sample)
        moved["position"][0] += 0.01
        self.assertFalse(evaluate_placement([sample, moved], self.geometry())["passed"])

    @staticmethod
    def sample():
        return dict(
            position=[0, 0, 0.03],
            quaternion_xyzw=[0, 0, 0, 1],
            minimum_bin=[-0.09, -0.09, 0.001],
            maximum_bin=[0.09, 0.09, 0.06],
        )

    @staticmethod
    def geometry():
        return dict(inner_xy_bounds_m=[[-0.2, -0.12], [0.18, 0.12]], floor_z_m=0.002, rim_max_z_m=0.187)


if __name__ == "__main__":
    unittest.main()
