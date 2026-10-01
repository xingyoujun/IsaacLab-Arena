# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the independent held-object collision guard."""

import numpy as np
import unittest
from scipy.spatial.transform import Rotation

from data_engine.pine_wm.collection.geometry import (
    collider_transform,
    penetration,
    validate_held_object,
    validate_table_footprints,
)


def box(name, center, half=(0.02, 0.02, 0.02), yaw=0.0):
    return name, np.array(center), Rotation.from_euler("z", yaw).as_matrix(), np.array(half)


class CollisionGuardTest(unittest.TestCase):
    def test_scaled_rigid_body_retains_collider_size(self):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, "/Body")
        body.AddTranslateOp().Set((1, 2, 3))
        body.AddScaleOp().Set((0.5, 0.6, 0.7))
        wall = UsdGeom.Cube.Define(stage, "/Body/Wall")
        wall.AddTranslateOp().Set((0, 0.088, 0.047))
        cache = UsdGeom.XformCache()
        T_B_C = collider_transform(cache, body.GetPrim(), wall.GetPrim())
        np.testing.assert_allclose(T_B_C[:3, 3], [0, 0.0528, 0.0329], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(T_B_C[:3, :3], axis=0), [0.5, 0.6, 0.7])

    def test_empty_receptacle_and_wall_contact(self):
        held = box("cube", [0, 0, 0.77])
        floor = box("tray", [0, 0, 0.742], (0.12, 0.09, 0.002))
        walls = [box("tray", [x, 0, 0.75], (0.002, 0.09, 0.01)) for x in [-0.118, 0.118]]
        validate_held_object([held, floor, *walls], "cube")
        with self.assertRaisesRegex(RuntimeError, "held_object_collision"):
            validate_held_object([box("cube", [0.11, 0, 0.76]), floor, *walls], "cube")

    def test_rotated_separation(self):
        angle = np.pi / 4
        a = box("a", [0, 0, 0.8], (0.1, 0.01, 0.01), angle)
        b = box("b", [-0.03, 0.03, 0.8], (0.1, 0.01, 0.01), angle)
        self.assertLess(penetration(a, b), 0)
        self.assertGreater(penetration(a, a), 0.019)

    def test_round_primitive_table_support(self):
        rotation = Rotation.from_euler("x", 0.3).as_matrix()
        cylinder = ("cylinder", np.array([0, 0, 0.7575]), rotation, np.array([0.04, 0.0175, 0.0175]))
        validate_held_object([cylinder], "cylinder", primitive_geometry={"cylinder": [("cylinder", 0)]})
        sphere = ("sphere", np.array([0, 0, 0.76]), rotation, np.full(3, 0.02))
        validate_held_object([sphere], "sphere", primitive_geometry={"sphere": [("sphere", None)]})
        with self.assertRaisesRegex(RuntimeError, "crosses_table"):
            low = (sphere[0], sphere[1] - [0, 0, 0.004], *sphere[2:])
            validate_held_object([low], "sphere", primitive_geometry={"sphere": [("sphere", None)]})

    def test_round_primitive_receptacle_floor(self):
        floor = box("tray", [0, 0, 0.742], (0.12, 0.09, 0.002))
        rotation = Rotation.from_euler("xyz", [0.3, 0.14, 0.5]).as_matrix()
        axial = abs(rotation[2, 0])
        extent = 0.04 * axial + 0.0175 * np.sqrt(1 - axial * axial)
        cylinder = ("cylinder", np.array([0, 0, 0.744 + extent]), rotation, np.array([0.04, 0.0175, 0.0175]))
        shapes = {"cylinder": [("cylinder", 0)]}
        validate_held_object([cylinder, floor], "cylinder", primitive_geometry=shapes)
        with self.assertRaisesRegex(RuntimeError, "held_object_collision"):
            lower = (cylinder[0], cylinder[1] - [0, 0, 0.003], *cylinder[2:])
            validate_held_object([lower, floor], "cylinder", primitive_geometry=shapes)
        sphere = box("sphere", [0, 0, 0.764], (0.02, 0.02, 0.02), 0.5)
        validate_held_object([sphere, floor], "sphere", primitive_geometry={"sphere": [("sphere", None)]})

    def test_convex_mesh_table_support(self):
        vertices = np.array([[-1, -1, -1], [1, -1, 1], [-1, 1, 1], [1, 1, -1]])
        half = np.full(3, 0.02)
        rotation = Rotation.from_euler("xy", [0.4, 0.5]).as_matrix()
        bottom = float(np.min((vertices * half) @ rotation[2, :]))
        item = ("mesh", np.array([0, 0, 0.74 - bottom]), rotation, half)
        shapes = {"mesh": [("mesh", vertices)]}
        validate_held_object([item], "mesh", primitive_geometry=shapes)
        with self.assertRaisesRegex(RuntimeError, "crosses_table"):
            lower = (item[0], item[1] - [0, 0, 0.003], *item[2:])
            validate_held_object([lower], "mesh", primitive_geometry=shapes)

    def test_table_penetration_and_edge(self):
        with self.assertRaisesRegex(RuntimeError, "crosses_table"):
            validate_held_object([box("cube", [0, 0, 0.75])], "cube")
        with self.assertRaisesRegex(RuntimeError, "outside_table"):
            validate_table_footprints([box("bar", [0.46, 0, 0.8], (0.06, 0.01, 0.01))])
        validate_table_footprints([box("bar", [0.30, 0, 0.8], (0.06, 0.01, 0.01))])


if __name__ == "__main__":
    unittest.main()
