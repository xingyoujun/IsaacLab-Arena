# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject empty grasps, table contact and unstable holds."""

import copy
import unittest

from isaaclab_arena_examples.g2_workcell.grasp_evaluation import evaluate_hold


class HoldTests(unittest.TestCase):
    def test_lift_is_required_even_when_relative_pose_is_constant(self):
        sample = dict(
            position=[0, 0, 0.03], minimum_z_m=0, relative_position=[0, 0, 0], relative_quaternion_xyzw=[0, 0, 0, 1]
        )
        self.assertFalse(evaluate_hold(sample, [sample, sample])["passed"])

    def test_tilted_object_still_touching_table_fails(self):
        initial = dict(position=[0, 0, 0.03])
        sample = dict(
            position=[0, 0, 0.13], minimum_z_m=0.001, relative_position=[0, 0, 0], relative_quaternion_xyzw=[0, 0, 0, 1]
        )
        self.assertFalse(evaluate_hold(initial, [sample, sample])["passed"])

    def test_slipping_object_fails_and_stable_lift_passes(self):
        initial = dict(position=[0, 0, 0.03])
        sample = dict(
            position=[0, 0, 0.13], minimum_z_m=0.1, relative_position=[0, 0, 0], relative_quaternion_xyzw=[0, 0, 0, 1]
        )
        self.assertTrue(evaluate_hold(initial, [sample, sample])["passed"])
        slipped = copy.deepcopy(sample)
        slipped["relative_position"][0] = 0.01
        self.assertFalse(evaluate_hold(initial, [sample, slipped])["passed"])


if __name__ == "__main__":
    unittest.main()
