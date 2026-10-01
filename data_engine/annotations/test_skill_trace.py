# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python regressions for control-boundary annotation semantics."""

import json
import tempfile
import unittest

from data_engine.annotations.skill_trace import SkillTrace, validate_trace, write_hdf5_trace


class SkillTraceTest(unittest.TestCase):
    def test_repeated_skills_rejected_candidate_and_exact_coverage(self):
        trace = SkillTrace("stack", 1 / 15, actor_id="left_arm")
        for obj in ("a", "b"):
            trace.start_skill("pick", entities={"object": obj})
            trace.stage("approach")
            trace.reject_stage("planning_failed")
            trace.stage("approach")
            trace.before_step()
            trace.after_step()
            trace.stage("close_gripper")
            trace.before_step()
            trace.after_step()
            trace.end_skill()
        doc = trace.finish(True)
        self.assertEqual(doc["step_skill_ids"], [0, 0, 1, 1])
        self.assertEqual(doc["step_stage_ids"], [1, 2, 4, 5])
        self.assertEqual(doc["stages"][0]["execution_status"], "failed")
        self.assertEqual(doc["stages"][0]["start_step"], doc["stages"][0]["end_step"])
        self.assertIsNone(doc["skills"][0]["success"])

    def test_label_frozen_during_step_and_abort_does_not_invent_frame(self):
        trace = SkillTrace("push", 0.1)
        trace.start_skill("push")
        trace.stage("contact")
        trace.before_step()
        with self.assertRaises(AssertionError):
            trace.stage("push")
        trace.abort_step("physics stopped")
        doc = trace.finish(False, "physics stopped")
        self.assertEqual(doc["num_steps"], 0)
        self.assertEqual(doc["skills"][0]["execution_status"], "failed")
        self.assertEqual(doc["events"][0]["name"], "incomplete_transition")

    def test_steps_require_labels_and_success_requires_evidence(self):
        trace = SkillTrace("any", 0.1)
        with self.assertRaises(AssertionError):
            trace.before_step()
        trace.start_skill("grasp")
        with self.assertRaises(AssertionError):
            trace.end_skill(success=True)

    def test_hdf5_round_trip_and_length_mismatch(self):
        import h5py

        trace = SkillTrace("any", 0.1, actor_id="right_gripper", env_id=3, source="human")
        trace.start_skill("grasp")
        trace.stage("close_gripper")
        trace.before_step()
        trace.after_step()
        trace.end_skill(success=False, evidence={"slip_m": 0.2})
        doc = trace.finish(False)
        with tempfile.NamedTemporaryFile(suffix=".hdf5") as file:
            with h5py.File(file.name, "w") as h5:
                demo = h5.create_group("demo")
                with self.assertRaises(AssertionError):
                    write_hdf5_trace(demo, doc, 2)
                write_hdf5_trace(demo, doc, 1)
                self.assertEqual(demo["annotations/step_stage_ids"][:].tolist(), [0])
                metadata = json.loads(demo["annotations/metadata_json"][()])
                self.assertEqual(metadata["actor_id"], "right_gripper")
                self.assertFalse(metadata["skills"][0]["success"])
                with self.assertRaises(AssertionError):
                    write_hdf5_trace(demo, doc, 1)
        doc["stages"][0]["end_step"] = 0
        with self.assertRaises(AssertionError):
            validate_trace(doc)


if __name__ == "__main__":
    unittest.main()
