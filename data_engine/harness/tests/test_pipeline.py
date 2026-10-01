# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline failure and provenance tests without an Isaac Sim process."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from data_engine.annotations.skill_trace import SkillTrace, validate_trace
from data_engine.harness import catalog
from data_engine.harness.adapters.existing import preview_command
from data_engine.harness.audit import check_raw
from data_engine.harness.runner import launch, preview, transition
from data_engine.harness.steps import ControlStepObserver
from data_engine.harness.storage import digest, identity, inside, lease, read, snapshot, verify_snapshot, write


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_catalog_import_does_not_boot_simulator(self):
        code = (
            "import sys; from data_engine.harness.catalog import scenarios; "
            "s=scenarios(); assert len(s)==25; "
            "assert not any(n.startswith(('isaaclab.', 'omni.', 'torch')) for n in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", code], cwd=catalog.ROOT, check=True)
        items = catalog.scenarios()
        self.assertNotEqual(items["pine_wm/T001"]["action_profile"], items["g2/stack_bowls"]["action_profile"])
        self.assertTrue(items["pine_wm/T031"]["preview_supported"])
        self.assertFalse(items["pine_wm/T031"]["annotation_supported"])
        from data_engine.harness.adapters.existing import preview_command

        pine = [item for item in items.values() if item["adapter"] == "pine_wm"]
        self.assertEqual(len(pine), 20)
        for item in pine:
            command = preview_command(dict(scenario=item, seed=0, device="cuda:0"), self.root)
            self.assertEqual("--annotate-skills" in command, item["annotation_supported"])
            self.assertIn("--enable_cameras", command)

    def test_identity_finite_and_order_independent(self):
        self.assertEqual(identity({"a": 1, "b": 2}), identity({"b": 2, "a": 1}))
        self.assertNotEqual(identity({"seed": 1}), identity({"seed": 2}))
        with self.assertRaises(ValueError):
            identity({"bad": float("nan")})

    def test_artifact_changes_and_path_escape_rejected(self):
        path = self.root / "demo"
        path.write_text("a")
        sealed = snapshot(self.root)
        verify_snapshot(self.root, sealed)
        path.write_text("b")
        with self.assertRaises(AssertionError):
            verify_snapshot(self.root, sealed)
        with self.assertRaises(AssertionError):
            inside(self.root, "../escape")
        (self.root / "link").symlink_to(self.root.parent)
        with self.assertRaises(AssertionError):
            inside(self.root, "link/escape")

    def test_lease_and_atomic_metadata(self):
        path = self.root / "run.json"
        write(path, {"stage": "planned"})
        with lease(self.root / ".lock"):
            with self.assertRaises(RuntimeError), lease(self.root / ".lock"):
                pass
            write(path, {"stage": "sealed"})
        self.assertEqual(read(path), {"stage": "sealed"})
        with lease(self.root / ".lock"):
            pass

    def test_failed_and_timed_out_workers_retain_log(self):
        log = self.root / "failed.log"
        with self.assertRaises(AssertionError):
            launch([sys.executable, "-c", "print('failure'); raise SystemExit(2)"], log, os.environ.copy(), 10)
        self.assertIn("failure", log.read_text())
        with self.assertRaises(subprocess.TimeoutExpired):
            launch(
                [sys.executable, "-c", "import time; time.sleep(10)"], self.root / "timeout.log", os.environ.copy(), 0.1
            )
        with self.assertRaises(FileExistsError):
            launch([sys.executable, "-c", "pass"], log, os.environ.copy(), 1)

    def test_specs_detect_config_and_manifest_changes(self):
        manifest = self.root / "manifest.json"
        write(manifest, {"assets": []})
        spec = dict(
            schema="arena.collection.spec.v1",
            purpose="single_success_preview",
            max_attempts=1,
            asset_root=str(self.root),
            asset_manifest_sha256=digest(manifest),
            scenario={"adapter": "pine_wm"},
            source_sha256={"file": "one"},
        )
        spec["spec_id"] = identity(spec)
        with patch.object(catalog, "source_files", return_value={"file": "one"}):
            catalog.validate_spec(spec)
        with patch.object(catalog, "source_files", return_value={"file": "two"}):
            with self.assertRaises(AssertionError):
                catalog.validate_spec(spec)
        spec["max_attempts"] = 20
        with self.assertRaises(AssertionError):
            catalog.validate_spec(spec, check_sources=False)

    def test_preview_never_overwrites_or_auto_retries(self):
        spec = {"device": "cuda:93", "scenario": {"preview_supported": True}}
        run = self.root / "run"
        with (
            patch("data_engine.harness.runner.validate_spec"),
            patch(
                "data_engine.harness.runner.preview_command",
                return_value=[sys.executable, "-c", "raise SystemExit(3)"],
            ),
            patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root / "cache")}),
        ):
            with self.assertRaises(AssertionError):
                preview({**spec, "asset_root": str(self.root)}, run)
            self.assertEqual(read(run / "run.json")["state"], "collection_failed")
            with self.assertRaises(FileExistsError):
                preview(spec, run)

    def test_adapter_one_attempt_and_no_shell(self):
        spec = dict(scenario=catalog.scenarios()["pine_wm/T001"], device="cuda:0", seed=3)
        command = preview_command(spec, self.root)
        self.assertEqual(command[command.index("--trials") + 1], "1")
        self.assertIn("--harness-run", command)

    def test_failed_launcher_does_not_leave_its_child_running(self):
        pid_file = self.root / "child.pid"
        code = (
            "import subprocess,sys; from pathlib import Path; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            f"Path({str(pid_file)!r}).write_text(str(child.pid)); raise SystemExit(2)"
        )
        with self.assertRaises(AssertionError):
            launch([sys.executable, "-c", code], self.root / "orphan.log", os.environ.copy(), 10)
        pid = int(pid_file.read_text())
        for _ in range(50):
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            self.fail("Collector descendant survived its failed launcher")

    def test_raw_rejects_time_offset_and_wrong_action_width(self):
        import h5py
        import numpy as np

        path = self.root / "episode.hdf5"
        with h5py.File(path, "w") as file:
            demo = file.create_group("data/demo_0")
            demo.attrs["success"] = True
            demo["actions"] = np.zeros((2, 3))
            demo["initial_state/articulation/robot/joint_position"] = np.array([[0, 0, 0]])
            demo["states/articulation/robot/joint_position"] = np.array([[1, 1, 1], [2, 2, 2]])
            demo["obs/robot_joint_pos"] = np.array([[0, 0, 0], [1, 1, 1]])
        self.assertEqual(check_raw(path, 3)["steps"], 2)
        with self.assertRaises(AssertionError):
            check_raw(path, 7)
        with h5py.File(path, "r+") as file:
            file["data/demo_0/obs/robot_joint_pos"][0] = [1, 1, 1]
        with self.assertRaises(AssertionError):
            check_raw(path, 3)

    def test_illegal_lifecycle_transition_cannot_publish_success(self):
        run = {"state": "planned", "events": []}
        with self.assertRaises(AssertionError):
            transition(self.root, run, "preview_validated")
        transition(self.root, run, "collecting")
        transition(self.root, run, "collection_failed")
        with self.assertRaises(AssertionError):
            transition(self.root, run, "preview_validated")

    def test_observer_preserves_completed_step_when_capture_fails(self):
        trace = SkillTrace("task", 0.1)
        trace.start_skill("pick")
        trace.stage("approach")

        def capture():
            raise RuntimeError("encoder failed")

        observer = ControlStepObserver(lambda action: "result", lambda: trace, capture)
        with self.assertRaises(RuntimeError):
            observer(0)
        doc = trace.finish(False, "encoder failed")
        self.assertEqual(doc["num_steps"], 1)
        self.assertEqual(doc["events"][0]["name"], "capture_failed")
        self.assertFalse(observer.busy)

    def test_observer_abort_and_negative_label_ids(self):
        trace = SkillTrace("task", 0.1)
        trace.start_skill("pick")
        trace.stage("approach")

        def incomplete(action):
            raise RuntimeError("simulation failed")

        observer = ControlStepObserver(incomplete, lambda: trace)
        with self.assertRaises(RuntimeError):
            observer(0)
        self.assertEqual(trace.step, 0)
        observer.step = lambda action: action
        self.assertEqual(observer(42), 42)
        doc = trace.finish(False)
        doc["step_stage_ids"] = [-1]
        with self.assertRaises(AssertionError):
            validate_trace(doc)

    def test_observer_rejects_reentry_without_inventing_steps(self):
        trace = SkillTrace("task", 0.1)
        trace.start_skill("move")
        trace.stage("approach")
        observer = ControlStepObserver(lambda action: observer(action), lambda: trace)
        with self.assertRaises(AssertionError):
            observer(0)
        self.assertEqual(trace.step, 0)
        self.assertFalse(observer.busy)


if __name__ == "__main__":
    unittest.main()
