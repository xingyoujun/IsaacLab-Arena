# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression checks for the eight-case orchestration and recoverable publication."""

import h5py
import importlib.util
import json
import numpy as np
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "isaaclab_arena_cumotion/scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("prepare_rr_sim2real_all", SCRIPTS / "prepare_rr_sim2real_all.py")
pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case = pipeline.CASES[-2]  # GPTSOL toast rests at +4 mm and presses negative.

    def recording(self, path, velocity=0):
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            data = f.create_group("data")
            data.attrs["env_args"] = '{"env_name":"legacy"}'
            for i, last in enumerate((-0.0535, 0.004)):
                d = data.create_group(f"demo_{i}")
                d.attrs.update(success=True, num_samples=2)
                targets = np.zeros((2, 7))
                targets[:, -1] = 0.785
                d.create_dataset("joint_pos_target", data=targets)
                d.create_dataset(f"states/articulation/{self.case.key}/joint_position", data=[[0.004], [last]])
                d.create_dataset(f"states/articulation/{self.case.key}/joint_velocity", data=[[velocity], [0]])

    def test_catalog(self):
        self.assertEqual(len(pipeline.CASES), 8)
        self.assertEqual(sum(bool(c.source) for c in pipeline.CASES), 4)
        self.assertEqual(len({c.name for c in pipeline.CASES}), 8)
        self.assertTrue(all("v2" not in c.name for c in pipeline.CASES))

    def test_negative_joint_and_velocity_rejection(self):
        source = self.root / "source.hdf5"
        self.recording(source)
        self.assertEqual(pipeline.good_demos(source, self.case, True), ["demo_0"])
        self.recording(source, velocity=0.02)
        self.assertEqual(pipeline.good_demos(source, self.case, True), [])
        self.assertEqual(pipeline.good_demos(source, self.case, False), ["demo_0"])

    def test_source_reuse_preserves_metadata(self):
        case = pipeline.Case(**dict(pipeline.asdict(self.case), source="source.hdf5"))
        self.recording(self.root / "rr_sim2real_raw/source.hdf5")
        work = self.root / "work"
        work.mkdir()
        with patch.object(pipeline, "DATASETS", self.root), patch.object(pipeline, "run_workers") as workers:
            merged = pipeline.collect(case, work, SimpleNamespace(target=1), "python")
            workers.assert_not_called()
        with h5py.File(merged) as f:
            self.assertEqual(list(f["data"]), ["demo_0"])
            self.assertEqual(json.loads(f["data"].attrs["env_args"])["env_name"], "legacy")
            self.assertEqual(f["data"].attrs["total"], 2)

    def test_render_uses_randomization_and_rejects_old_cache(self):
        work = self.root / "work"
        work.mkdir()
        (work / "logs").mkdir()
        merged = work / "example.hdf5"
        sidecars = Path(f"{merged}.cameras")
        sidecars.mkdir()
        args = SimpleNamespace(target=3, workers=2, render_workers=None, randomize_seed=0, skies=self.root, stagger=0)
        calls = []

        def fake_workers(commands, stagger):
            calls.extend(commands)
            pipeline.write_json(sidecars / "render_appearance.json", pipeline.APPEARANCE)
            for i in range(3):
                (sidecars / f"demo_{i}_realsense_d435_rgb.mp4").write_bytes(b"unit-test-placeholder")
                pipeline.write_json(sidecars / f"demo_{i}_randomization.json", {"index": i})
            return [0, 0]

        with patch.object(pipeline, "run_workers", side_effect=fake_workers):
            pipeline.render(self.case, work, merged, args, "python")
            self.assertEqual(len(calls), 2)
            self.assertTrue(all("--randomize" in command for command, _ in calls))
            pipeline.render(self.case, work, merged, args, "python")
            self.assertEqual(len(calls), 2)  # Complete renders are skipped.
            pipeline.write_json(sidecars / "render_appearance.json", {"gripper": "black_v1"})
            with self.assertRaises(AssertionError):
                pipeline.render(self.case, work, merged, args, "python")

    def test_publication_resume_between_pair_members(self):
        work = self.root / self.case.name
        work.mkdir()
        final = self.root / "final"
        final.mkdir()
        sources = (work / "lerobot", work / "zarr")
        names = (self.case.name, self.case.name + "_dp.zarr")
        for source, name in zip(sources, names):
            source.mkdir()
            (source / "payload").write_text("new")
            (final / name).mkdir()
            (final / name / "payload").write_text("old")
        original = Path.rename

        def fail_second(source, destination):
            if source == final / names[1]:
                raise OSError("simulated interruption between LeRobot and Zarr publication")
            return original(source, destination)

        with patch.object(Path, "rename", fail_second):
            with self.assertRaises(OSError):
                pipeline.publish_pair(work, final, self.case, sources, {"episodes": 200})
        pipeline.publish_pair(work, final, self.case, sources, {"episodes": 200})
        pipeline.publish_pair(work, final, self.case, sources, {"episodes": 200})
        transaction = json.loads((work / "publication.json").read_text())
        self.assertTrue(transaction["done"])
        for name in names:
            self.assertEqual((final / name / "payload").read_text(), "new")
            backup = work.parent / "previous_datasets" / transaction["id"] / name
            self.assertEqual((backup / "payload").read_text(), "old")


if __name__ == "__main__":
    unittest.main()
