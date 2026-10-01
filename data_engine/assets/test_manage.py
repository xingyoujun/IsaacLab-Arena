# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for release integrity and manifest confinement."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_engine.assets import manage

spec = importlib.util.spec_from_file_location(
    "bundle_resolver", manage.ROOT / "isaaclab_arena/assets/usdcraft_scene.py"
)
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


class ReleaseIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        payload = self.root / "assets/cube/model.usdc"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"test model")
        self.manifest = {
            "schema_version": 1,
            "files": {"assets/cube/model.usdc": {"sha256": manage.digest(payload), "bytes": payload.stat().st_size}},
            "entries": {"cube": {"path": "assets/cube/model.usdc"}},
        }
        self.save()

    def save(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_default_root_is_checkout_local(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolver.bundle_root(), manage.ROOT / "local_assets/USDCraft-Scene")
            self.assertEqual(manage.bundle_root(), resolver.bundle_root())

    def test_root_override(self):
        with patch.dict("os.environ", {"ARENA_USDCRAFT_SCENE_ROOT": str(self.root)}):
            self.assertEqual(resolver.bundle_root(), self.root)
            self.assertEqual(resolver.bundle_root(manage.ROOT), manage.ROOT)

    def test_relocation_and_tamper(self):
        manage.verify(self.root)
        path = resolver.resolve_asset("cube", self.root)
        path.write_bytes(b"changed!!!")
        with self.assertRaises(AssertionError):
            manage.verify(self.root)

    def test_entry_cannot_escape_bundle(self):
        self.manifest["entries"]["cube"]["path"] = "../outside.usdc"
        self.save()
        with self.assertRaises(AssertionError):
            resolver.resolve_asset("cube", self.root)

    def test_manifest_file_cannot_escape_bundle(self):
        self.manifest["files"]["../outside.usdc"] = {"sha256": "", "bytes": 0}
        self.save()
        with self.assertRaises(AssertionError):
            manage.verify(self.root)

    def test_missing_dependency_rejected(self):
        (self.root / "assets/cube/model.usdc").unlink()
        with self.assertRaises(AssertionError):
            manage.verify(self.root)


if __name__ == "__main__":
    unittest.main()
