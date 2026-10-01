# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collection references into existing Arena task families, without a parallel environment registry."""

import platform
import subprocess
import sys
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from data_engine.harness.storage import digest, identity, inside

ROOT = Path(__file__).resolve().parents[2]


def scenarios():
    """List existing Arena collection references; do not register new environment classes."""
    from data_engine.harness.adapters.catalogs import scenario_definitions

    return scenario_definitions()


def source_files(adapter):
    """Pin executable/configuration inputs, excluding docs, outputs and unrelated task families."""
    prefixes = [
        "data_engine/harness",
        "data_engine/annotations",
        "isaaclab_arena/recording",
        "data_engine/recording",
        "data_engine/cli",
        "data_engine/planning",
        "isaaclab_arena/environments",
        "isaaclab_arena/terms",
        "isaaclab_arena/assets",
        "data_engine/assets",
        "tools/data_collection",
    ]
    files = set()
    for prefix in prefixes:
        for path in (ROOT / prefix).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml"} and "__pycache__" not in path.parts:
                if "tests" not in path.parts and not path.name.startswith("test_"):
                    files.add(path)
    from data_engine.harness.adapters.catalogs import source_patterns

    for pattern in source_patterns(adapter):
        for path in ROOT.glob(pattern):
            if (
                path.is_file()
                and path.suffix in {".py", ".json", ".yaml", ".yml"}
                and "__pycache__" not in path.parts
                and "tests" not in path.parts
                and not path.name.startswith("test_")
            ):
                files.add(path)
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(files)}


def resolve(scenario_id, assets, seed=0, device="cuda:0"):
    """Freeze a preview spec and verify asset bytes; this is not physical task qualification."""
    from data_engine.assets.manage import verify

    catalog = scenarios()
    assert scenario_id in catalog, f"Unknown scenario: {scenario_id}"
    assert device.startswith("cuda:") and device[5:].isdigit(), "Specify a CUDA device index"
    assets = Path(assets).resolve()
    with redirect_stdout(sys.stderr):
        manifest = verify(assets)
    scenario = catalog[scenario_id]
    for name in scenario["required_assets"]:
        assert name in manifest["entries"], f"Missing asset ID: {name}"
        inside(assets, manifest["entries"][name]["path"])
    from data_engine.harness.adapters.catalogs import preflight_family

    with redirect_stdout(sys.stderr):
        preflight_family(assets, scenario)
    packages = {}
    for name in ("isaacsim", "isaaclab", "torch", "warp-lang", "numpy", "h5py"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not in distribution metadata"
    spec = dict(
        schema="arena.collection.spec.v1",
        scenario=scenario,
        seed=seed,
        device=device,
        asset_root=str(assets),
        asset_manifest_sha256=digest(assets / "manifest.json"),
        source_sha256=source_files(scenario["adapter"]),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        purpose="single_success_preview",
        max_attempts=1,
        quality_profile="strict_gpu_training_v1",
        runtime={"python": platform.python_version(), "platform": platform.platform(), "packages": packages},
    )
    spec["spec_id"] = identity(spec)
    return spec


def validate_spec(spec, check_sources=True):
    """Require an unchanged resolved preview contract before launching a worker."""
    assert spec["schema"] == "arena.collection.spec.v1"
    assert spec["purpose"] == "single_success_preview" and spec["max_attempts"] == 1
    assert spec["spec_id"] == identity({k: v for k, v in spec.items() if k != "spec_id"})
    assert digest(Path(spec["asset_root"]) / "manifest.json") == spec["asset_manifest_sha256"]
    if check_sources:
        assert (
            source_files(spec["scenario"]["adapter"]) == spec["source_sha256"]
        ), "Source/config changed; resolve a new run"


def verify_assets(spec):
    """Recheck payload bytes, including modified files with an unchanged manifest."""
    from data_engine.assets.manage import verify

    with redirect_stdout(sys.stderr):
        verify(Path(spec["asset_root"]))
