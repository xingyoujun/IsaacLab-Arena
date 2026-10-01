# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serial Arena collection jobs with bounded workers and immutable successful payloads."""

import os
import signal
import subprocess
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from data_engine.harness.adapters.existing import preview_command
from data_engine.harness.audit import audit_run
from data_engine.harness.catalog import ROOT, validate_spec, verify_assets
from data_engine.harness.storage import lease, read, snapshot, write


def now():
    return datetime.now(timezone.utc).isoformat()


def transition(run, manifest, state, **details):
    permitted = {
        "planned": {"planned", "collecting"},
        "collecting": {"raw_sealed", "collection_failed", "cancelled"},
        "raw_sealed": {"preview_validated", "audit_failed"},
        "preview_validated": {"preview_validated", "audit_failed"},
        "audit_failed": {"preview_validated", "audit_failed"},
    }
    assert state in permitted.get(manifest["state"], set()), f"Invalid transition: {manifest['state']} -> {state}"
    manifest["state"] = state
    manifest["events"].append(dict(at=now(), state=state, **details))
    write(run / "run.json", manifest)


def launch(command, log, env, timeout_s):
    """Run one process group with bounded shutdown, retaining diagnostic output on any failure."""
    assert timeout_s > 0
    started = time.monotonic()
    with log.open("x") as output:
        child = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = child.wait(timeout=timeout_s)
            assert code == 0, f"Collector exited {code}; inspect {log}"
        except BaseException:
            # A failed launcher may leave its simulator child alive after it exits.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            raise
    return time.monotonic() - started


def preview(spec, destination, timeout_s=1800):
    """Execute exactly one preview attempt; never overwrite or automatically retry a recording."""
    validate_spec(spec)
    assert spec["scenario"]["preview_supported"]
    run = Path(destination).resolve()
    # Shared across checkouts using this user's CUDA device, not just one output directory.
    lock_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "arena-data-harness"
    device_lock = lock_root / (spec["device"].replace(":", "_") + ".lock")
    with lease(device_lock):
        run.mkdir(parents=True, exist_ok=False)
        with lease(run / ".lock"):
            manifest = dict(
                schema="arena.collection.run.v1",
                run_id=str(uuid.uuid4()),
                state="planned",
                spec=spec,
                events=[],
                artifacts={},
                created_at=now(),
            )
            transition(run, manifest, "planned")
            command = preview_command(spec, run)
            env = dict(
                os.environ, ARENA_USDCRAFT_SCENE_ROOT=spec["asset_root"], OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y"
            )
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            transition(run, manifest, "collecting", command=command)
            try:
                elapsed = launch(command, run / "worker.log", env, timeout_s)
                validate_spec(spec)
                verify_assets(spec)
                manifest["artifacts"] = snapshot(run / "payload")
                assert manifest["artifacts"], "Worker produced no recording"
                transition(run, manifest, "raw_sealed", worker_seconds=elapsed)
            except BaseException as error:
                state = "cancelled" if isinstance(error, KeyboardInterrupt) else "collection_failed"
                transition(run, manifest, state, reason=f"{type(error).__name__}: {error}")
                raise
            return _audit(run, manifest)


def _audit(run, manifest):
    try:
        report = audit_run(run)
    except BaseException as error:
        transition(run, manifest, "audit_failed", reason=f"{type(error).__name__}: {error}")
        raise
    transition(run, manifest, "preview_validated")
    return report


def resume_audit(run):
    """Resume only validation of sealed data; never recollect or continue a partial simulator state."""
    run = Path(run).resolve()
    with lease(run / ".lock"):
        manifest = read(run / "run.json")
        assert manifest["state"] in {"raw_sealed", "preview_validated", "audit_failed"}, "No sealed recording to audit"
        validate_spec(manifest["spec"])
        return _audit(run, manifest)
