#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# One fixed-pose successful HDF5 demo per comparison toaster, not bulk collection.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH="$REPO_ROOT:${ARENA_ISAACLAB_SOURCE:-/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab}"
mkdir -p outputs
RUN_ROOT="$(mktemp -d "$REPO_ROOT/outputs/toaster_knobs_XXXXXX")"
echo "Qualification outputs: $RUN_ROOT"
for method in articraft miniworkflow_gptsol miniworkflow_astra; do
    echo "Testing $method"
    .venv/bin/python isaaclab_arena_cumotion/scripts/ur7e_turn_toaster_knob_cumotion.py \
        --env "ur7e_${method}_turn_toaster_knob" --output "$RUN_ROOT/$method" \
        --record-demo --audit-self-collision > "$RUN_ROOT/${method}.log" 2>&1
    .venv/bin/python - "$RUN_ROOT/$method/result.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["success"], report
assert Path(report["recorded_demo"]).is_file(), report
assert report["self_collision_audit"]["colliding_steps"] == 0, report
print(report["environment"], "PASS", report["final_angle_deg"], "degrees")
PY
done
echo "Three fixed-pose qualification demos complete: $RUN_ROOT"
