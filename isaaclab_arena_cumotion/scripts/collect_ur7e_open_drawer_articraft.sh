#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Two-worker collection through opening, then randomized rendering and conversion.
# Relaunch with the same settings to resume. Defaults to 200 successful episodes.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
exec .venv/bin/python isaaclab_arena_cumotion/scripts/collect_ur7e_open_drawer_articraft.py \
    --target "${TARGET_SUCCESS:-200}" --batch-size "${DEMOS_PER_WORKER:-8}" \
    --seed "${SEED_BASE:-0}" --randomize-seed "${RANDOMIZE_SEED:-0}" \
    --max-rounds "${MAX_ROUNDS:-40}" "$@"
