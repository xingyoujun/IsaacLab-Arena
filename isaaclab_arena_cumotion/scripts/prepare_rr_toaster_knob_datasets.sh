#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
RR_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RR_REPO"
exec "$RR_REPO/.venv/bin/python" -u isaaclab_arena_cumotion/scripts/prepare_rr_sim2real_all.py \
    --work-root /home/ubuntu/playground/datasets/rr_sim2real_aux/toaster_knob_all \
    --final-root /home/ubuntu/playground/datasets/rr_sim2real \
    --target 200 --workers 3 --render-workers 2 --batch-size 8 --stagger 20 \
    --only usdcraft_turn_toaster_knob articraft_turn_toaster_knob \
      miniworkflow_gptsol_turn_toaster_knob miniworkflow_astra_turn_toaster_knob "$@"
