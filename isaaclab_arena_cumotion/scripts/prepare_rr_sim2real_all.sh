#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
RR_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RR_REPO"
exec "$RR_REPO/.venv/bin/python" -u "$RR_REPO/isaaclab_arena_cumotion/scripts/prepare_rr_sim2real_all.py" "$@"
