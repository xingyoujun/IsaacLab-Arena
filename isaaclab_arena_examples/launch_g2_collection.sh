#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Run on the host (e.g. in tmux); all simulation/data work runs as the host user in its cuRobo container.
set -euo pipefail
ARENA_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${ARENA_CONTAINER:-}" ]]; then
    mapfile -t ARENA_MATCHES < <(docker ps --filter "volume=$ARENA_REPO" --filter ancestor=isaaclab_arena:curobo --format '{{.Names}}')
    if [[ ${#ARENA_MATCHES[@]} != 1 ]]; then
        echo "Expected one running cuRobo container mounting $ARENA_REPO; found ${#ARENA_MATCHES[@]}." >&2
        echo "Start the workspace cuRobo container or explicitly set ARENA_CONTAINER." >&2
        exit 1
    fi
    ARENA_CONTAINER=${ARENA_MATCHES[0]}
fi
ARENA_MOUNT=$(docker inspect "$ARENA_CONTAINER" --format '{{range .Mounts}}{{if eq .Destination "/workspaces/isaaclab_arena"}}{{.Source}}{{end}}{{end}}')
[[ "$ARENA_MOUNT" == "$ARENA_REPO" ]] || { echo "Container mounts a different workspace" >&2; exit 1; }
ARENA_TTY=(-i)
if [[ -t 0 && -t 1 ]]; then
    ARENA_TTY+=(-t)
fi
printf -v ARENA_COMMAND '%q ' /isaac-sim/python.sh isaaclab_arena_examples/g2_collect_dataset.py \
    --robot_yaml /datasets/agibot_dataset_v1_raw/assets/g2/robot.yaml \
    --robot_urdf /datasets/agibot_dataset_v1_raw/assets/g2/robot.urdf "$@"
exec docker exec "${ARENA_TTY[@]}" "$ARENA_CONTAINER" su "$(id -un)" -c \
    "cd /workspaces/isaaclab_arena && exec env PYTHONUNBUFFERED=1 $ARENA_COMMAND"
