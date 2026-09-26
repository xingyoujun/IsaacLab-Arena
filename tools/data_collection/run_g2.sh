#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Host launcher; discover this checkout's container and run as the host user.
set -euo pipefail
ARENA_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mapfile -t ARENA_MATCHES < <(docker ps --filter "volume=$ARENA_REPO" --format '{{.Names}}')
[[ ${#ARENA_MATCHES[@]} == 1 ]] || { echo "Expected one running container for $ARENA_REPO" >&2; exit 1; }
ARENA_ISOLATED=false
if [[ "${1:-}" == --isolated ]]; then
    ARENA_ISOLATED=true
    shift
fi
printf -v ARENA_COMMAND '%q ' /isaac-sim/python.sh tools/data_collection/g2.py "$@"
ARENA_ENV='PYTHONPATH=/workspaces/isaaclab_arena:/workspaces/isaaclab_arena/submodules/IsaacLab/source/isaaclab PYTHONUNBUFFERED=1 GENIESIM_ASSETS_DIR=/missing_source_assets'
if $ARENA_ISOLATED; then
    ARENA_IMAGE=$(docker inspect "${ARENA_MATCHES[0]}" --format '{{.Config.Image}}')
    # Only the checkout (including its asset cache) is mounted: no /datasets, /tmp,
    # host home, source archives, prior asset wrappers, or external robot configurations.
    exec docker run --rm --gpus all --shm-size=8g \
        -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e NVIDIA_DRIVER_CAPABILITIES=all \
        --mount "type=bind,source=$ARENA_REPO,target=/workspaces/isaaclab_arena" \
        --entrypoint /bin/bash "$ARENA_IMAGE" -c \
        'usermod -aG isaac-sim "$1" && exec su "$1" -c "$2"' _ "$(id -un)" \
        "test ! -e /datasets/GenieSimAssets && test ! -e /tmp/Assets && cd /workspaces/isaaclab_arena && exec env $ARENA_ENV $ARENA_COMMAND"
fi
exec docker exec -i "${ARENA_MATCHES[0]}" su "$(id -un)" -c \
    "cd /workspaces/isaaclab_arena && exec env $ARENA_ENV $ARENA_COMMAND"
