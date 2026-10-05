#!/bin/bash
# Rebuild the test_assets_v0 manifest and render any asset without renders (8091 /reviews/asset-baselines).
set -e
cd "$(dirname "$0")/../.."
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH=/home/ubuntu/code/IsaacLab-Arena-tasks:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab
.venv/bin/python tools/asset_baselines/build_test_assets.py 2>&1 | grep -v -i "warning"
.venv/bin/python tools/asset_baselines/render_assets.py > outputs/asset_baselines/render_last.log 2>&1
grep "\[asset\]" outputs/asset_baselines/render_last.log || echo "no new assets to render"
