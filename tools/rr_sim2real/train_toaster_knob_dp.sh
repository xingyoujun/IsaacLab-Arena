#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Source-host sequential training; reuse the pinned press task's identical data shapes.
set -Eeuo pipefail
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
CHECK_ONLY=false
if [[ ${1:-} == --check-only && $# -eq 1 ]]; then
    CHECK_ONLY=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash $0 [--check-only]" >&2
    exit 2
fi

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/code/diffusion_policy}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/playground/datasets/rr_sim2real}"
CONDA_BASE="${CONDA_BASE:-/home/ubuntu/miniconda3}"
CONDA_ENV="${CONDA_ENV:-robodiff}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/data/outputs/rr_toaster_knob_bs128_e80_${RUN_TAG}}"

CASES=(
    usdcraft_turn_toaster_knob
    articraft_turn_toaster_knob
    miniworkflow_gptsol_turn_toaster_knob
    miniworkflow_astra_turn_toaster_knob
)

# Conda activation/deactivation hooks may read unset variables. Keep nounset
# disabled only during environment setup, including repeated activation.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
cd "$REPO_ROOT"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

python -m pip check
python - <<'PY'
import hashlib
from pathlib import Path
expected = {
    "diffusion_policy/workspace/train_diffusion_unet_image_workspace.py": "38cf15505ff723542fe4ae5db31a1717985e78de8c5aa4f02329c8e99cbf061a",
    "diffusion_policy/dataset/ur7e_drawer_image_dataset.py": "4a1d96eae54e4aa3184201b906c76801497923ed5a440971916193bbc7f42f37",
    "diffusion_policy/config/task/ur7e_press_toaster_image.yaml": "2658570cf74774e67ad8fba1a9892ba84e5f4cf40790dc7fed7e14809bb93a81",
    "diffusion_policy/config/train_diffusion_unet_real_image_workspace.yaml": "d371f2219d90c165b0bd119e9aa3399d97a20166ced88087c823979bf0016b6d",
}
for name, digest in expected.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
print("Training code matches remote handoff")
PY
for CASE in "${CASES[@]}"; do
    ZARR_PATH="$DATA_ROOT/${CASE}_dp.zarr"
    python - "$ZARR_PATH" "$CASE" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()
path = Path(sys.argv[1])
case = sys.argv[2]
build = json.loads((path / "rr_sim2real_build.json").read_text())
assert build["case"] == case
work = Path("/home/ubuntu/playground/datasets/rr_sim2real_aux/toaster_knob_all") / case
complete = json.loads((work / "complete.json").read_text())
publication = json.loads((work / "publication.json").read_text())
lerobot_build = json.loads((path.parent / case / "rr_sim2real_build.json").read_text())
assert complete["published"] and complete["episodes"] == 200
assert publication["done"]
assert publication["id"] == build["build_id"] == lerobot_build["build_id"]
assert build["episodes"] == 200
assert build["appearance"]["gripper"] == "black_fingertips_direct"
assert build["randomize"] is True
root = zarr.open_group(str(path), mode="r")
ends = root["meta/episode_ends"][:]
assert len(ends) == 200 and np.all(np.diff(np.r_[0, ends]) > 0)
n = int(ends[-1])
assert build["training_frames"] == n
expected = {
    "camera_0": (n, 240, 320, 3),
    "robot_eef_pose": (n, 9),
    "gripper_pos": (n, 1),
    "action": (n, 10),
}
for key, shape in expected.items():
    assert root[f"data/{key}"].shape == shape, key
assert root["data/camera_0"].dtype == np.dtype("uint8")
for i in (0, n // 2, n - 1):
    assert root["data/camera_0"][i].shape == (240, 320, 3)
for key in ("action", "robot_eef_pose", "gripper_pos"):
    assert np.isfinite(root[f"data/{key}"][:]).all(), key
print("READY", case, "episodes=", len(ends), "frames=", n, "build_id=", build["build_id"])
PY

done

if "$CHECK_ONLY"; then
    echo "All four local datasets and pinned training code verified. No training started."
    exit 0
fi

python -c 'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))'
FREE_KIB=$(df -Pk "$REPO_ROOT/data/outputs" | awk 'END {print $4}')
if (( FREE_KIB < 65 * 1024 * 1024 )); then
    echo "Need at least 65 GiB free for four runs and checkpoint headroom." >&2
    exit 1
fi
# Refuse occupied GPUs instead of silently changing the agreed batch size.
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU compute process detected. Stop evaluation/inference before training." >&2
    exit 1
fi
# Fresh output directory; never overwrite an earlier training run.
mkdir "$RUN_ROOT"
python -m pip freeze > "$RUN_ROOT/environment.txt"
cp "$SCRIPT_PATH" "$RUN_ROOT/training_script.sh"
git rev-parse HEAD > "$RUN_ROOT/code_commit.txt"
git diff -- diffusion_policy/workspace/train_diffusion_unet_image_workspace.py > "$RUN_ROOT/workspace.patch"


for CASE in "${CASES[@]}"; do
    ZARR_PATH="$DATA_ROOT/${CASE}_dp.zarr"
    cp "$ZARR_PATH/rr_sim2real_build.json" "$RUN_ROOT/${CASE}.build.json"
    CASE_RUN_DIR="$RUN_ROOT/$CASE"
    mkdir -p "$CASE_RUN_DIR"

    echo "TRAINING $CASE -> $CASE_RUN_DIR"
    python train.py \
        --config-name=train_diffusion_unet_real_image_workspace \
        task=ur7e_press_toaster_image \
        task.name="${CASE}_image" \
        task.dataset_path="$ZARR_PATH" \
        logging.project=rr_sim2real \
        logging.name="${CASE}_black_bs128_e80" \
        logging.mode=online \
        training.resume=false \
        training.seed=42 \
        training.num_epochs=80 \
        training.checkpoint_every=40 \
        training.rollout_every=1000 \
        training.val_every=10 \
        training.sample_every=20 \
        dataloader.batch_size=128 \
        dataloader.num_workers=4 \
        val_dataloader.batch_size=128 \
        val_dataloader.num_workers=4 \
        checkpoint.topk.k=2 \
        hydra.run.dir="$CASE_RUN_DIR" \
        2>&1 | tee "$RUN_ROOT/${CASE}.console.log"
    compgen -G "$CASE_RUN_DIR/checkpoints/epoch=0079-*.ckpt" >/dev/null || {
        echo "Missing final epoch-79 checkpoint for $CASE; stopping." >&2
        exit 1
    }
    touch "$CASE_RUN_DIR/TRAINING_COMPLETE"
done
echo "All four knob trainings complete: $RUN_ROOT"
