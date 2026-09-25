#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Upload only the four completed knob datasets; never delete remote files.
set -euo pipefail

HF=/home/ubuntu/.local/bin/hf
REPO=xingyoujun/rr_real2sim_data
DATA_ROOT=/home/ubuntu/playground/datasets/rr_sim2real
WORK_ROOT=/home/ubuntu/playground/datasets/rr_sim2real_aux/toaster_knob_all
ARCHIVE_ROOT=/home/ubuntu/playground/datasets/rr_sim2real_aux/hf_zarr_archives

[[ $# -eq 0 ]] || { echo "Usage: bash $0" >&2; exit 2; }
[[ -x "$HF" ]] || { echo "Missing HF CLI: $HF" >&2; exit 1; }
for cmd in tar sha256sum flock grep df awk; do
    command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 1; }
done

CASES=(
    usdcraft_turn_toaster_knob
    articraft_turn_toaster_knob
    miniworkflow_gptsol_turn_toaster_knob
    miniworkflow_astra_turn_toaster_knob
)

# Validate every case before starting uploads.
for CASE in "${CASES[@]}"; do
    [[ -f "$WORK_ROOT/$CASE/complete.json" ]] &&
        grep -Eq '"published"[[:space:]]*:[[:space:]]*true' "$WORK_ROOT/$CASE/complete.json" &&
        [[ -f "$DATA_ROOT/$CASE/meta/info.json" ]] &&
        [[ -f "$DATA_ROOT/$CASE/rr_sim2real_build.json" ]] &&
        [[ -f "$DATA_ROOT/${CASE}_dp.zarr/rr_sim2real_build.json" ]] || {
            echo "Missing or unfinished dataset: $CASE" >&2
            exit 1
        }
done

mkdir -p "$ARCHIVE_ROOT"
# Share the lock with the existing eight-case uploader.
exec 9>"$ARCHIVE_ROOT/sync.lock"
flock -n 9 || { echo "Another dataset uploader is running." >&2; exit 1; }
trap 'echo "Upload stopped at line $LINENO. Local datasets are intact. For HTTP 429, wait for the indicated reset time, then rerun this script." >&2' ERR

echo "Upload destination: https://huggingface.co/datasets/$REPO"
echo "Do not modify these datasets or start another uploader during this run."

for CASE in "${CASES[@]}"; do
    NAME="${CASE}_dp.zarr.tar"
    ARCHIVE="$ARCHIVE_ROOT/$NAME"
    if [[ -f "$ARCHIVE" && -f "$ARCHIVE.sha256" ]] &&
        tar --compare -f "$ARCHIVE" -C "$DATA_ROOT" >/dev/null 2>&1 &&
        (cd "$ARCHIVE_ROOT" && sha256sum --status -c "$NAME.sha256"); then
        echo "Reuse verified archive: $NAME"
    else
        FREE_KIB=$(df -Pk "$ARCHIVE_ROOT" | awk 'END {print $4}')
        if (( FREE_KIB < 40 * 1024 * 1024 )); then
            echo "Less than 40 GiB free; stop before creating another archive." >&2
            exit 1
        fi
        echo "Pack and verify: $CASE"
        tar --exclude='*/.cache/huggingface' -cf "$ARCHIVE.part" \
            -C "$DATA_ROOT" "${CASE}_dp.zarr"
        tar --compare -f "$ARCHIVE.part" -C "$DATA_ROOT"
        mv "$ARCHIVE.part" "$ARCHIVE"
        (cd "$ARCHIVE_ROOT" && sha256sum "$NAME" > "$NAME.sha256")
    fi

    "$HF" upload "$REPO" "$ARCHIVE" "$NAME" --repo-type dataset
    "$HF" upload "$REPO" "$ARCHIVE.sha256" "$NAME.sha256" --repo-type dataset
    "$HF" upload "$REPO" "$DATA_ROOT/$CASE" "$CASE" --repo-type dataset
done

if [[ -f "$DATA_ROOT/README.md" ]]; then
    "$HF" upload "$REPO" "$DATA_ROOT/README.md" README.md --repo-type dataset
fi
echo "Done: all four knob datasets uploaded. No remote files deleted."
echo "Local datasets and archives retained: $ARCHIVE_ROOT"
