#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Publish completed datasets, replacing loose remote Zarr files with tar archives.
# Run on the source host. Does not modify local training datasets or stop collection.
set -euo pipefail

REPO=xingyoujun/rr_real2sim_data
DATA_ROOT=/home/ubuntu/playground/datasets/rr_sim2real
WORK_ROOT=/home/ubuntu/playground/datasets/rr_sim2real_aux/black_gripper_all
ARCHIVE_ROOT=/home/ubuntu/playground/datasets/rr_sim2real_aux/hf_zarr_archives

if [[ $# -ne 0 ]]; then
  echo "Usage: bash $0"
  echo "Uploads completed cases to $REPO and removes their loose remote Zarr layout."
  exit 2
fi

for cmd in hf tar sha256sum flock grep df awk; do
  command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 1; }
done
mkdir -p "$ARCHIVE_ROOT"
exec 9>"$ARCHIVE_ROOT/sync.lock"
flock -n 9 || { echo 'Another dataset upload script is running.' >&2; exit 1; }
trap 'echo "Stopped at line $LINENO. Local source datasets are intact. If HTTP 429 occurred, wait for the indicated reset time, then rerun this script." >&2' ERR

CASES=(
  usdcraft_open_drawer
  usdcraft_press_toaster
  articraft_open_drawer
  articraft_press_toaster
  miniworkflow_gptsol_open_drawer
  miniworkflow_gptsol_press_toaster
  miniworkflow_astra_open_drawer
  miniworkflow_astra_press_toaster
)
READY=()
DELETE_PATTERNS=()

for CASE in "${CASES[@]}"; do
  # Snapshot only already-published cases; newly completed ones join the next run.
  if [[ ! -f "$WORK_ROOT/$CASE/complete.json" ]] ||
     ! grep -Eq '"published"[[:space:]]*:[[:space:]]*true' "$WORK_ROOT/$CASE/complete.json" ||
     [[ ! -f "$DATA_ROOT/${CASE}_dp.zarr/rr_sim2real_build.json" ]] ||
     [[ ! -f "$DATA_ROOT/$CASE/meta/info.json" ]]; then
    echo "Skip unfinished case: $CASE"
    continue
  fi
  READY+=("$CASE")
done

if (( ${#READY[@]} == 0 )); then
  echo 'No completed datasets. Nothing uploaded or deleted.'
  exit 0
fi

echo "Destination: https://huggingface.co/datasets/$REPO"
echo "Completed cases: ${READY[*]}"
echo 'Archives will be uploaded BEFORE removing loose remote Zarr directories.'
echo 'Do not run a second uploader or replace these published datasets during this run.'

for CASE in "${READY[@]}"; do
  ARCHIVE="$ARCHIVE_ROOT/${CASE}_dp.zarr.tar"

  # Compare content and metadata against the source before reusing an archive.
  if [[ -f "$ARCHIVE" ]] && tar --compare -f "$ARCHIVE" -C "$DATA_ROOT" >/dev/null 2>&1 &&
     [[ -f "$ARCHIVE.sha256" ]] &&
     (cd "$ARCHIVE_ROOT" && sha256sum --status -c "${CASE}_dp.zarr.tar.sha256"); then
    echo "Reuse validated archive: $ARCHIVE"
  else
    FREE_KIB=$(df -Pk "$ARCHIVE_ROOT" | awk 'END {print $4}')
    if (( FREE_KIB < 40 * 1024 * 1024 )); then
      echo 'Less than 40 GiB free; stop before creating another archive.' >&2
      exit 1
    fi
    echo "Pack and compare: $CASE"
    tar --exclude='*/.cache/huggingface' -cf "$ARCHIVE.part" \
      -C "$DATA_ROOT" "${CASE}_dp.zarr"
    tar --compare -f "$ARCHIVE.part" -C "$DATA_ROOT"
    mv "$ARCHIVE.part" "$ARCHIVE"
    (cd "$ARCHIVE_ROOT" && sha256sum "${CASE}_dp.zarr.tar" > "${CASE}_dp.zarr.tar.sha256")
  fi

  echo "Upload archive: $CASE"
  hf upload "$REPO" "$ARCHIVE" "${CASE}_dp.zarr.tar" --repo-type dataset
  hf upload "$REPO" "$ARCHIVE.sha256" "${CASE}_dp.zarr.tar.sha256" --repo-type dataset
done

# Only exact known Zarr directory prefixes; never matches .tar or LeRobot folders.
# Include unfinished cases to remove any partial loose uploads from the old uploader.
for CASE in "${CASES[@]}"; do
  DELETE_PATTERNS+=("${CASE}_dp.zarr/*")
done
echo 'Remove loose remote Zarr files (ordinary commit; history is retained):'
printf '  %s\n' "${DELETE_PATTERNS[@]}"
hf repo-files delete "$REPO" "${DELETE_PATTERNS[@]}" --repo-type dataset \
  --commit-message 'Replace loose Zarr directories with validated tar archives'

for CASE in "${READY[@]}"; do
  echo "Upload LeRobot: $CASE"
  hf upload "$REPO" "$DATA_ROOT/$CASE" "$CASE" --repo-type dataset
done
if [[ -f "$DATA_ROOT/README.md" ]]; then
  hf upload "$REPO" "$DATA_ROOT/README.md" README.md --repo-type dataset
fi

echo "Done: ${#READY[@]} cases. Local Zarr directories and archives retained."
echo "Archives: $ARCHIVE_ROOT"
echo 'Download CASE_dp.zarr.tar plus its .sha256, verify with sha256sum -c, then tar -xf.'
echo 'Run this script again after the remaining collection cases finish.'
