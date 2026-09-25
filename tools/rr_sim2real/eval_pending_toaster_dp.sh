#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
cd /home/ubuntu/code/IsaacLab-Arena-tasks
DP_PY=/home/ubuntu/miniconda3/envs/robodiff/bin/python
DP_ROOT=/home/ubuntu/code/diffusion_policy
TRAIN_ROOT="$DP_ROOT/data/outputs/rr_pending_toaster_bs128_e80_20260912_163644"
RUN_ROOT="$(mktemp -d "$PWD/outputs/toast_epoch79_eval_XXXXXXXX")"
SERVER_PID=''
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=''
    fi
}
trap cleanup EXIT
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
export ARENA_RR_DP_AUDIT=1
echo "EVALUATION_ROOT=$RUN_ROOT"
for CASE in miniworkflow_gptsol_press_toaster miniworkflow_astra_press_toaster; do
    CHECKPOINT="$TRAIN_ROOT/$CASE/checkpoints/epoch=0079-train_loss=0.001.ckpt"
    test -f "$CHECKPOINT"
    echo "START $CASE: $CHECKPOINT"
    "$DP_PY" -u "$DP_ROOT/serve_ur7e_drawer_policy.py" \
        --ckpt "$CHECKPOINT" --port 5758 --num-inference-steps 100 \
        > "$RUN_ROOT/$CASE.server.log" 2>&1 &
    SERVER_PID=$!
    # Confirm a live, responsive server before invoking the Experiment Runner.
    "$DP_PY" - "$SERVER_PID" <<'PY'
import os
import pickle
import sys
import time
import zmq
pid = int(sys.argv[1])
deadline = time.monotonic() + 120
context = zmq.Context()
while time.monotonic() < deadline:
    os.kill(pid, 0)
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect('tcp://127.0.0.1:5758')
    socket.send(pickle.dumps({'cmd': 'meta'}))
    if socket.poll(1000):
        meta = pickle.loads(socket.recv())
        assert meta['n_obs_steps'] == 2 and meta['n_action_steps'] == 8, meta
        print('SERVER_READY', meta, flush=True)
        socket.close()
        break
    socket.close()
else:
    raise RuntimeError('DP server did not become ready within 120 seconds')
context.term()
PY
    PYTHONPATH="$PWD:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab" \
        .venv/bin/python isaaclab_arena/evaluation/experiment_runner.py \
        --experiment_config isaaclab_arena_environments/experiment_configs/ur7e_miniworkflow_toast_dp_eval.yaml \
        --viz none --enable_cameras --output_base_dir "$RUN_ROOT/$CASE" \
        "shared.environment.type=ur7e_$CASE" \
        "shared.policy.audit_object=${CASE%_press_toaster}_toast" \
        > "$RUN_ROOT/$CASE.eval.log" 2>&1
    cleanup
    echo "FINISHED $CASE"
done
echo "ALL EVALUATIONS FINISHED: $RUN_ROOT"
