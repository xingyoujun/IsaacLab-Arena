# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve a trained UR7e open-drawer diffusion policy over ZMQ for Isaac Lab-Arena evaluation.

Run in the ``robodiff`` conda environment::

    python serve_ur7e_drawer_policy.py --ckpt data/outputs/.../checkpoints/latest.ckpt --port 5758

Protocol (pickle over ZMQ REQ/REP; arrays travel as dtype/shape/bytes so numpy 1.x and 2.x interoperate): the client sends ``{"cmd": "meta"}``, ``{"cmd": "reset"}`` or
``{"cmd": "act", "camera_0": uint8 (T,H,W,3), "robot_eef_pose": float32 (T,9), "gripper_pos": float32 (T,1)}``
with T = n_obs_steps frames, oldest first, at any input resolution; frames are resized to the
policy's shape_meta with PIL bilinear (the same resampling the training zarr was built with).
The reply to "act" is ``{"action": float32 (n_action_steps, 10)}``: absolute TCP xyz + rotation_6d
(rows) in the UR base frame, then the gripper value (finger_joint / (pi/4)).
"""

import argparse
import numpy as np
import pathlib
import pickle
import sys
import time
import torch

import dill
import hydra
import zmq
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def encode_arrays(message: dict) -> dict:
    """Replace numpy arrays by (dtype, shape, bytes) triples so the pickle is numpy-version agnostic."""
    out = {}
    for key, value in message.items():
        if isinstance(value, np.ndarray):
            out[key] = {"__nd__": True, "dtype": str(value.dtype), "shape": list(value.shape), "data": value.tobytes()}
        else:
            out[key] = value
    return out


def decode_arrays(message: dict) -> dict:
    """Inverse of ``encode_arrays``."""
    out = {}
    for key, value in message.items():
        if isinstance(value, dict) and value.get("__nd__"):
            out[key] = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()
        else:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--port", type=int, default=5758)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=16, help="DDIM steps (100 in training).")
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(args.device).eval()
    policy.num_inference_steps = args.num_inference_steps
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    n_obs_steps = int(cfg.n_obs_steps)
    print(f"loaded {args.ckpt}: epoch {payload['pickles'].get('epoch')}, step {payload['pickles'].get('global_step')}")
    print(f"shape_meta {shape_meta}; n_obs_steps {n_obs_steps}; n_action_steps {policy.n_action_steps}")

    def prepare(obs):
        out = {}
        for key, attr in shape_meta["obs"].items():
            if attr.get("type") == "rgb":
                c, h, w = attr["shape"]
                frames = obs[key]
                assert frames.ndim == 4 and frames.shape[-1] == c, f"{key}: got {frames.shape}"
                resized = np.stack(
                    [np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR), dtype=np.uint8) for f in frames]
                )
                out[key] = torch.from_numpy(np.moveaxis(resized, -1, 1).astype(np.float32) / 255.0)
            else:
                out[key] = torch.from_numpy(np.asarray(obs[key], dtype=np.float32))
            assert out[key].shape[0] == n_obs_steps, f"{key}: expected {n_obs_steps} frames, got {out[key].shape[0]}"
        return {k: v.unsqueeze(0).to(args.device) for k, v in out.items()}

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"serving on tcp://*:{args.port} (ready)", flush=True)
    while True:
        request = decode_arrays(pickle.loads(socket.recv()))
        cmd = request.get("cmd")
        if cmd == "meta":
            reply = {"shape_meta": shape_meta, "n_obs_steps": n_obs_steps, "n_action_steps": int(policy.n_action_steps)}
        elif cmd == "reset":
            policy.reset()
            reply = {"ok": True}
        elif cmd == "act":
            t0 = time.time()
            with torch.no_grad():
                result = policy.predict_action(prepare(request))
            action = result["action"][0].detach().cpu().numpy().astype(np.float32)
            reply = {"action": action, "inference_s": time.time() - t0}
        else:
            reply = {"error": f"unknown cmd {cmd}"}
        socket.send(pickle.dumps(encode_arrays(reply)))


if __name__ == "__main__":
    main()
