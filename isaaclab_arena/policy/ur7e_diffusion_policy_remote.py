# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Closed-loop client for a diffusion_policy model trained on the UR7e open-drawer dataset.

The model runs in its own process (``diffusion_policy/serve_ur7e_drawer_policy.py``, robodiff
conda env) because its torch/diffusers versions are incompatible with the Arena environment. This
policy builds the observation the model was trained on -- the D435 frame, the tool centre point
pose in the UR ``base`` frame (xyz + rotation_6d rows) and the normalised gripper opening -- sends
the last ``n_obs_steps`` of them over ZMQ, and turns the returned absolute TCP poses into joint
targets with cuMotion's IK on the same URDF the training labels were computed from. It drives the
``ur7e_robotiq_joint_pos`` embodiment: six absolute joint targets plus a binary gripper.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pickle
import torch
from collections import deque
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

ROT_ROOT_BASE = np.diag([-1.0, -1.0, 1.0])
"""Rotation of the UR ``base`` frame in the URDF root (``world``/``base_link``) frame: yaw of 180 deg."""
TCP_FRAME = "tcp"
"""URDF frame the dataset's eef_9d refers to (tool0 + 0.1628 m)."""


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


@dataclass
class Ur7eDiffusionPolicyRemoteCfg(PolicyCfg):
    """Configure the remote diffusion-policy client."""

    host: str = "127.0.0.1"
    port: int = 5758
    n_action_steps: int | None = None
    """How many of the returned actions to execute before re-planning; the server's default when None."""
    gripper_close_threshold: float = 0.35
    """Gripper values above this (finger_joint / (pi/4)) close the binary gripper; the knob grasp reads ~0.7."""
    max_ik_jump_rad: float = 0.5
    """Reject an IK solution that moves any joint more than this from the previous target (branch flip)."""
    camera_key: str = "realsense_d435_rgb"


@register_policy
class Ur7eDiffusionPolicyRemote(PolicyBase[Ur7eDiffusionPolicyRemoteCfg]):
    """Run a diffusion_policy checkpoint served over ZMQ on the UR7e open-drawer environment."""

    name = "ur7e_dp_remote"

    def __init__(self, config: Ur7eDiffusionPolicyRemoteCfg):
        super().__init__(config)
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{config.host}:{config.port}")
        meta = self._request({"cmd": "meta"})
        self.n_obs_steps = int(meta["n_obs_steps"])
        self.n_action_steps = int(config.n_action_steps or meta["n_action_steps"])
        print(
            f"[ur7e_dp_remote] server shape_meta {meta['shape_meta']}, obs steps {self.n_obs_steps}, action steps"
            f" {self.n_action_steps}"
        )
        self._history: deque = deque(maxlen=self.n_obs_steps)
        self._queue: list[np.ndarray] = []
        self._ik_ready = False
        self._last_q: np.ndarray | None = None
        self._steps = 0
        self._ik_failures = 0

    # ------------------------------------------------------------------------------ plumbing ---
    def _request(self, message: dict) -> dict:
        self._socket.send(pickle.dumps(encode_arrays(message)))
        reply = decode_arrays(pickle.loads(self._socket.recv()))
        assert "error" not in reply, reply["error"]
        return reply

    def _init_ik(self, env) -> None:
        from isaaclab.sim.utils.extensions import enable_extension

        enable_extension("isaacsim.robot_motion.cumotion")
        from isaaclab_arena_cumotion.embodiment_cumotion_registry import get_cumotion_cfg_by_name
        from isaaclab_arena_cumotion.robot_description import import_cumotion, load_robot_description

        self._cumotion = import_cumotion()
        self._cfg = get_cumotion_cfg_by_name("ur7e_robotiq")
        self._kinematics = load_robot_description(self._cfg).kinematics()
        robot = env.unwrapped.scene.articulations["robot"]
        self._arm_ids, _ = robot.find_joints(self._cfg.arm_joint_names, preserve_order=True)
        self._arm_ids = [
            int(i) for i in (self._arm_ids.torch.tolist() if hasattr(self._arm_ids, "torch") else self._arm_ids)
        ]
        self._finger_id = int(robot.find_joints(self._cfg.gripper_joint_names)[0][0])
        self._robot = robot
        self._ik_ready = True

    def _arm_q(self) -> np.ndarray:
        import warp as wp

        return wp.to_torch(self._robot.data.joint_pos)[0, self._arm_ids].detach().cpu().numpy().astype(np.float64)

    def _tcp_pose_base(self, q: np.ndarray) -> np.ndarray:
        """TCP pose in the UR base frame as eef_9d (xyz + rotation_6d rows)."""
        pose = self._kinematics.pose(q, TCP_FRAME)
        t_root = np.asarray(pose.translation, dtype=np.float64)
        r_root = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        t_base = ROT_ROOT_BASE.T @ t_root
        r_base = ROT_ROOT_BASE.T @ r_root
        return np.concatenate([t_base, r_base[0], r_base[1]]).astype(np.float32)

    @staticmethod
    def _matrix_from_rot6d_rows(rot6d: np.ndarray) -> np.ndarray:
        r0 = rot6d[0:3] / np.linalg.norm(rot6d[0:3])
        r1 = rot6d[3:6] - np.dot(rot6d[3:6], r0) * r0
        r1 /= np.linalg.norm(r1)
        return np.stack([r0, r1, np.cross(r0, r1)], axis=0)

    def _solve_ik(self, action: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        """Joint targets putting the TCP at an action's base-frame pose, or None."""
        t_root = ROT_ROOT_BASE @ action[0:3].astype(np.float64)
        r_root = ROT_ROOT_BASE @ self._matrix_from_rot6d_rows(action[3:9].astype(np.float64))
        transform = np.eye(4)
        transform[:3, :3] = r_root
        transform[:3, 3] = t_root
        ik_cfg = self._cumotion.IkConfig()
        ik_cfg.cspace_seeds = [np.asarray(seed, dtype=np.float64)]
        result = self._cumotion.solve_ik(self._kinematics, self._cumotion.Pose3(transform), TCP_FRAME, ik_cfg)
        if not result.success:
            return None
        for attr in ("cspace_position", "cspace_positions", "q", "joint_positions", "solution"):
            if hasattr(result, attr):
                return np.asarray(getattr(result, attr), dtype=np.float64).reshape(-1)
        raise AttributeError(f"cuMotion IkResult exposes no known solution attribute: {dir(result)}")

    # -------------------------------------------------------------------------------- policy ---
    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        assert env.unwrapped.num_envs == 1, "ur7e_dp_remote drives a single environment"
        if not self._ik_ready:
            self._init_ik(env)
        q = self._arm_q()
        if self._last_q is None:
            self._last_q = q.copy()

        frame = observation["camera_obs"][self.config.camera_key][0].detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        gripper = observation["policy"]["gripper_pos"][0].detach().cpu().numpy().astype(np.float32).reshape(1)
        self._history.append(
            {"camera_0": frame[..., :3], "robot_eef_pose": self._tcp_pose_base(q), "gripper_pos": gripper}
        )
        while len(self._history) < self.n_obs_steps:
            self._history.appendleft(self._history[0])

        if not self._queue:
            batch = {key: np.stack([h[key] for h in self._history]) for key in self._history[0]}
            reply = self._request({"cmd": "act", **batch})
            self._queue = list(reply["action"][: self.n_action_steps])
        action = self._queue.pop(0)

        q_target = self._solve_ik(action, seed=self._last_q)
        if q_target is None or np.max(np.abs(q_target - self._last_q)) > self.config.max_ik_jump_rad:
            self._ik_failures += 1
            q_target = self._last_q  # hold
        self._last_q = q_target
        self._steps += 1
        gripper_cmd = 1.0 if float(action[9]) > self.config.gripper_close_threshold else 0.0
        vector = np.concatenate([q_target, [gripper_cmd]]).astype(np.float32)
        return torch.from_numpy(vector).unsqueeze(0).to(torch.device(env.unwrapped.device))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._steps:
            print(f"[ur7e_dp_remote] episode: {self._steps} steps, {self._ik_failures} IK holds")
        self._history.clear()
        self._queue = []
        self._last_q = None
        self._steps = 0
        self._ik_failures = 0
        self._request({"cmd": "reset"})

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()
