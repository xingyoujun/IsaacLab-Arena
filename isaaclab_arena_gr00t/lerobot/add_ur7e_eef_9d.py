# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Add end-effector pose modalities (``observation.eef_9d`` / ``action.eef_9d``) to a UR7e LeRobot dataset.

Both are computed by forward kinematics of the URDF the cuMotion driver plans with, so the state
(from the recorded joint positions) and the action (from the recorded joint targets) live in one
consistent frame: the UR ``base`` frame, which is what the real controller reports TCP poses in.
The representation follows the Agibot ``*_with_ee_pose`` datasets: ``xyz`` followed by the first two
COLUMNS of the rotation matrix (``rot6d_c0``, ``rot6d_c1``). Consumers that want row-major 6D
rotations (GR00T, pytorch3d / diffusion_policy) transpose offline.

The FK is validated against the simulator's own end-effector frame recorded in the HDF5 (world frame),
which catches a wrong URDF, a wrong base pose or a quaternion-order slip before anything is written.

Usage::

    .venv/bin/python isaaclab_arena_gr00t/lerobot/add_ur7e_eef_9d.py \\
        --hdf5 /path/open_drawer.hdf5 --lerobot /path/open_drawer/lerobot [--urdf <path>] [--max-fk-error-mm 3]
"""

from __future__ import annotations

import argparse
import h5py
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
DEFAULT_URDF = Path(__file__).resolve().parents[2] / "isaaclab_arena/embodiments/ur7e/rmpflow/ur7e_robotiq.urdf"
EEF_FRAME = "base"
"""UR's controller base frame (URDF link ``base``): the frame RTDE reports ``ActualTCPPose`` in."""
TCP_LINK = "tcp"
"""tool0 shifted 0.1628 m along its z: the Robotiq 2F-85 tool centre point, see ur7e_robotiq.urdf."""
EEF_NAMES = ["x", "y", "z", "rot6d_c0_x", "rot6d_c0_y", "rot6d_c0_z", "rot6d_c1_x", "rot6d_c1_y", "rot6d_c1_z"]


# ----------------------------------------------------------------------------------- kinematics ---
def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class UrdfChain:
    """Forward kinematics for a serial chain read from a URDF (fixed and revolute joints only)."""

    def __init__(self, urdf_path: Path):
        root = ET.parse(urdf_path).getroot()
        self.joints: dict[str, dict] = {}
        self.parent_joint: dict[str, str] = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()])
            axis_el = joint.find("axis")
            axis = np.array([float(v) for v in (axis_el.get("xyz") if axis_el is not None else "0 0 1").split()])
            child = joint.find("child").get("link")
            self.joints[joint.get("name")] = {
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": child,
                "R": _rpy_matrix(*rpy),
                "t": xyz,
                "axis": axis,
            }
            self.parent_joint[child] = joint.get("name")

    def path(self, link: str) -> list[str]:
        """Joint names from the URDF root down to ``link``."""
        chain = []
        while link in self.parent_joint:
            joint_name = self.parent_joint[link]
            chain.append(joint_name)
            link = self.joints[joint_name]["parent"]
        return chain[::-1]

    def pose(self, frame: str, link: str, q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Pose of ``link`` expressed in ``frame`` (any two links of the tree): (rotation matrix, translation)."""
        R_wf, t_wf = self._pose_in_root(frame, q)
        R_wl, t_wl = self._pose_in_root(link, q)
        return R_wf.T @ R_wl, R_wf.T @ (t_wl - t_wf)

    def _pose_in_root(self, link: str, q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        R, t = np.eye(3), np.zeros(3)
        for joint_name in self.path(link):
            joint = self.joints[joint_name]
            t = t + R @ joint["t"]
            R = R @ joint["R"]
            if joint["type"] in ("revolute", "continuous"):
                R = R @ _axis_angle_matrix(joint["axis"], q[joint_name])
            elif joint["type"] != "fixed":
                raise ValueError(f"unsupported joint type {joint['type']} on {joint_name}")
        return R, t


def eef_9d(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([t, R[:, 0], R[:, 1]]).astype(np.float32)


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------------------- main ---
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5", type=Path, required=True, help="Recorded HDF5 the LeRobot dataset was converted from."
    )
    parser.add_argument("--lerobot", type=Path, required=True, help="LeRobot dataset root (contains data/, meta/).")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--state-key", default="robot_joint_pos", help="HDF5 obs key with the articulation joints.")
    parser.add_argument("--action-key", default="joint_pos_target")
    parser.add_argument(
        "--max-fk-error-mm",
        type=float,
        default=5.0,
        help="Allowed FK vs recorded TCP position error (calibrated URDF vs nominal USD sits at ~3 mm).",
    )
    args = parser.parse_args()

    chain = UrdfChain(args.urdf)
    info_path = args.lerobot / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    reports = []
    with h5py.File(args.hdf5, "r", locking=False) as handle:
        # convert_hdf5_to_lerobot.py numbers episodes in the HDF5's own (alphabetical) key order,
        # so demo_10 becomes episode 2; the joint-state check below guards the pairing.
        demos = list(handle["data"].keys())
        for episode, demo_name in enumerate(demos):
            demo = handle["data"][demo_name]
            parquet = args.lerobot / info["data_path"].format(
                episode_chunk=episode // info["chunks_size"], episode_index=episode
            )
            df = pd.read_parquet(parquet)
            obs_joints = np.array(demo["obs"][args.state_key])[:-1]  # the converter drops the last row
            targets = np.array(demo[args.action_key])[:-1]
            assert (
                len(df) == len(obs_joints) == len(targets)
            ), f"{demo_name}: {len(df)} rows vs {len(obs_joints)} states"
            # The LeRobot episode must be this demo: compare the arm joints.
            state = np.stack(df["observation.state"].to_numpy())
            assert np.allclose(state[:, :6], obs_joints[:, :6], atol=1e-5), f"{demo_name} is not episode {episode}"

            eef_state = np.zeros((len(df), 9), np.float32)
            eef_action = np.zeros((len(df), 9), np.float32)
            fk_world_error = []
            root_pose = np.array(demo["states"]["articulation"]["robot"]["root_pose"])[:-1]  # xyzw quaternion
            eef_pos_rec = np.array(demo["obs"]["eef_pos"])[:-1]
            for i in range(len(df)):
                q_state = dict(zip(ARM_JOINTS, obs_joints[i, :6]))
                q_target = dict(zip(ARM_JOINTS, targets[i, :6]))
                R, t = chain.pose(EEF_FRAME, TCP_LINK, q_state)
                eef_state[i] = eef_9d(R, t)
                Ra, ta = chain.pose(EEF_FRAME, TCP_LINK, q_target)
                eef_action[i] = eef_9d(Ra, ta)
                # Validation: FK in the world (URDF 'world' link == simulated robot root) vs the recorded TCP.
                Rw, tw = chain.pose("world", TCP_LINK, q_state)
                world_t = root_pose[i, :3] + quat_xyzw_to_matrix(root_pose[i, 3:7]) @ tw
                fk_world_error.append(np.linalg.norm(world_t - eef_pos_rec[i]))
            err_mm = float(np.max(fk_world_error)) * 1000
            assert err_mm < args.max_fk_error_mm, (
                f"{demo_name}: FK disagrees with the recorded TCP by up to {err_mm:.1f} mm; check URDF, root pose"
                " and quaternion order"
            )
            df["observation.eef_9d"] = list(eef_state)
            df["action.eef_9d"] = list(eef_action)
            df.to_parquet(parquet, index=False)
            reports.append(
                {"episode": episode, "demo": demo_name, "rows": len(df), "fk_vs_recorded_max_mm": round(err_mm, 3)}
            )
            print(f"episode {episode} ({demo_name}): {len(df)} rows, FK vs recorded TCP max {err_mm:.2f} mm")

    for key in ("observation.eef_9d", "action.eef_9d"):
        info["features"][key] = {"dtype": "float32", "shape": [9], "names": EEF_NAMES}
    info_path.write_text(json.dumps(info, indent=4))

    modality_path = args.lerobot / "meta" / "modality.json"
    modality = json.loads(modality_path.read_text())
    modality["state"]["eef_9d"] = {"original_key": "observation.eef_9d", "start": 0, "end": 9}
    modality["action"]["eef_9d"] = {"original_key": "action.eef_9d", "start": 0, "end": 9}
    modality_path.write_text(json.dumps(modality, indent=4))

    (args.lerobot / "meta" / "eef_9d.json").write_text(
        json.dumps(
            {
                "representation": "eef_9d = xyz (m) + first two COLUMNS of the rotation matrix (rot6d_c0, rot6d_c1)",
                "row_major_note": (
                    "GR00T / pytorch3d rotation_6d use the first two ROWS: transpose the 3x2 block offline"
                ),
                "coordinate_frame": (
                    f"UR '{EEF_FRAME}' frame (URDF link '{EEF_FRAME}'; the frame RTDE ActualTCPPose uses)"
                ),
                "tcp": f"URDF link '{TCP_LINK}' = tool0 shifted 0.1628 m along its z (Robotiq 2F-85 tool centre point)",
                "observation_source": f"FK of obs/{args.state_key} (measured joint positions)",
                "action_source": f"FK of {args.action_key} (absolute joint targets applied by the drives)",
                "gripper": (
                    "not part of eef_9d; observation.state[6] / action[6] carry finger_joint (rad, 0 open, pi/4 closed)"
                ),
                "urdf": str(args.urdf),
                "validation": {
                    "method": "FK in the world frame vs the recorded ee_frame TCP position",
                    "episodes": reports,
                },
            },
            indent=4,
        )
    )
    print(f"added observation.eef_9d / action.eef_9d to {len(reports)} episodes in {args.lerobot}")


if __name__ == "__main__":
    main()
