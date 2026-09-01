# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify and repair the ``eef_9d`` rotations of the ``*_with_ee_pose`` LeRobot datasets.

The ``*_with_ee_pose`` datasets carry per-arm end-effector poses (``observation.eef_9d`` /
``action.eef_9d``: xyz + the first two rotation-matrix columns, base_link frame). Their
positions are exact, but the rotations are scrambled by two quaternion-order bugs in the
pipeline that produced them:

1. **Both arms**: the final xyzw quaternion was converted to a rotation matrix by a consumer
   expecting wxyz. The components are cyclically shifted, which is NOT a fixed frame offset --
   the resulting rotation error varies frame by frame with the pose itself (>120 deg drift
   within one episode), so no static calibration can absorb it.
2. **Left arm only**: a -90 deg-about-Y offset quaternion, written as wxyz
   ``(0.7071, 0, -0.7071, 0)``, was consumed as xyzw -- which reads as a 180 deg rotation
   about ``(1, 0, -1)/sqrt(2)``. (The offset itself is also unnecessary: it cancels a URDF
   tool-frame quirk, but the USD ``gripper_center`` frames are already exact left/right
   mirror images.)

Both bugs are invertible, so the datasets are repaired offline, without re-running Isaac Sim:
matrix -> quaternion -> shift the components back -> matrix, and for the left arm right-multiply
the (self-inverse) 180 deg offset away first. The repaired left frame is the raw USD
``gripper_center``, mirroring the right arm's convention.

Ground truth for both the diagnosis and the verification is an analytic FK over the A2D
robot's USD physics joints (base_link -> gripper_center chain), evaluated at the recorded
joint states. Against it, the repaired rotations agree to <0.05 deg at every timestep while
positions were already exact -- and a scrambled dataset is >100 deg off. ``observation.state``
stores joint angles relative to the initial pose (all-zero at t=0); absolute angles are
``state + the first action row``. Torso joints are fixed at the environment defaults.

Verify (non-destructive, exits non-zero when the rotations don't match clean FK)::

    .venv/bin/python isaaclab_arena_cumotion/scripts/fix_eef_9d_rotations.py \\
        --dataset /home/ubuntu/playground/datasets/agibot_arena_v0/handover_toast_with_ee_pose

Repair, grafting the corrected columns into a sibling dataset (e.g. the joint-only original)::

    .venv/bin/python isaaclab_arena_cumotion/scripts/fix_eef_9d_rotations.py \\
        --dataset /home/ubuntu/playground/datasets/agibot_arena_v0/handover_toast_with_ee_pose \\
        --apply-to /home/ubuntu/playground/datasets/agibot_arena_v0/handover_toast
"""

import argparse
import json
import pathlib
import sys
import urllib.request
from collections import deque

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

A2D_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Agibot/A2D/A2D_physics.usd"
)
EEF_KEYS = ("observation.eef_9d", "action.eef_9d")
EEF_TARGETS = {"left": "/A2D/gripper_center", "right": "/A2D/right_gripper_center"}
ARM_JOINTS = {
    "left": [f"left_arm_joint{i}" for i in range(1, 8)],
    "right": [f"right_arm_joint{i}" for i in range(1, 8)],
}
ARM_STATE_SLICES = {"left": slice(0, 7), "right": slice(10, 17)}
# Environment defaults the demos were recorded with (agibot_make_toast/stack_bowls envs).
TORSO_ANGLES = {
    "joint_lift_body": 0.1995,
    "joint_body_pitch": 0.6025,
    "joint_head_yaw": 0.0,
    "joint_head_pitch": 0.6708,
}

_SQRT_HALF = float(np.sqrt(0.5))
# The spurious left offset actually applied by the buggy pipeline: the intended -90 deg-about-Y
# wxyz quaternion (0.7071, 0, -0.7071, 0), consumed as xyzw = 180 deg about (1,0,-1)/sqrt(2).
# It is its own inverse.
LEFT_SPURIOUS_OFFSET = Rotation.from_quat([_SQRT_HALF, 0.0, -_SQRT_HALF, 0.0]).as_matrix()

VERIFY_POS_TOL_MM = 1.0
VERIFY_ROT_TOL_DEG = 0.5


# ------------------------------------------------------------------- the actual repair ---
def rot6d_to_matrices(rot6d: np.ndarray) -> np.ndarray:
    """(N, 6) first-two-columns representation -> (N, 3, 3)."""
    col0, col1 = rot6d[:, 0:3], rot6d[:, 3:6]
    return np.stack([col0, col1, np.cross(col0, col1)], axis=2)


def matrices_to_rot6d(matrices: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 6) first two columns."""
    return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)


def unscramble(matrices: np.ndarray) -> np.ndarray:
    """Invert the xyzw-quaternion-read-as-wxyz scramble.

    The buggy pipeline held a correct quaternion in xyzw order and converted it with a wxyz
    consumer, i.e. it built the matrix from the cyclically shifted tuple. Converting the bad
    matrix back to a quaternion recovers that shifted tuple (up to overall sign, which does
    not change the rotation), so one cyclic shift in the opposite direction restores the
    original quaternion.
    """
    shifted = Rotation.from_matrix(matrices).as_quat()  # scipy: xyzw
    return Rotation.from_quat(np.roll(shifted, 1, axis=1)).as_matrix()


def repair_eef_9d(eef: np.ndarray) -> np.ndarray:
    """Repair one (N, 18) eef_9d array: positions kept, rotations unscrambled per arm.

    The left arm additionally sheds the spurious 180 deg offset, restoring the raw USD
    ``gripper_center`` frame (the exact mirror of the right arm's convention).
    """
    repaired = {}
    for side, offset in (("left", 0), ("right", 9)):
        pos = eef[:, offset : offset + 3]
        mats = unscramble(rot6d_to_matrices(eef[:, offset + 3 : offset + 9]))
        if side == "left":
            mats = mats @ LEFT_SPURIOUS_OFFSET
        repaired[side] = np.concatenate([pos, matrices_to_rot6d(mats)], axis=1)
    return np.concatenate([repaired["left"], repaired["right"]], axis=1).astype(np.float32)


# --------------------------------------------------- analytic USD FK (the ground truth) ---
def load_fk_chains(usd_path: str) -> dict:
    """Parse the A2D USD physics joints into base_link -> gripper_center FK chains.

    Each joint contributes ``parent_T_child = T0 @ Rz(theta) @ T1^-1`` (Tz for prismatic
    joints), with T0/T1 the joint's localPos/Rot on body0/body1. Chains are found by BFS over
    the joint graph, so fixed attachment joints along the way are handled uniformly.
    """
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    joints = []
    for prim in stage.Traverse():
        if prim.GetTypeName() not in ("PhysicsRevoluteJoint", "PhysicsFixedJoint", "PhysicsPrismaticJoint"):
            continue
        joint = UsdPhysics.Joint(prim)
        bodies0, bodies1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not bodies0 or not bodies1:
            continue

        def frame(pos_attr, rot_attr):
            quat = rot_attr.Get()
            w, (x, y, z) = quat.GetReal(), quat.GetImaginary()
            T = np.eye(4)
            T[:3, :3] = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(w, x, y, z))).GetTranspose())
            T[:3, 3] = np.array(pos_attr.Get())
            return T

        joints.append(
            dict(
                name=prim.GetName(),
                prismatic=prim.GetTypeName() == "PhysicsPrismaticJoint",
                fixed=prim.GetTypeName() == "PhysicsFixedJoint",
                body0=str(bodies0[0]),
                body1=str(bodies1[0]),
                T0=frame(joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                T1inv=np.linalg.inv(frame(joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr())),
            )
        )

    adjacency = {}
    for index, joint in enumerate(joints):
        adjacency.setdefault(joint["body0"], []).append((index, +1))
        adjacency.setdefault(joint["body1"], []).append((index, -1))

    def chain_to(target: str) -> list:
        prev = {"/A2D/base_link": None}
        queue = deque(["/A2D/base_link"])
        while queue:
            body = queue.popleft()
            if body == target:
                break
            for index, direction in adjacency.get(body, []):
                nxt = joints[index]["body1"] if direction > 0 else joints[index]["body0"]
                if nxt not in prev:
                    prev[nxt] = (body, index, direction)
                    queue.append(nxt)
        assert target in prev, f"no joint chain from base_link to {target}"
        links, body = [], target
        while prev[body] is not None:
            parent, index, direction = prev[body]
            links.append((joints[index], direction))
            body = parent
        return list(reversed(links))

    return {side: chain_to(path) for side, path in EEF_TARGETS.items()}


def fk(chain: list, angles: dict) -> np.ndarray:
    """Pose of the chain tip in the base_link frame for the given joint angles."""
    T = np.eye(4)
    for joint, direction in chain:
        theta = 0.0 if joint["fixed"] else angles.get(joint["name"], 0.0)
        mid = np.eye(4)
        if joint["prismatic"]:
            mid[2, 3] = theta
        else:
            c, s = np.cos(theta), np.sin(theta)
            mid[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        link = joint["T0"] @ mid @ joint["T1inv"]
        T = T @ (np.linalg.inv(link) if direction < 0 else link)
    return T


def verify_episode(eef: np.ndarray, absolute_joints: np.ndarray, chains: dict, stride: int) -> tuple:
    """Max position (mm) / rotation (deg) error of an eef_9d array against analytic FK."""
    max_pos_mm, max_rot_deg = 0.0, 0.0
    for t in range(0, len(eef), stride):
        angles = dict(TORSO_ANGLES)
        for side in ARM_JOINTS:
            angles.update(zip(ARM_JOINTS[side], absolute_joints[t, ARM_STATE_SLICES[side]]))
        for side, offset in (("left", 0), ("right", 9)):
            T = fk(chains[side], angles)
            R_data = rot6d_to_matrices(eef[t : t + 1, offset + 3 : offset + 9])[0]
            max_pos_mm = max(max_pos_mm, float(np.linalg.norm(T[:3, 3] - eef[t, offset : offset + 3])) * 1e3)
            cos = np.clip((np.trace(T[:3, :3].T @ R_data) - 1.0) / 2.0, -1.0, 1.0)
            max_rot_deg = max(max_rot_deg, float(np.degrees(np.arccos(cos))))
    return max_pos_mm, max_rot_deg


# ------------------------------------------------------------------------------ driver ---
def resolve_usd(path_arg: str | None) -> str:
    if path_arg:
        return path_arg
    cached = pathlib.Path.home() / ".cache" / "isaaclab_arena" / "A2D_physics.usd"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {A2D_USD_URL} -> {cached}")
        urllib.request.urlretrieve(A2D_USD_URL, cached)
    return str(cached)


def update_meta(dataset: pathlib.Path, source: pathlib.Path) -> None:
    """Copy the eef feature schema into the target's meta and record provenance."""
    info = json.loads((dataset / "meta/info.json").read_text())
    source_info = json.loads((source / "meta/info.json").read_text())
    for key in EEF_KEYS:
        info["features"][key] = source_info["features"][key]
    (dataset / "meta/info.json").write_text(json.dumps(info, indent=4))

    modality_path = dataset / "meta/modality.json"
    if modality_path.exists():
        modality = json.loads(modality_path.read_text())
        for group, key in (("state", EEF_KEYS[0]), ("action", EEF_KEYS[1])):
            modality[group]["left_eef_9d"] = {"start": 0, "end": 9, "original_key": key}
            modality[group]["right_eef_9d"] = {"start": 9, "end": 18, "original_key": key}
        modality_path.write_text(json.dumps(modality, indent=4))

    (dataset / "meta/eef_9d.json").write_text(
        json.dumps(
            {
                "representation": "eef_9d = xyz + first two columns of the rotation matrix",
                "layout": "left_eef_9d[0:9] + right_eef_9d[9:18]",
                "coordinate_frame": "base_link",
                "left_target": "gripper_center (raw USD frame, no offset)",
                "right_target": "right_gripper_center (raw USD frame, no offset)",
                "convention_note": (
                    "left and right frames are exact mirror images across the robot's xz-plane;"
                    " no gripper channel -- take it from action/state joint indices 7 and 17"
                ),
                "provenance": f"rotations repaired from {source.name} by fix_eef_9d_rotations.py",
            },
            indent=4,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=pathlib.Path, required=True, help="The *_with_ee_pose LeRobot dataset.")
    parser.add_argument(
        "--apply-to",
        type=pathlib.Path,
        default=None,
        help="Graft the repaired eef_9d columns into this sibling dataset (episodes must match row-for-row).",
    )
    parser.add_argument("--usd", type=str, default=None, help="Local A2D_physics.usd (downloaded when omitted).")
    parser.add_argument("--verify-episodes", type=int, default=3, help="Episodes to FK-verify (spread evenly).")
    parser.add_argument("--verify-stride", type=int, default=25, help="Verify every Nth frame.")
    args = parser.parse_args()

    episodes = sorted((args.dataset / "data/chunk-000").glob("episode_*.parquet"))
    assert episodes, f"no episodes under {args.dataset}"
    chains = load_fk_chains(resolve_usd(args.usd))

    picks = [episodes[i] for i in np.linspace(0, len(episodes) - 1, args.verify_episodes, dtype=int)]
    failed = False
    for path in picks:
        frame = pd.read_parquet(path)
        state = np.stack(frame["observation.state"].to_numpy())
        absolute = state + np.stack(frame["action"].to_numpy())[0]  # state is relative to the initial pose
        eef = np.stack(frame["observation.eef_9d"].to_numpy())
        raw = verify_episode(eef, absolute, chains, args.verify_stride)
        fixed = verify_episode(repair_eef_9d(eef), absolute, chains, args.verify_stride)
        print(
            f"{path.name}: as-stored pos {raw[0]:.3f} mm rot {raw[1]:.2f} deg"
            f" | repaired pos {fixed[0]:.3f} mm rot {fixed[1]:.2f} deg"
        )
        if fixed[0] > VERIFY_POS_TOL_MM or fixed[1] > VERIFY_ROT_TOL_DEG:
            failed = True
        if raw[1] <= VERIFY_ROT_TOL_DEG:
            print(f"  note: {path.name} already matches FK as stored -- repairing it would corrupt it.")
            failed = True
    if failed:
        print("verification FAILED: this dataset does not carry the known scramble; not repairing.")
        return 1

    if args.apply_to is None:
        print("verification passed (repair transform reproduces clean FK). Rerun with --apply-to to graft.")
        return 0

    for path in episodes:
        source = pd.read_parquet(path)
        target_path = args.apply_to / "data/chunk-000" / path.name
        target = pd.read_parquet(target_path)
        assert len(source) == len(target), f"{path.name}: row count mismatch"
        for key in ("observation.state", "action"):
            assert np.array_equal(
                np.stack(source[key].to_numpy()), np.stack(target[key].to_numpy())
            ), f"{path.name}: {key} differs between datasets"
        for key in EEF_KEYS:
            target[key] = list(repair_eef_9d(np.stack(source[key].to_numpy())))
        target.to_parquet(target_path, index=False)
    update_meta(args.apply_to, args.dataset)
    print(f"grafted repaired eef_9d into {len(episodes)} episodes of {args.apply_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
