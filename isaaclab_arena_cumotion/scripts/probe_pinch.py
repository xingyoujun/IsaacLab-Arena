# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Can the Agibot gripper hold this object at all? The mid-air ramped pinch-and-swing gate.

The object is teleported between the open finger pads with no table underneath, the gripper's
target is ramped closed the way both the teleop stack and the cuMotion executor do, the object
is released from its pin the moment the fingers are blocked by it, and the arm is then swung to
load the grip. What comes out is the object's peak speed during the close and whether it is still
between the pads after the close and after the swing. Nothing else is in the picture -- no
approach, no table press, no controller -- so a failure here is the object's geometry against
the gripper's, and no configuration will fix it (size the object to the pads instead).

This is the arbiter for asset graspability, and the first thing to run when an object "flies out
of the gripper": passed here, the fling is the approach or the table (see ``docs/agibot/triage``).
A part is fit for a task when every repeat reads HELD/HELD and the close peak stays well under
1 m/s; a clean hold reads ~0.1-0.35 m/s, an ejection 2-7 m/s.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_pinch.py --headless \\
        --env agibot_tidy_workbench --object metal_billet --centre_offset 0.010 \\
        --env-arg billet_scale=1.2

``--centre_offset`` is the distance from the object's origin to the point put at the pads'
midpoint, along the object's local z: half the height for a bottom-origin part, 0 for a
centre-origin one. ``--object_roll_deg`` rolls the object about the approach axis so the other
in-plane side is pinched.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True, help="Registered environment holding the object.")
parser.add_argument("--object", type=str, required=True, help="Scene key of the object to pinch.")
parser.add_argument("--arm", type=str, default="right", choices=("left", "right"))
parser.add_argument(
    "--centre_offset", type=float, default=0.0, help="Object origin -> pinch point, along object z (m)."
)
parser.add_argument("--object_roll_deg", type=float, default=0.0, help="Roll of the object about the approach axis.")
parser.add_argument("--ramp_seconds", type=float, default=200 / 120, help="Open-to-closed target ramp time.")
parser.add_argument("--close_target", type=float, default=None, help="Closed gripper target; default fully closed.")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--swing_rad", type=float, default=0.2, help="Shoulder swing applied after the close.")
parser.add_argument("--pass_peak", type=float, default=1.0, help="Close-peak speed above which the pinch fails (m/s).")
from isaaclab_arena_cumotion.scripts.probe_common import add_env_override_arg  # noqa: E402

add_env_override_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
simulation_app = AppLauncher(args).app

import math  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

from isaaclab_arena.embodiments.agibot.agibot import AgibotDualArmJointActionsCfg  # noqa: E402
from isaaclab_arena_cumotion.embodiment_cumotion_registry import get_embodiment_cumotion_cfg  # noqa: E402
from isaaclab_arena_cumotion.scripts.probe_common import (  # noqa: E402
    build_probe_env,
    environment_cfg_type,
    object_position,
    object_speed,
    parse_env_overrides,
    to_torch,
)

cfg_type = environment_cfg_type(args.env)


def joint_space(arena_env):
    """Drive the arms in joint space, the recording pipeline's path, so the grip is a held target."""
    arena_env.embodiment.action_config = AgibotDualArmJointActionsCfg()


env, arena_env = build_probe_env(args.env, parse_env_overrides(args.env_arg, cfg_type), prepare=joint_space)
cumotion_cfg = get_embodiment_cumotion_cfg(arena_env.embodiment, arm=args.arm)
open_target = cumotion_cfg.gripper_open_pos
close_target = cumotion_cfg.gripper_closed_pos if args.close_target is None else args.close_target

robot = env.scene.articulations["robot"]
body_names = list(robot.data.body_names)
tool_index = body_names.index(cumotion_cfg.tool_frame)
left_pad = body_names.index(f"{args.arm}_Left_Pad_Link")
right_pad = body_names.index(f"{args.arm}_Right_Pad_Link")

manager = env.action_manager
offsets, start = {}, 0
for term_name, dim in zip(manager.active_terms, manager.action_term_dim):
    offsets[term_name] = (start, dim)
    start += dim
term_joint_ids = {term_name: list(manager.get_term(term_name)._joint_ids) for term_name in manager.active_terms}
gripper_term, arm_term = f"{args.arm}_gripper_action", f"{args.arm}_arm_action"
hand_joint = term_joint_ids[gripper_term][0]
ramp_steps = max(1, round(args.ramp_seconds / env.step_dt))
print(
    f"[pinch] {args.object} in the {args.arm} hand: ramp {ramp_steps} steps ({args.ramp_seconds:.2f} s), target"
    f" {close_target}"
)


def joint_positions():
    return to_torch(robot.data.joint_pos)[0].float().clone()


def body_position(index):
    return to_torch(robot.data.body_pos_w)[0, index].float()


def body_quat(index):
    return to_torch(robot.data.body_quat_w)[0, index].float()


def pad_gap_mm():
    return float((body_position(left_pad) - body_position(right_pad)).norm()) * 1000.0


def pad_midpoint():
    return 0.5 * (body_position(left_pad) + body_position(right_pad))


def pin_object_between_pads():
    """Write the object so its pinch point sits at the pads' midpoint, aligned with the tool."""
    quat = body_quat(tool_index)
    rotation = math_utils.matrix_from_quat(quat.unsqueeze(0))[0]
    if args.object_roll_deg:
        roll = math.radians(args.object_roll_deg)
        about_z = torch.tensor(
            [[math.cos(roll), -math.sin(roll), 0.0], [math.sin(roll), math.cos(roll), 0.0], [0.0, 0.0, 1.0]],
            device=rotation.device,
        )
        rotation = rotation @ about_z
        quat = math_utils.quat_from_matrix(rotation.unsqueeze(0))[0]
    origin = pad_midpoint() - rotation[:, 2] * args.centre_offset
    asset = env.scene[args.object]
    ids = torch.tensor([0], device=env.device)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([origin, quat]).reshape(1, 7), env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)
    env.scene.write_data_to_sim()


def action(hold, gripper_target, arm_delta=None):
    """Absolute joint targets: ``hold`` everywhere, the probed gripper at ``gripper_target``."""
    vector = torch.zeros(1, manager.total_action_dim, device=env.device)
    for term_name, (first, dim) in offsets.items():
        values = hold[term_joint_ids[term_name]].clone()
        if term_name == gripper_term:
            values[:] = gripper_target
        if arm_delta and term_name == arm_term:
            for joint, delta in arm_delta.items():
                values[joint] += delta
        vector[0, first : first + dim] = values
    return vector


def distance_to_pads():
    return float((object_position(env, args.object) - pad_midpoint()).norm())


results = []
for repeat in range(args.repeats):
    env.reset()
    hold = joint_positions()
    for _ in range(5):
        env.step(action(hold, open_target))
    hold = joint_positions()
    hold[term_joint_ids[gripper_term]] = open_target

    pinned, release_step, close_peak = True, None, 0.0
    for step in range(ramp_steps + 23):
        target = open_target + (close_target - open_target) * min(1.0, (step + 1) / ramp_steps)
        if pinned:
            pin_object_between_pads()
        env.step(action(hold, target))
        # Let go of the pin once the fingers are blocked by the object: the commanded target has
        # dropped below the measured joint.
        if pinned and target < float(joint_positions()[hand_joint]) - 0.02:
            pinned, release_step = False, step
        if not pinned:
            close_peak = max(close_peak, object_speed(env, args.object))
    held = release_step is not None and distance_to_pads() < 0.06
    gap_after_close = pad_gap_mm()

    swing_peak = 0.0
    for step in range(10):
        env.step(action(hold, close_target, arm_delta={1: args.swing_rad * (step + 1) / 10}))
        swing_peak = max(swing_peak, object_speed(env, args.object))
    for _ in range(10):
        env.step(action(hold, close_target, arm_delta={1: args.swing_rad}))
        swing_peak = max(swing_peak, object_speed(env, args.object))
    carried = distance_to_pads() < 0.06

    results.append((close_peak, held, swing_peak, carried))
    print(
        f"[pinch] repeat {repeat}: fingers blocked at step {release_step} of {ramp_steps} | close peak"
        f" {close_peak:.2f} m/s, pad gap {gap_after_close:.1f} mm -> {'HELD' if held else 'LOST'} | swing peak"
        f" {swing_peak:.2f} m/s -> {'HELD' if carried else 'LOST'}",
        flush=True,
    )

held_count = sum(result[1] for result in results)
carried_count = sum(result[3] for result in results)
worst_close = max(result[0] for result in results)
verdict = "PASS" if held_count == carried_count == args.repeats and worst_close < args.pass_peak else "FAIL"
print(
    f"[pinch] {verdict}: {args.object} ({args.arm}) held {held_count}/{args.repeats}, carried"
    f" {carried_count}/{args.repeats}, worst close peak {worst_close:.2f} m/s (limit {args.pass_peak})"
)
if verdict == "FAIL":
    print(
        "[pinch] A mid-air failure is the object's geometry against the gripper's -- too long for the pad"
        " region (~80 mm), too wide for the 104 mm jaws, a grip band shorter than ~12 mm, or a wide base"
        " under a narrow neck. Resize or regenerate the asset; do not touch the gripper."
    )

env.close()
simulation_app.close()
