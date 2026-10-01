# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify and record physical dial turns for the four toasters on two boxx supports."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preview-only", action="store_true")
parser.add_argument("--output", required=True)
parser.add_argument("--turn-deg", type=float, default=15.0)
parser.add_argument("--audit-self-collision", action="store_true")
parser.add_argument("--env", default="ur7e_usdcraft_turn_toaster_knob")
parser.add_argument("--num-demos", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--randomize-pose", action="store_true")
parser.add_argument("--init-joint-std", type=float, default=0.0)
parser.add_argument("--no-video", action="store_true")
parser.add_argument(
    "--record-demo",
    action="store_true",
    help="Export one successful states-only HDF5 demo at 15 Hz.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert args.num_demos > 0
args.enable_cameras = not args.no_video
launcher = AppLauncher(args)

import json
import math
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

import imageio.v2 as iio
import warp as wp
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils.math import matrix_from_quat

import isaaclab_arena_environments  # noqa: F401
from data_engine.motion.cumotion.executor import ArmExecutor, EnvActionExecutor, JointActionInterface
from data_engine.motion.cumotion.grasps import quat_wxyz_from_matrix
from data_engine.motion.cumotion.planner import CumotionArmPlanner
from data_engine.motion.cumotion.robot_description import import_cumotion
from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.embodiments.ur7e.demo_recorders import ur7e_demo_recorder_cfg
from isaaclab_arena.embodiments.ur7e.ur7e import TCP_OFFSET_FROM_GRIPPER_BASE_M, Ur7eJointRecordingActionsCfg
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena_environments.ur7e_turn_toaster_knob_environment import BOX_HEIGHT_M, BOX_SIZE_M

output = Path(args.output)
assert not output.exists() or not any(
    output.iterdir()
), "Use a fresh output directory; keep earlier qualification artifacts"
output.mkdir(parents=True, exist_ok=True)
factory = EnvironmentRegistry().get_component_by_name(args.env)()
cfg = factory._legacy_argparse_cfg_type(enable_cameras=not args.no_video, randomize_toaster_pose=args.randomize_pose)
arena_env = factory.build(cfg)
arena_env.embodiment.event_config.reset_robot_joints.params["std"] = args.init_joint_std
if args.record_demo:
    arena_env.embodiment.action_config = Ur7eJointRecordingActionsCfg()
    previous_callback = arena_env.env_cfg_callback

    def recording_config(patched):
        patched = previous_callback(patched) if previous_callback else patched
        patched.env_name = factory.name
        patched.recorders = ur7e_demo_recorder_cfg(with_cameras=False)
        patched.recorders.dataset_export_dir_path = str(output)
        patched.recorders.dataset_filename = "demo"
        patched.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        patched.terminations.success = None
        patched.episode_length_s = 600.0
        return patched

    arena_env.env_cfg_callback = recording_config
cli = get_isaaclab_arena_cli_parser().parse_args(
    ["--num_envs", "1", "--seed", str(args.seed)] + ([] if args.no_video else ["--enable_cameras"])
)
env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(cli)).make_registered().unwrapped
env.sim.reset()
env.reset()
planner = CumotionArmPlanner(env, arena_env.embodiment)
assert planner.kinematics_error_m() < 0.003
writers = {
    name: iio.get_writer(str(output / f"{name}.mp4"), fps=15, codec="libx264", macro_block_size=8)
    for name in (() if args.no_video else ("d435", "scene"))
}
frames = [0]
audit_inspector = (
    import_cumotion().create_robot_world_inspector(planner.robot_description) if args.audit_self_collision else None
)
audit_rows = []
phase = "initial_hold"


def record():
    frames[0] += 1
    if audit_inspector is not None:
        measured_q = planner.joint_positions()
        collisions = list(audit_inspector.frames_in_self_collision(measured_q.reshape(-1, 1)))
        audit_rows.append({
            "step": frames[0],
            "phase": phase,
            "q": measured_q.tolist(),
            "collisions": [str(pair) for pair in collisions],
        })
        assert not collisions, f"Measured self collision during {phase}: {collisions}"
    video_every = 1 if args.record_demo else 8
    if frames[0] % video_every or not writers:
        return
    for name, key in (("d435", "realsense_d435"), ("scene", "scene_cam")):
        frame = env.scene[key].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        writers[name].append_data(frame)
        if frames[0] == (15 if args.record_demo else 120):
            iio.imwrite(str(output / f"{name}.png"), frame)


executor = (
    EnvActionExecutor(
        env,
        planner,
        JointActionInterface(env),
        "arm_action",
        "gripper_action",
        on_step=record,
    )
    if args.record_demo
    else ArmExecutor(env, planner, on_step=record)
)
asset = arena_env.task.openable_object
obj = env.scene[asset.name]
joint = list(obj.data.joint_names).index(asset.openable_joint_name)
body = list(obj.data.body_names).index(asset.knob_body)


def steps(physics_steps):
    return max(1, round(physics_steps * env.sim.get_physics_dt() / executor.dt))


def angle():
    return float(wp.to_torch(obj.data.joint_pos)[0, joint])


def solve(position, rotation, seed):
    cm = import_cumotion()
    quat = planner.to_tool_frame(quat_wxyz_from_matrix(rotation))
    target = np.eye(4)
    target[:3, :3] = cm.Rotation3(*quat).matrix()
    target[:3, 3] = position - planner.base_pos
    ik = cm.IkConfig()
    ik.cspace_seeds = [seed]
    result = cm.solve_ik(planner.kinematics, cm.Pose3(target), planner.cfg.tool_frame, ik)
    if not result.success:
        return None
    for attr in (
        "cspace_position",
        "cspace_positions",
        "q",
        "joint_positions",
        "solution",
    ):
        if hasattr(result, attr):
            return np.asarray(getattr(result, attr), dtype=float).reshape(-1)
    raise RuntimeError("Unrecognized IK result")


def run_episode(episode_output):
    global phase
    phase = "initial_hold"
    report = {
        "environment": factory.name,
        "threshold_deg": cfg.turn_threshold_deg,
        "success": False,
    }
    try:
        executor.step(arm_target=planner.joint_positions(), steps=steps(120))
        report["passive_angle_deg"] = math.degrees(angle())
        assert abs(angle()) < math.radians(0.2), "Dial drifts without contact"
        center = wp.to_torch(obj.data.body_pos_w)[0, body].cpu().numpy().astype(float)
        body_rotation = matrix_from_quat(wp.to_torch(obj.data.body_quat_w)[0, body].unsqueeze(0))[0].cpu().numpy()
        center += body_rotation @ np.array(asset.knob_offset_local)
        report["knob_center_w"] = center.tolist()
        root_position = wp.to_torch(obj.data.root_pos_w)[0].cpu().numpy().astype(float)
        root_rotation = matrix_from_quat(wp.to_torch(obj.data.root_quat_w)[0].unsqueeze(0))[0].cpu().numpy()
        outward = root_rotation @ np.array([0.0, -1.0, 0.0])
        report["toaster_root_w"] = root_position.tolist()
        print("DIAL", report, flush=True)
        if not args.preview_only:
            planner.add_box_obstacle("/obstacles/table", np.array([0.0, 0.25, 0.69]), (1.2, 1.1, 0.10))
            for i in range(2):
                planner.add_box_obstacle(
                    f"/obstacles/support_{i}",
                    root_position - np.array([0.0, 0.0, (1.5 - i) * BOX_HEIGHT_M]),
                    BOX_SIZE_M,
                    quat_wxyz=quat_wxyz_from_matrix(root_rotation),
                    safety_tolerance_m=0.0,
                )
            inspector = import_cumotion().create_robot_world_inspector(planner.robot_description)
            q0 = planner.joint_positions()
            chosen = None
            for tilt in (75, 60, 45, 90):
                for spin in (0, 180):
                    z = np.array([0.0, math.sin(math.radians(tilt)), -math.cos(math.radians(tilt))])
                    y = np.array([1.0, 0.0, 0.0])
                    x = np.cross(y, z)
                    rotation = (
                        root_rotation
                        @ np.column_stack([x, y, z])
                        @ Rotation.from_euler("z", spin, degrees=True).as_matrix()
                    )
                    z = rotation[:, 2]
                    # Grip the protruding front half of the dial, not the housing face.
                    tcp = center + outward * 0.003
                    tool = tcp - z * TCP_OFFSET_FROM_GRIPPER_BASE_M
                    standoff = tool - z * 0.065
                    q_pre = solve(standoff, rotation, q0)
                    q_grasp = solve(tool, rotation, q_pre) if q_pre is not None else None
                    if q_grasp is None:
                        print("CANDIDATE", tilt, spin, "no IK", flush=True)
                        continue
                    if list(inspector.frames_in_self_collision(q_grasp.reshape(-1, 1))):
                        print("CANDIDATE", tilt, spin, "self collision", flush=True)
                        continue
                    plan = planner.plan_config(q0, q_pre)
                    if plan is not None and plan.is_executable():
                        chosen = (rotation, tcp, tool, standoff, plan)
                        report["tilt_spin_deg"] = [tilt, spin]
                        break
                if chosen is not None:
                    break
            assert chosen is not None, "No collision-free approach candidate"
            rotation, tcp, tool, standoff, plan = chosen
            phase = "open_gripper"
            executor.open_gripper()
            phase = "planned_standoff"
            executor.follow(plan.path, speed=0.2)
            phase = "straight_approach"
            for p in np.linspace(standoff, tool, 25)[1:]:
                q = solve(p, rotation, planner.joint_positions())
                assert q is not None, "Approach IK failed"
                assert np.max(np.abs(q - planner.joint_positions())) < 0.5, "Approach IK branch jump"
                assert not list(inspector.frames_in_self_collision(q.reshape(-1, 1))), "Approach self collision"
                executor.step(arm_target=q, steps=steps(12))
            phase = "close_gripper"
            executor.close_gripper()
            report["angle_after_grasp_deg"] = math.degrees(angle())
            phase = "turn"
            for degrees in np.linspace(0, args.turn_deg, 31)[1:]:
                # Rotate about outward world -Y; either signed joint motion counts.
                turn = Rotation.from_rotvec(outward * math.radians(degrees)).as_matrix()
                rotated = turn @ rotation
                target = tcp - rotated[:, 2] * TCP_OFFSET_FROM_GRIPPER_BASE_M
                q = solve(target, rotated, planner.joint_positions())
                assert q is not None, "Turn IK failed"
                assert not list(inspector.frames_in_self_collision(q.reshape(-1, 1))), "Turn self collision"
                for _ in range(steps(12)):
                    executor.step(arm_target=q)
                    if args.record_demo and abs(angle()) > math.radians(cfg.turn_threshold_deg):
                        break
                print("TURN", degrees, "actual_deg", math.degrees(angle()), flush=True)
                if abs(angle()) > math.radians(cfg.turn_threshold_deg):
                    break
            phase = "settle"
            if not args.record_demo:
                executor.step(arm_target=planner.joint_positions(), steps=steps(60))
            report["final_angle_deg"] = math.degrees(angle())
            report["success"] = bool(
                arena_env.task.openable_object.is_open(env, threshold=math.radians(cfg.turn_threshold_deg))[0]
            )
            if args.record_demo and report["success"]:
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                report["recorded_demo"] = str(output / "demo.hdf5")
    except Exception as error:
        report["error"] = str(error)
        report["success"] = False
        print("FAILED", report, flush=True)
    finally:
        if audit_inspector is not None:
            (episode_output / "self_collision_audit.json").write_text(json.dumps(audit_rows, indent=2) + "\n")
            report["self_collision_audit"] = {
                "checked_steps": len(audit_rows),
                "sampling": "control step" if args.record_demo else "physics step",
                "colliding_steps": sum(bool(row["collisions"]) for row in audit_rows),
                "pairs": sorted({pair for row in audit_rows for pair in row["collisions"]}),
                "model": "cuMotion configured collision spheres and ignore pairs; not mesh contact",
            }
        (episode_output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("RESULT", report, flush=True)

    return report


reports = []
try:
    for episode in range(args.num_demos):
        if episode:
            if args.record_demo:
                env.recorder_manager.reset()
            env.reset()
            planner = CumotionArmPlanner(env, arena_env.embodiment)
            executor = (
                EnvActionExecutor(
                    env, planner, JointActionInterface(env), "arm_action", "gripper_action", on_step=record
                )
                if args.record_demo
                else ArmExecutor(env, planner, on_step=record)
            )
        audit_rows.clear()
        episode_output = output if args.num_demos == 1 else output / f"attempt_{episode:04d}"
        episode_output.mkdir(exist_ok=True)
        reports.append(run_episode(episode_output))
    (output / "summary.json").write_text(
        json.dumps(
            {
                "attempts": len(reports),
                "successes": sum(r["success"] for r in reports),
                "seed": args.seed,
                "randomize_pose": args.randomize_pose,
            },
            indent=2,
        )
        + "\n"
    )
finally:
    for writer in writers.values():
        writer.close()
    env.close()
    launcher.app.close()
