# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuMotion cable wrap on the Pine workcell: grasp the free plug, pull the cable around a post, set it down.

The far plug is pinned; the robot drags the free plug along a counter-clockwise arc around the post
so the tensioned cable wraps it. Success is judged from the settled cable links, not the motion.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cable_usd", type=str, default=None, help="Defaults to ARENA_PINE_WM_CABLE_USD.")
parser.add_argument("--grip_clearance_m", type=float, default=0.0015, help="Closed fingertip height above the table.")
parser.add_argument("--carry_height_m", type=float, default=0.012, help="Closed fingertip height while dragging.")
parser.add_argument("--end_angle_deg", type=float, default=160.0, help="Final plug angle around the post.")
parser.add_argument("--wrap_radius_m", type=float, default=0.055, help="Plug distance from the post axis on the arc.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--enable_cameras", action="store_true")
args = parser.parse_args()
args.headless = True
cameras_enabled = args.enable_cameras
launcher = AppLauncher(args)

SUCCESS = dict(min_winding_deg=240.0, max_post_gap_m=0.006, max_plug_height_m=0.01, min_post_clearance_m=0.03)
"""Settled-state wrap criteria, and the closest the commanded plug path may pass the post axis."""
TABLE_TOP_Z = 0.74
GRIP_POINT_IN_PLUG = (0.0, 0.0, -0.0125)
"""USDCraft's annotated plug grasp point; the plug body extends along its local -z."""
GRIPPER_BODY_PREFIX = "ee_link/Robotiq_2F_85/"


def main():  # noqa: C901
    import h5py
    import hashlib
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    import warp as wp
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from isaaclab_physx.physics import PhysxManager
    from pxr import Usd, UsdPhysics

    from data_engine.annotations.skill_trace import SkillTrace, write_hdf5_trace
    from data_engine.harness.steps import ControlStepObserver
    from data_engine.interactions.tool_contact import fingertip_offset
    from data_engine.motion.cumotion.executor import EnvActionExecutor, JointActionInterface
    from data_engine.motion.cumotion.grasps import quat_wxyz_from_matrix
    from data_engine.motion.cumotion.planner import CumotionArmPlanner, JointPath
    from data_engine.motion.cumotion.robot_description import import_cumotion
    from data_engine.pine_wm.collection.live_preview import LivePreview
    from isaaclab_arena.embodiments.ur7e.demo_recorders import ur7e_demo_recorder_cfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.pine_wm_cable_environment import (
        PineWmCableWrapEnvironment,
        PineWmCableWrapEnvironmentCfg,
    )
    from isaaclab_arena_environments.pine_wm_cable_environment import post_xy as cable_post_xy

    args.output.mkdir(parents=True, exist_ok=False)
    cfg = PineWmCableWrapEnvironmentCfg(enable_cameras=cameras_enabled, cable_usd=args.cable_usd)
    arena = PineWmCableWrapEnvironment().build(cfg)
    cable_usd = arena.scene.assets["cable"].usd_path
    cable_sha256 = hashlib.sha256(Path(cable_usd).read_bytes()).hexdigest()

    def configure(env_cfg):
        env_cfg.recorders = ur7e_demo_recorder_cfg(with_cameras=False)
        env_cfg.recorders.dataset_export_dir_path = str(args.output)
        env_cfg.recorders.dataset_filename = "demos"
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
        env_cfg.env_name = PineWmCableWrapEnvironment.name
        env_cfg.sim.render_interval = env_cfg.decimation
        return env_cfg

    arena.env_cfg_callback = configure
    if cameras_enabled:
        arena.embodiment.camera_config.set_use_tiled_camera(False)
    env = (
        ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(device=args.device, solve_relations=False, seed=args.seed))
        .make_registered()
        .unwrapped
    )
    env.reset()
    planner = CumotionArmPlanner(env, arena.embodiment)
    kinematics_error = planner.kinematics_error_m()
    assert kinematics_error < 0.002, f"cuMotion description disagrees with the simulated robot: {kinematics_error} m"
    executor = EnvActionExecutor(env, planner, JointActionInterface(env), "arm_action", "gripper_action")
    cable = env.scene["cable"]
    post_xy = np.array(cable_post_xy(cfg), dtype=np.float64)
    post_radius = cfg.post_radius_m
    plug_b = cable.body_names.index("plug_b")

    def links():
        return cable.data.body_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)

    def plug_grip():
        pose = cable.data.body_pose_w.torch[0, plug_b].detach().cpu().numpy().astype(np.float64)
        rotation = Rotation.from_quat(pose[3:])
        return pose[:3] + rotation.apply(GRIP_POINT_IN_PLUG), rotation.apply([0.0, 0.0, -1.0])

    def winding_deg(points):
        # Total signed angle swept around the post axis along the cable, pinned plug to free plug.
        angles = np.unwrap(np.arctan2(points[:, 1] - post_xy[1], points[:, 0] - post_xy[0]))
        return float(np.degrees(angles[-1] - angles[0]))

    def post_gap(points):
        return float(np.min(np.linalg.norm(points[:, :2] - post_xy, axis=1)) - post_radius)

    # Robot contacts: only the gripper may touch anything (plug, cable, post, table).
    robot_root = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")
    robot_bodies = [str(p.GetPath()) for p in Usd.PrimRange(robot_root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    contact_view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(robot_bodies)
    contact_paths = [path.split("/Robot/", 1)[1] for path in contact_view.sensor_paths]
    contacts = dict(max_gripper_force_n=0.0, max_other_force_n=0.0, forbidden=[])
    current = {"phase": "reset"}

    def check_contacts():
        net = wp.to_torch(contact_view.get_net_contact_forces(dt=env.sim.get_physics_dt())).cpu().numpy()
        for path, force in zip(contact_paths, np.linalg.norm(net, axis=1)):
            if path in {"base_link", "base_link_inertia"}:
                continue
            if path.startswith(GRIPPER_BODY_PREFIX):
                contacts["max_gripper_force_n"] = max(contacts["max_gripper_force_n"], float(force))
                continue
            contacts["max_other_force_n"] = max(contacts["max_other_force_n"], float(force))
            if force > 1.0 and current["phase"] != "reset":
                contacts["forbidden"].append(dict(body=path, force_n=float(force), phase=current["phase"]))
                raise RuntimeError(f"forbidden_robot_contact:{path}:{force:.2f}N")

    original_physics_hook = env.recorder_manager.record_post_physics_decimation_step

    def physics_hook():
        original_physics_hook()
        check_contacts()

    env.recorder_manager.record_post_physics_decimation_step = physics_hook

    # Gripper geometry measured in the tool frame (which matches the canonical +z-approach frame).
    def pinch_geometry():
        left, right = fingertip_offset(env, planner, "left"), fingertip_offset(env, planner, "right")
        return (left + right) / 2, (left - right) / np.linalg.norm(left - right)

    executor.open_gripper()
    _, closing_axis = pinch_geometry()
    executor.close_gripper()
    closed_tip, _ = pinch_geometry()
    geometry = dict(closed_tip_m=closed_tip.tolist(), closing_axis=closing_axis.tolist())
    print("PINCH_GEOMETRY", json.dumps(geometry), flush=True)

    # Settle the cable with the arm holding its ready configuration.
    executor.open_gripper()
    executor.step(steps=int(round(1.0 / env.step_dt)))
    rest = links()
    assert np.isfinite(rest).all(), "Cable did not settle to a finite state"
    env.recorder_manager.reset([0])
    env.recorder_manager.record_post_reset([0])
    trace = SkillTrace("wrap_cable_around_post", float(env.step_dt))
    preview = LivePreview(env, args.output) if cameras_enabled else None
    if preview:
        preview.begin(0)
    samples = []

    def capture():
        points = links()
        samples.append(dict(stage=current["phase"], winding_deg=winding_deg(points), plug_b=points[plug_b].tolist()))
        if preview:
            preview.capture()

    env.step = ControlStepObserver(env.step, lambda: trace, capture)
    for label, center, size in (
        ("front", [0.0, 0.115, 0.70], (1.0, 0.77, 0.08)),
        ("left", [-0.335, -0.385, 0.70], (0.33, 0.23, 0.08)),
        ("right", [0.335, -0.385, 0.70], (0.33, 0.23, 0.08)),
    ):
        planner.add_box_obstacle("worktable_" + label, np.array(center), size, safety_tolerance_m=0.002)
    planner.add_box_obstacle(
        "post",
        np.array([*post_xy, TABLE_TOP_Z + cfg.post_height_m / 2]),
        (2 * post_radius, 2 * post_radius, cfg.post_height_m),
        safety_tolerance_m=0.005,
    )
    ready_joints = planner.joint_positions().copy()

    def mark(name):
        current["phase"] = name
        trace.stage(name)

    def gripper_rotation(plug_axis_xy):
        """Top-down tool orientation whose jaws close across the given horizontal plug axis."""
        axis_world = Rotation.from_matrix(np.diag([-1.0, 1.0, -1.0])).apply(closing_axis)
        yaw = np.arctan2(plug_axis_xy[1], plug_axis_xy[0]) + np.pi / 2 - np.arctan2(axis_world[1], axis_world[0])
        return Rotation.from_matrix(Rotation.from_euler("z", yaw).as_matrix() @ np.diag([-1.0, 1.0, -1.0]))

    def tool_target(tip, rotation):
        return np.asarray(tip) - rotation.apply(closed_tip)

    def quat(rotation):
        return quat_wxyz_from_matrix(rotation.as_matrix())

    def ik(tool, rotation, seed):
        cm = import_cumotion()
        transform = np.eye(4)
        transform[:3, :3] = cm.Rotation3(*planner.to_tool_frame(quat(rotation))).matrix()
        transform[:3, 3] = tool - planner.base_pos
        ik_cfg = cm.IkConfig()
        ik_cfg.cspace_seeds = [np.asarray(seed, dtype=np.float64)]
        result = cm.solve_ik(planner.kinematics, cm.Pose3(transform), planner.cfg.tool_frame, ik_cfg)
        if not result.success:
            raise RuntimeError(f"ik_unreachable:{current['phase']}")
        q = np.asarray(result.cspace_position, dtype=np.float64).reshape(-1)
        if np.max(np.abs(q - seed)) > 0.3:
            raise RuntimeError(f"ik_branch_jump:{current['phase']}")
        return q

    def follow_poses(poses, speed):
        """Servo through densely sampled tool poses, keeping one IK branch."""
        points = [planner.joint_positions()]
        for tip, rotation in poses:
            points.append(ik(tool_target(tip, rotation), rotation, points[-1]))
        executor.follow(JointPath(points), speed=speed)

    def linear(tip, rotation, speed=0.25):
        start = planner.tool_position() + rotation.apply(closed_tip)
        count = max(2, int(np.ceil(np.linalg.norm(np.asarray(tip) - start) / 0.005)))
        follow_poses([(start + (np.asarray(tip) - start) * t, rotation) for t in np.linspace(0, 1, count)[1:]], speed)

    decisions = []
    success, failure, metrics = False, None, {}
    initial_winding = winding_deg(rest)
    try:
        grip, plug_axis = plug_grip()
        grip_tip = np.array([*grip[:2], TABLE_TOP_Z + args.grip_clearance_m])
        trace.start_skill("grasp_plug", entities={"object": "cable", "part": "plug_b"})
        mark("approach")
        chosen = None
        for flip in (1.0, -1.0):
            rotation = gripper_rotation(flip * plug_axis[:2])
            candidate = planner.plan_pose(
                planner.joint_positions(), tool_target(grip_tip + [0, 0, 0.08], rotation), quat(rotation)
            )
            feasible = candidate is not None and candidate.is_executable()
            decisions.append(dict(flip=flip, approach_feasible=feasible))
            if feasible:
                chosen = (rotation, candidate)
                break
        assert chosen is not None, "No feasible top-down approach to the free plug"
        rotation, candidate = chosen
        executor.follow(candidate.path, speed=0.35)
        mark("descend")
        planner.set_obstacle_enabled("worktable_front", False)
        linear(grip_tip, rotation, speed=0.15)
        mark("close_gripper")
        executor.close_gripper()
        executor.step(steps=5)
        mark("lift")
        linear(grip_tip + [0, 0, args.carry_height_m - args.grip_clearance_m], rotation)
        lift = float(links()[plug_b, 2] - rest[plug_b, 2])
        trace.end_skill(success=lift > 0.005, evidence=dict(plug_lift_m=lift))
        if lift <= 0.005:
            raise RuntimeError(f"plug_not_lifted:{lift:.4f}")

        trace.start_skill("wrap_around_post", entities={"object": "cable", "post": "cable_post"})
        mark("wrap")
        start = links()[plug_b, :2] - post_xy
        start_angle = float(np.arctan2(start[1], start[0]))
        end_angle = np.radians(args.end_angle_deg)
        if end_angle <= start_angle:
            end_angle += 2 * np.pi
        assert args.wrap_radius_m >= SUCCESS["min_post_clearance_m"] + post_radius, "Arc passes too close to post"
        # First pull the plug in towards the post for slack, then circle it counter-clockwise at a fixed
        # radius; the plug heads between the tangent and the outward radius so the cable trails it.
        arc_start = post_xy + args.wrap_radius_m * np.array([np.cos(start_angle), np.sin(start_angle)])
        carry_z = TABLE_TOP_Z + args.carry_height_m
        start_heading = float(np.arctan2(plug_axis[1], plug_axis[0]))
        held_xy = (planner.tool_position() + rotation.apply(closed_tip))[:2]
        path = []
        for t in np.linspace(0, 1, 10)[1:]:
            path.append((np.array([*(held_xy + (arc_start - held_xy) * t), carry_z]), None))
        for angle in np.linspace(start_angle, end_angle, max(8, int(np.degrees(end_angle - start_angle) / 3)))[1:]:
            path.append(
                (np.array([*(post_xy + args.wrap_radius_m * np.array([np.cos(angle), np.sin(angle)])), carry_z]), angle)
            )
        poses = []
        previous = rotation
        for index, (tip, angle) in enumerate(path):
            target = start_angle + np.pi / 4 if angle is None else angle + np.pi / 4
            blend = min(1.0, (index + 1) / 20)
            heading_angle = start_heading + np.angle(np.exp(1j * (target - start_heading))) * blend
            heading = np.array([np.cos(heading_angle), np.sin(heading_angle)])
            # Both jaw senses hold the plug; keep the one nearest the previous pose so the wrist turns smoothly.
            options = (gripper_rotation(heading), gripper_rotation(-heading))
            previous = min(options, key=lambda option: (option * previous.inv()).magnitude())
            poses.append((tip, previous))
        follow_poses(poses, speed=0.2)
        mark("set_down")
        place_tip = poses[-1][0].copy()
        place_tip[2] = TABLE_TOP_Z + args.grip_clearance_m
        linear(place_tip, poses[-1][1], speed=0.15)
        mark("release")
        executor.open_gripper()
        executor.step(steps=5)
        mark("retreat")
        linear(place_tip + [0, 0, 0.08], poses[-1][1])
        planner.set_obstacle_enabled("worktable_front", True)
        trace.end_skill(success=None, evidence=dict(winding_deg=winding_deg(links())))

        trace.start_skill("return_to_ready")
        mark("return_to_ready")
        transit = planner.plan_config(planner.joint_positions(), ready_joints)
        assert transit is not None and transit.is_executable(), "No safe transit back to ready"
        executor.follow(transit.path, speed=0.35)
        mark("settle")
        executor.step(steps=int(round(1.0 / env.step_dt)))
        tracking = float(np.max(np.abs(planner.joint_positions() - ready_joints)))
        trace.end_skill(success=tracking < 0.03, evidence=dict(max_joint_error_rad=tracking))

        final = links()
        assert np.isfinite(final).all(), "Cable state became non-finite"
        metrics = dict(
            initial_winding_deg=initial_winding,
            winding_deg=winding_deg(final),
            post_gap_m=post_gap(final),
            plug_height_above_table_m=float(final[plug_b, 2] - TABLE_TOP_Z),
            plug_lift_m=lift,
        )
        success = (
            metrics["winding_deg"] >= SUCCESS["min_winding_deg"]
            and metrics["post_gap_m"] <= SUCCESS["max_post_gap_m"]
            and metrics["plug_height_above_table_m"] <= SUCCESS["max_plug_height_m"]
            and not contacts["forbidden"]
        )
        print("WRAP_METRICS", json.dumps(metrics), "SUCCESS", success, flush=True)
    except (RuntimeError, AssertionError) as error:
        failure = str(error)
        print("WRAP_FAILED", failure, flush=True)
    finally:
        document = trace.finish(success, failure)
        (args.output / "annotations.json").write_text(json.dumps(document, indent=2))
        report = dict(
            task="wrap_cable_around_post",
            environment=PineWmCableWrapEnvironment.name,
            seed=args.seed,
            success=bool(success),
            error=failure,
            metrics=metrics,
            criteria=SUCCESS,
            parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            cable=dict(usd=cable_usd, sha256=cable_sha256, links=cable.body_names),
            post=dict(xy=post_xy.tolist(), radius_m=post_radius, height_m=cfg.post_height_m),
            gripper_geometry=geometry,
            decisions=decisions,
            contacts=contacts,
            safety=planner.safety.report,
            samples=samples,
        )
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[success]], device=env.device))
        env.recorder_manager.export_episodes([0])
        raw_path = args.output / ("demos.hdf5" if success else "demos_failed.hdf5")
        if raw_path.exists():
            with h5py.File(raw_path, "a") as file:
                demo = file["data/demo_0"]
                write_hdf5_trace(demo, document, len(samples))
                group = demo.create_group("cable")
                group.create_dataset("rest_link_pos_w", data=rest)
                group.create_dataset("final_link_pos_w", data=links())
                group.attrs["usd"] = cable_usd
                group.attrs["usd_sha256"] = cable_sha256
                group.attrs["link_names"] = json.dumps(cable.body_names)
                group.attrs["post_xy"] = post_xy
                group.attrs["post_radius_m"] = post_radius
        if preview:
            preview.finish(success)
        env.close()
    assert success, f"Cable wrap failed: {failure or metrics}"


code = 0
try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    code = 1
finally:
    launcher.app.close(exit_code=code)
