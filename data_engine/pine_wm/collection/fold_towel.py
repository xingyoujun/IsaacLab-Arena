# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuMotion diagonal towel fold on the Pine workcell: friction-only corner pinch, carry and lay-down.

The towel is a free PhysX surface deformable; nothing attaches it to the gripper. Success is judged
from the settled towel nodes, not from the commanded motion.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--towel_usd", type=str, default=None, help="Defaults to ARENA_PINE_WM_TOWEL_USD.")
parser.add_argument("--towel_position", type=float, nargs=3, default=(0.0, 0.05, 0.745))
parser.add_argument("--towel_yaw_deg", type=float, default=0.0)
parser.add_argument("--grasp_inset_m", type=float, default=0.045, help="Pinch point inset from the corner.")
parser.add_argument(
    "--pinch_clearance_m", type=float, default=-0.002, help="Lowest fingertip height relative to the table top."
)
parser.add_argument("--lift_m", type=float, default=0.10)
parser.add_argument("--carry_height_m", type=float, default=0.14)
parser.add_argument("--place_short_m", type=float, default=0.11, help="Lay-down point short of the far corner.")
parser.add_argument("--place_height_m", type=float, default=0.004)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--enable_cameras", action="store_true")
args = parser.parse_args()
args.headless = True
cameras_enabled = args.enable_cameras
launcher = AppLauncher(args)

SUCCESS = dict(max_mirror_error_m=0.04, max_footprint_ratio=0.7, max_height_above_table_m=0.04, min_lift_m=0.05)
"""Settled-state fold criteria; a perfect diagonal fold has zero mirror error and footprint ratio 0.5."""
TABLE_TOP_Z = 0.74
GRIPPER_BODY_PREFIX = "ee_link/Robotiq_2F_85/"


def main():  # noqa: C901
    import h5py
    import hashlib
    import numpy as np
    import torch
    from scipy.spatial import ConvexHull
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
    from data_engine.motion.cumotion.planner import CumotionArmPlanner
    from data_engine.pine_wm.collection.live_preview import LivePreview
    from isaaclab_arena.embodiments.ur7e.demo_recorders import ur7e_demo_recorder_cfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.pine_wm_towel_environment import (
        PineWmFoldTowelEnvironment,
        PineWmFoldTowelEnvironmentCfg,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    cfg = PineWmFoldTowelEnvironmentCfg(
        enable_cameras=cameras_enabled,
        towel_usd=args.towel_usd,
        towel_position=tuple(args.towel_position),
        towel_yaw_deg=args.towel_yaw_deg,
    )
    arena = PineWmFoldTowelEnvironment().build(cfg)
    towel_usd = arena.scene.assets["towel"].spawner_cfg.usd_path
    towel_sha256 = hashlib.sha256(Path(towel_usd).read_bytes()).hexdigest()

    def configure(env_cfg):
        env_cfg.recorders = ur7e_demo_recorder_cfg(with_cameras=False)
        env_cfg.recorders.dataset_export_dir_path = str(args.output)
        env_cfg.recorders.dataset_filename = "demos"
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
        env_cfg.env_name = PineWmFoldTowelEnvironment.name
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
    towel = env.scene["towel"]

    def nodes():
        return towel.data.nodal_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)

    # Robot contacts: only the gripper may touch anything (towel and, while pinching, the table).
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
    open_tip, closing_axis = pinch_geometry()
    executor.close_gripper()
    closed_tip, _ = pinch_geometry()
    # Whichever fingertip state reaches lower sets the descent height; the other sweeps above it.
    tip_offset = open_tip if open_tip[2] >= closed_tip[2] else closed_tip
    geometry = dict(open_tip_m=open_tip.tolist(), closed_tip_m=closed_tip.tolist(), closing_axis=closing_axis.tolist())
    print("PINCH_GEOMETRY", json.dumps(geometry), flush=True)

    # Settle the towel with the arm holding its ready configuration.
    executor.open_gripper()
    executor.step(steps=int(round(1.5 / env.step_dt)))
    rest = nodes()
    assert np.isfinite(rest).all(), "Towel did not settle to a finite state"
    env.recorder_manager.reset([0])
    env.recorder_manager.record_post_reset([0])
    trace = SkillTrace("fold_towel_diagonal", float(env.step_dt))
    preview = LivePreview(env, args.output) if cameras_enabled else None
    if preview:
        preview.begin(0)
    samples = []

    def capture():
        current_nodes = nodes()
        samples.append(
            dict(
                stage=current["phase"],
                towel_max_z=float(current_nodes[:, 2].max()),
                grasp_corner=current_nodes[grasp_corner].tolist() if grasp_corner is not None else None,
            )
        )
        if preview:
            preview.capture()

    grasp_corner = None
    env.step = ControlStepObserver(env.step, lambda: trace, capture)
    for label, center, size in (
        ("front", [0.0, 0.115, 0.70], (1.0, 0.77, 0.08)),
        ("left", [-0.335, -0.385, 0.70], (0.33, 0.23, 0.08)),
        ("right", [0.335, -0.385, 0.70], (0.33, 0.23, 0.08)),
    ):
        planner.add_box_obstacle("worktable_" + label, np.array(center), size, safety_tolerance_m=0.002)
    ready_joints = planner.joint_positions().copy()

    def mark(name):
        current["phase"] = name
        trace.stage(name)

    def top_down(yaw):
        return Rotation.from_matrix(Rotation.from_euler("z", yaw).as_matrix() @ np.diag([-1.0, 1.0, -1.0]))

    def tool_target(tip, rotation):
        return np.asarray(tip) - rotation.apply(tip_offset)

    def quat(rotation):
        return quat_wxyz_from_matrix(rotation.as_matrix())

    def linear(tip, rotation, speed=0.25):
        candidate = planner.plan_cartesian(planner.joint_positions(), tool_target(tip, rotation), quat(rotation))
        if candidate is None or not candidate.is_executable():
            planner.safety.path_constraint = None
            raise RuntimeError(f"no_linear_path:{current['phase']}")
        executor.follow(candidate.path, speed=speed)
        planner.safety.path_constraint = None

    # Corners of the settled towel: extreme nodes along the two diagonals.
    center_xy = rest[:, :2].mean(axis=0)
    relative = rest[:, :2] - center_xy
    corners = {}
    for name, direction in (("ne", (1, 1)), ("sw", (-1, -1)), ("nw", (-1, 1)), ("se", (1, -1))):
        corners[name] = int(np.argmax(relative @ np.asarray(direction, dtype=np.float64)))
    opposite = {"ne": "sw", "sw": "ne", "nw": "se", "se": "nw"}
    decisions = []
    success, failure, metrics = False, None, {}
    try:
        trace.start_skill("select_corner", entities={"object": "towel"})
        mark("select_corner")
        chosen = None
        # Pinch a corner near the robot and lay it on the far one: the carry stays in front of the base.
        for name in sorted(corners, key=lambda n: rest[corners[n], 1]):
            source, target = rest[corners[name]], rest[corners[opposite[name]]]
            diagonal = (target[:2] - source[:2]) / np.linalg.norm(target[:2] - source[:2])
            pinch = np.array([*(source[:2] + diagonal * args.grasp_inset_m), TABLE_TOP_Z + args.pinch_clearance_m])
            # Close across the diagonal so the pads bunch the corner between them.
            axis_world = top_down(0.0).apply(closing_axis)
            base_yaw = np.arctan2(diagonal[1], diagonal[0]) + np.pi / 2 - np.arctan2(axis_world[1], axis_world[0])
            for yaw in (base_yaw, base_yaw + np.pi):
                rotation = top_down(yaw)
                above = pinch + [0, 0, 0.08]
                candidate = planner.plan_pose(planner.joint_positions(), tool_target(above, rotation), quat(rotation))
                feasible = candidate is not None and candidate.is_executable()
                decisions.append(dict(corner=name, yaw_rad=float(yaw), approach_feasible=feasible))
                if feasible:
                    chosen = (name, source, target, diagonal, pinch, rotation, candidate)
                    break
            if chosen is not None:
                break
        assert chosen is not None, "No feasible top-down approach to any towel corner"
        name, source, target, diagonal, pinch, rotation, candidate = chosen
        grasp_corner, place_corner = corners[name], corners[opposite[name]]
        trace.end_skill(success=True, evidence=dict(corner=name, pinch_m=pinch.tolist()))

        trace.start_skill("pinch_corner", entities={"object": "towel", "corner": name})
        mark("approach")
        executor.follow(candidate.path, speed=0.35)
        mark("descend")
        planner.set_obstacle_enabled("worktable_front", False)
        linear(pinch, rotation, speed=0.15)
        mark("close_gripper")
        executor.close_gripper()
        executor.step(steps=5)
        patch = np.where(np.linalg.norm(rest[:, :2] - pinch[:2], axis=1) < 0.03)[0]
        mark("lift")
        linear(pinch + [0, 0, args.lift_m], rotation)
        lift = float(nodes()[patch, 2].mean() - rest[patch, 2].mean())
        trace.end_skill(success=lift >= SUCCESS["min_lift_m"], evidence=dict(patch_lift_m=lift))
        if lift < SUCCESS["min_lift_m"]:
            raise RuntimeError(f"towel_not_lifted:{lift:.3f}")

        trace.start_skill("fold_over", entities={"object": "towel", "target_corner": opposite[name]})
        planner.set_obstacle_enabled("worktable_front", True)
        place = np.array([*(target[:2] - diagonal * args.place_short_m), TABLE_TOP_Z])
        mark("carry")
        linear(np.array([*((pinch[:2] + place[:2]) / 2), TABLE_TOP_Z + args.carry_height_m]), rotation)
        linear(place + [0, 0, 0.05], rotation)
        mark("lay_down")
        planner.set_obstacle_enabled("worktable_front", False)
        linear(place + [0, 0, args.place_height_m], rotation, speed=0.15)
        mark("release")
        executor.open_gripper()
        executor.step(steps=5)
        mark("retreat")
        linear(place + [0, 0, 0.10], rotation)
        planner.set_obstacle_enabled("worktable_front", True)
        trace.end_skill(success=None, evidence=dict(place_m=place.tolist()))

        trace.start_skill("return_to_ready")
        mark("return_to_ready")
        transit = planner.plan_config(planner.joint_positions(), ready_joints)
        assert transit is not None and transit.is_executable(), "No safe transit back to ready"
        executor.follow(transit.path, speed=0.35)
        mark("settle")
        executor.step(steps=int(round(1.5 / env.step_dt)))
        tracking = float(np.max(np.abs(planner.joint_positions() - ready_joints)))
        trace.end_skill(success=tracking < 0.03, evidence=dict(max_joint_error_rad=tracking))

        final = nodes()
        assert np.isfinite(final).all(), "Towel state became non-finite"
        # Each node of the folded half should land on its mirror image across the fold diagonal.
        fold_axis = np.array([-diagonal[1], diagonal[0]])
        offset = rest[:, :2] - center_xy
        along = offset @ fold_axis
        across = offset @ diagonal
        folded = np.where(across < -0.01)[0]
        mirrored = center_xy + np.outer(along[folded], fold_axis) - np.outer(across[folded], diagonal)
        partner = np.argmin(np.linalg.norm(rest[None, :, :2] - mirrored[:, None, :], axis=2), axis=1)
        mirror_error = np.linalg.norm(final[folded, :2] - final[partner, :2], axis=1)
        metrics = dict(
            corner=name,
            mirror_error_mean_m=float(mirror_error.mean()),
            mirror_error_p90_m=float(np.percentile(mirror_error, 90)),
            corner_gap_m=float(np.linalg.norm(final[grasp_corner] - final[place_corner])),
            footprint_ratio=float(ConvexHull(final[:, :2]).volume / ConvexHull(rest[:, :2]).volume),
            max_height_above_table_m=float(final[:, 2].max() - TABLE_TOP_Z),
            min_z_m=float(final[:, 2].min()),
            patch_lift_m=lift,
        )
        success = (
            metrics["mirror_error_mean_m"] <= SUCCESS["max_mirror_error_m"]
            and metrics["footprint_ratio"] <= SUCCESS["max_footprint_ratio"]
            and metrics["max_height_above_table_m"] <= SUCCESS["max_height_above_table_m"]
            and metrics["min_z_m"] > TABLE_TOP_Z - 0.01
            and not contacts["forbidden"]
        )
        print("FOLD_METRICS", json.dumps(metrics), "SUCCESS", success, flush=True)
    except (RuntimeError, AssertionError) as error:
        failure = str(error)
        print("FOLD_FAILED", failure, flush=True)
    finally:
        document = trace.finish(success, failure)
        (args.output / "annotations.json").write_text(json.dumps(document, indent=2))
        report = dict(
            task="fold_towel_diagonal",
            environment=PineWmFoldTowelEnvironment.name,
            seed=args.seed,
            success=bool(success),
            error=failure,
            metrics=metrics,
            criteria=SUCCESS,
            parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            towel=dict(usd=towel_usd, sha256=towel_sha256, nodes=int(rest.shape[0])),
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
                group = demo.create_group("towel")
                group.create_dataset("rest_nodal_pos_w", data=rest)
                group.attrs["usd"] = towel_usd
                group.attrs["usd_sha256"] = towel_sha256
                if grasp_corner is not None:
                    group.attrs["grasp_corner_node"] = grasp_corner
                    group.attrs["place_corner_node"] = place_corner
        if preview:
            preview.finish(success)
        env.close()
    assert success, f"Towel fold failed: {failure or metrics}"


code = 0
try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    code = 1
finally:
    launcher.app.close(exit_code=code)
