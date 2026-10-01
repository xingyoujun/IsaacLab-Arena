# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single robot preview of annotated appliance interactions, with actual mechanism state and cameras."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=["kettle_release", "toaster_cancel"], required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--harness-run", type=Path)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--enable_cameras", action="store_true")
args = parser.parse_args()
args.headless = True
cameras_enabled = args.enable_cameras
launcher = AppLauncher(args)


def main():  # noqa: C901
    import h5py
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    from isaaclab.managers.recorder_manager import DatasetExportMode
    from pxr import Usd, UsdGeom, UsdPhysics

    from data_engine.annotations.skill_trace import SkillTrace, write_hdf5_trace
    from data_engine.assets.interactions import interaction, world_pose
    from data_engine.harness.steps import ControlStepObserver
    from data_engine.interactions.evidence import evaluate
    from data_engine.interactions.scene_guard import ApplianceGuard
    from data_engine.interactions.tool_contact import fingertip_offset
    from data_engine.motion.cumotion.executor import EnvActionExecutor, JointActionInterface
    from data_engine.motion.cumotion.planner import CumotionArmPlanner
    from data_engine.pine_wm.collection.contacts import ContactAudit
    from data_engine.pine_wm.collection.live_preview import LivePreview
    from isaaclab_arena.embodiments.ur7e.demo_recorders import ur7e_demo_recorder_cfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.pine_wm_interactions_environment import (
        PineWmKettleEnvironment,
        PineWmKettleEnvironmentCfg,
        PineWmToasterEnvironment,
        PineWmToasterEnvironmentCfg,
    )

    task = json.loads(Path(__file__).with_name("tasks.json").read_text())["tasks"][args.task]
    if args.harness_run:
        spec = json.loads((args.harness_run / "run.json").read_text())["spec"]
        assert spec["scenario"]["layout"] == task and spec["seed"] == args.seed
        assert args.output.resolve() == args.harness_run.resolve() / "payload"
    kind = task["kind"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "layout_snapshot.json").write_text(json.dumps(task, indent=2))
    factory, config = (
        (PineWmKettleEnvironment, PineWmKettleEnvironmentCfg)
        if kind == "kettle"
        else (PineWmToasterEnvironment, PineWmToasterEnvironmentCfg)
    )
    arena = factory().build(
        config(
            enable_cameras=cameras_enabled, object_position=tuple(task["position_m"]), object_yaw_deg=task["yaw_deg"]
        )
    )
    configure_scene = arena.env_cfg_callback

    def configure(cfg):
        cfg = configure_scene(cfg)
        cfg.recorders = ur7e_demo_recorder_cfg(with_cameras=False)
        cfg.recorders.dataset_export_dir_path = str(args.output)
        cfg.recorders.dataset_filename = "demos"
        cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
        cfg.env_name = task["environment"]
        cfg.sim.render_interval = cfg.decimation
        return cfg

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
    interface = JointActionInterface(env)
    executor = EnvActionExecutor(env, planner, interface, "arm_action", "gripper_action")
    asset = env.scene[kind]
    latch = env.interaction_latches[kind]
    current = {"phase": "reset", "manipulated": kind, "contact_bodies": [], "contact_details": True}
    audit = ContactAudit(env, current, [kind])
    original_physics_hook = env.recorder_manager.record_post_physics_decimation_step

    def physics_hook():
        original_physics_hook()
        audit.check()

    env.recorder_manager.record_post_physics_decimation_step = physics_hook
    executor.step(steps=15)
    env.recorder_manager.reset([0])
    env.recorder_manager.record_post_reset([0])
    trace = SkillTrace(args.task, float(env.step_dt))
    preview = LivePreview(env, args.output) if cameras_enabled else None
    if preview:
        preview.begin(0)
    samples, decisions = [], []
    initial_latch_state = list(latch.engaged[0])
    initial_latch_event_count = len(latch.events)

    def capture():
        samples.append(
            dict(
                q=asset.data.joint_pos.torch[0].cpu().tolist(),
                engaged=list(latch.engaged[0]),
                physics_step=latch.physics_step,
                stage=current["phase"],
            )
        )
        if preview:
            preview.capture()

    env.step = ControlStepObserver(env.step, lambda: trace, capture)
    guard = ApplianceGuard(env, planner, kind)
    executor.on_step = lambda: guard.check(planner.joint_positions(), "measured_joints")
    original_check = planner.safety.check_configuration

    def check_configuration(q, stage="planned"):
        original_check(q, stage)
        guard.check(q, stage)

    planner.safety.check_configuration = check_configuration
    # Leave the robot mounting footprint out of the planning obstacle; otherwise
    # the fixed base collision spheres make every initial configuration invalid.
    for label, center, size in (
        ("front", [0.0, 0.115, 0.70], (1.0, 0.77, 0.08)),
        ("left", [-0.335, -0.385, 0.70], (0.33, 0.23, 0.08)),
        ("right", [0.335, -0.385, 0.70], (0.33, 0.23, 0.08)),
    ):
        planner.add_box_obstacle("worktable_" + label, np.array(center), size, safety_tolerance_m=0.002)
    # Protect the main body in planning; contact links remain subject to the PhysX audit.
    root = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{kind}/Links/{kind if kind == 'kettle' else 'body'}")
    cache = UsdGeom.BBoxCache(0, ["default", "render", "proxy", "guide"])
    body_obstacles = []
    for index, prim in enumerate(Usd.PrimRange(root)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            planner.add_box_obstacle(
                f"body_{index}", np.array(box.GetMidpoint()), tuple(box.GetSize()), safety_tolerance_m=0.001
            )
            body_obstacles.append(f"body_{index}")

    def mark(name):
        current["phase"] = name
        trace.stage(name)

    tool_offset = np.array([0.0, 0.0, task["tcp_offset_m"]])
    contact_segment_start = None
    last_contact_command = None

    def move(point, rotation, linear=False, contact_goal=False):
        nonlocal last_contact_command
        quat = rotation.as_quat()[[3, 0, 1, 2]]
        tool = np.asarray(point) - rotation.apply(tool_offset)
        print("MOVE", current["phase"], np.round(point, 4), flush=True)
        if linear:
            # Preserve the commanded spring preload across small corrections.
            # Restarting at measured (deflected) joints drops that preload and
            # lets the carriage recoil before every additional press.
            start = (
                last_contact_command if contact_goal and last_contact_command is not None else planner.joint_positions()
            )
            candidate = planner.plan_cartesian(start, tool, quat, step_m=0.004)
        else:
            planner.safety.path_constraint = None
            candidate = planner.plan_pose(planner.joint_positions(), tool, quat)
        assert candidate is not None and candidate.is_executable(), "No collision-checked executable path"
        if contact_goal:
            # Compliance may move back along the press segment during a small
            # correction. Keep the original finite contact corridor, including
            # its unchanged lateral/angle tolerances, rather than shrinking its
            # start to the loaded endpoint on every correction.
            planner.safety.path_constraint["start"] = contact_segment_start
        command = executor.follow(candidate.path, speed=task["contact_speed"] if linear else 0.35)
        if contact_goal:
            last_contact_command = command
        planner.safety.path_constraint = None
        error = float(np.linalg.norm(planner.tool_position() - tool))
        limit = task["max_contact_tracking_error_m"] if contact_goal else 0.004
        assert error < limit, f"Tool tracking error: {error}"

    success, failure = False, None
    metrics = {}
    try:
        trace.start_skill("verify_initial_latch", entities={"object": kind})
        mark("initial_hold")
        executor.close_gripper()
        executor.step(steps=task["hold_steps"])
        initial_engaged = kind == "kettle"
        assert latch.engaged[0][0] == initial_engaged, "Unexpected initial latch state"
        trace.end_skill(success=True, evidence={"engaged": latch.engaged[0][0], "expected": initial_engaged})
        ready_joints = planner.joint_positions().copy()
        for sequence_index, name in enumerate(task["sequence"]):
            for obstacle in body_obstacles:
                planner.set_obstacle_enabled(obstacle, True)
            if sequence_index:
                trace.start_skill("transit", entities={"object": kind})
                mark("return_to_ready")
                planner.safety.path_constraint = None
                transit = planner.plan_config(planner.joint_positions(), ready_joints)
                assert transit is not None and transit.is_executable(), "No safe transit back to ready"
                executor.follow(transit.path, speed=0.35)
                tracking = float(np.max(np.abs(planner.joint_positions() - ready_joints)))
                assert tracking < 0.03, f"Ready transit tracking error: {tracking}"
                trace.end_skill(success=True, evidence={"max_joint_error_rad": tracking})
            candidate = interaction(arena.interaction_annotations, name)
            current["contact_bodies"] = [candidate["body"]]
            trace.start_skill("prepare_contact_tool", entities={"object": kind, "interaction": name})
            mark("configure_finger_" + name)
            if name in task.get("single_finger_interactions", []):
                executor.open_gripper()
                executor.step(steps=20)
                tool_offset = fingertip_offset(env, planner)
            else:
                executor.close_gripper()
                tool_offset = np.array([0.0, 0.0, task["tcp_offset_m"]])
            trace.end_skill(success=None, evidence={"tool_contact_offset_m": tool_offset.tolist()})
            body_id = asset.body_names.index(candidate["body"])
            T_W_L = asset.data.body_pose_w.torch[0, body_id].cpu().numpy()
            T_W_G = world_pose(candidate, T_W_L)
            point = T_W_G[:3]
            annotated_rotation = Rotation.from_quat(T_W_G[3:])
            axis = Rotation.from_quat(T_W_L[3:]).apply(candidate["joint"]["axis_in_body"])
            distance = candidate["joint"]["target"] - float(
                asset.data.joint_pos.torch[0, asset.joint_names.index(candidate["joint"]["name"])]
            )
            decisions.append(
                dict(
                    interaction=name,
                    body=candidate["body"],
                    T_W_G=T_W_G.tolist(),
                    distance_m=distance,
                    tool_contact_offset_m=tool_offset.tolist(),
                    source_usd_sha256=arena.interaction_annotations["source"]["usd_sha256"],
                )
            )
            trace.start_skill("press", entities={"object": kind, "body": candidate["body"]}, parameters=decisions[-1])
            mark("select_" + name)
            approach = annotated_rotation.as_matrix()[:, 2]
            inward = asset.data.root_pos_w.torch[0].cpu().numpy() - point
            inward[2] = 0
            inward /= np.linalg.norm(inward)
            chosen = None
            for tilt in task.get("interaction_tool_tilt_degrees", {}).get(name, task["tool_tilt_degrees"]):
                if abs(approach[2]) > 0.9:
                    z_axis = np.cos(np.deg2rad(tilt)) * approach + np.sin(np.deg2rad(tilt)) * inward
                else:
                    z_axis = np.cos(np.deg2rad(tilt)) * approach + np.sin(np.deg2rad(tilt)) * np.array([0, 0, -1])
                lateral_reference = inward if abs(approach[2]) > 0.9 else approach
                y_axis = np.cross([0, 0, 1], lateral_reference)
                y_axis /= np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                for spin in (0, 180):
                    rotation = Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis])) * Rotation.from_euler(
                        "z", spin, degrees=True
                    )
                    pregrasp = point - z_axis * task["standoff_m"]
                    quat = rotation.as_quat()[[3, 0, 1, 2]]
                    tool = pregrasp - rotation.apply(tool_offset)
                    planner.safety.path_constraint = None
                    path = planner.plan_pose(planner.joint_positions(), tool, quat)
                    accepted = path is not None and path.is_executable()
                    decisions.append(
                        dict(
                            interaction=name,
                            tilt_deg=tilt,
                            spin_deg=spin,
                            approach_feasible=accepted,
                            plan_returned=path is not None,
                            margin=path.limit_margin_rad if path else None,
                            travel=path.max_travel_rad if path else None,
                        )
                    )
                    if accepted:
                        chosen = (rotation, pregrasp, path)
                        break
                if chosen is not None:
                    break
            assert chosen is not None, "No feasible annotated-contact tool orientation"
            rotation, pregrasp, path = chosen
            mark("approach_" + name)
            executor.follow(path.path, speed=0.35)
            for obstacle in body_obstacles:
                planner.set_obstacle_enabled(obstacle, False)
            mark("contact_" + name)
            move(point - approach * 0.005, rotation, True)
            mark("actuate_" + name)
            contact_segment_start = planner.tool_position().copy()
            last_contact_command = None
            end = point + axis * (distance + task["press_overtravel_m"])
            expected_latch = bool(candidate.get("held_by"))
            reached = False
            for correction in task["contact_corrections_m"]:
                move(end + axis * correction, rotation, True, contact_goal=True)
                executor.step(steps=5)
                reached = latch.engaged[0][0] == expected_latch
                trace.event(
                    "contact_goal_checked",
                    {
                        "correction_m": correction,
                        "expected_engaged": expected_latch,
                        "engaged": latch.engaged[0][0],
                        "q": samples[-1]["q"],
                    },
                )
                if reached:
                    break
            assert reached, "Contact stroke exhausted without the required latch transition"
            executor.step(steps=5)
            mark("retreat_" + name)
            move(pregrasp, rotation, True)
            mark("hold_" + name)
            executor.step(steps=task["hold_steps"])
            trace.end_skill(
                success=None, evidence={"joint_positions": samples[-1]["q"], "latch_engaged": latch.engaged[0][0]}
            )
        metrics = evaluate(task, asset.joint_names, samples, latch.events)
        success = metrics["success"] and not audit.forbidden and audit.events > 0
    except (RuntimeError, AssertionError) as error:
        failure = str(error)
        print("INTERACTION_FAILED", failure, flush=True)
    finally:
        document = trace.finish(success, failure)
        (args.output / "annotations.json").write_text(json.dumps(document, indent=2))
        report = dict(
            task=args.task,
            seed=args.seed,
            success=bool(success),
            error=failure,
            metrics=metrics,
            mode="robot_contact",
            joint_names=asset.joint_names,
            samples=samples,
            events=latch.events,
            forbidden_contacts=audit.forbidden,
            contact_events=audit.events,
            decisions=decisions,
            safety=planner.safety.report,
            annotation_source=arena.interaction_annotations["source"],
        )
        report["scene_guard"] = dict(checks=guard.checks, minimum_clearance_m=guard.minimum)
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[success]], device=env.device))
        env.recorder_manager.export_episodes([0])
        raw_path = args.output / ("demos.hdf5" if success else "demos_failed.hdf5")
        if raw_path.exists():
            with h5py.File(raw_path, "a") as file:
                write_hdf5_trace(file["data/demo_0"], document, len(samples))
                mechanism = file["data/demo_0"].create_group("mechanism")
                post = np.asarray([sample["engaged"] for sample in samples], dtype=np.bool_)
                pre = np.concatenate([np.asarray([initial_latch_state], dtype=np.bool_), post[:-1]], axis=0)
                mechanism.create_dataset("latch_engaged_pre", data=pre)
                mechanism.create_dataset("latch_engaged_post", data=post)
                mechanism.create_dataset(
                    "events_json", data=json.dumps(latch.events[initial_latch_event_count:]), dtype=h5py.string_dtype()
                )
                mechanism.attrs["asset"] = kind
                mechanism.attrs["latch_names"] = json.dumps([rule["name"] for rule in latch.rules])
                mechanism.attrs["joint_names"] = json.dumps(asset.joint_names)
        if preview:
            preview.finish(success)
        env.close()
    assert success, f"Robot preview failed: {failure or metrics}"


code = 0
try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    code = 1
finally:
    launcher.app.close(exit_code=code)
