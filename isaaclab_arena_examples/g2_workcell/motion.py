# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plan and physically execute G2 workcell grasp, carry and release motions."""

import copy
import json
import time
import xml.etree.ElementTree as ET

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse


def movable_joints(urdf, links):
    """Find independent movable joints upstream of the selected links."""
    joints = {j.find("child").attrib["link"]: j for j in ET.parse(urdf).getroot().findall("joint")}
    names = set()
    for link in links:
        while link in joints:
            joint = joints[link]
            if joint.attrib["type"] != "fixed" and joint.find("mimic") is None:
                names.add(joint.attrib["name"])
            link = joint.find("parent").attrib["link"]
    return names


def configure_payload(kin, side, capacity=384):
    """Reserve a disabled planning-only payload link, ignoring intentional gripper overlap."""
    kin["extra_links"] = {
        "attached_object": {
            "parent_link_name": kin["ee_link"],
            "link_name": "attached_object",
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_type": "FIXED",
            "joint_name": "workcell_payload_joint",
        }
    }
    kin["collision_link_names"].append("attached_object")
    kin["extra_collision_spheres"] = {"attached_object": capacity}
    ignore = kin["self_collision_ignore"]
    for link in list(ignore):
        if "attached_object" in link:
            del ignore[link]
        else:
            ignore[link] = [n for n in ignore[link] if "attached_object" not in n]
    ignore["attached_object"] = [n for n in kin["collision_link_names"] if n.startswith(f"gripper_{side[0]}_")]


def restore_payload(planner, attachments, side):
    """Restore a measured payload after cuRobo updates locked joints."""
    if side in attachments:
        planner.attach_spheres_to_robot(sphere_tensor=attachments[side])


def save_planner_config(args, side, config):
    """Preserve separate planner configurations when switching joint-margin models."""
    import yaml

    path = args.output_dir / f"{side}_planner.yaml"
    if path.exists():
        path = args.output_dir / f"{side}_{args.robot_urdf.stem}_planner.yaml"
    path.write_text(yaml.safe_dump(config))


def prepare_execution_points(args, planner, points, name, base, entry):
    """Retime geometric paths when requested and preserve the exact execution commands."""
    import numpy as np
    import torch

    prefix = f"{args.tool}_" if args.mode == "sequence" else ""
    if name in {*getattr(args, "graph_phases", []), *getattr(args, "retime_phases", [])}:
        from isaaclab_arena_examples.g2_workcell.trajectory import planner_feasibility, prepare_trajectory

        np.save(args.output_dir / f"{prefix}{name}_geometric_path.npy", points.cpu().numpy())
        retimed, entry["retiming"] = prepare_trajectory(
            points.cpu().numpy(), planner_feasibility(planner), base.step_dt
        )
        points = torch.as_tensor(retimed, device=base.device, dtype=points.dtype)
    np.save(args.output_dir / f"{prefix}{name}_joint_targets.npy", points.cpu().numpy())
    return points


def run_step_guards(*guards):
    """Apply both retained-placement and current-motion checks after each simulator step."""
    for guard in guards:
        if guard[0] is not None:
            guard[0]()


def select_planning_target(planner, pose, rotation, joint_goal, names, tensor_args, entry):
    """Return a checked planning target and its matching planner method.

    Args:
        planner: Active MotionGen with the current world and locked joints.
        pose: Requested base-relative TCP pose.
        rotation: Requested TCP rotation as a SciPy Rotation.
        joint_goal: Optional reference joint target, already refined to the TCP pose.
        names: Ordered active joint names.
        tensor_args: Device and tensor conversion settings.
        entry: Report entry to receive reference-target provenance.
    """
    if joint_goal is None:
        return pose, planner.plan_single
    import numpy as np
    from scipy.spatial.transform import Rotation

    from curobo.types.state import JointState

    from isaaclab_arena_examples.g2_workcell.trajectory import planner_feasibility

    assert planner_feasibility(planner)(np.asarray(joint_goal)[None]), "Seeded joint goal is invalid"
    planning_goal = JointState.from_position(tensor_args.to_device([joint_goal]), joint_names=names)
    fk = planner.compute_kinematics(planning_goal).ee_pose
    fk_q = fk.quaternion[0].cpu().numpy()
    fk_rotation = Rotation.from_quat([*fk_q[1:], fk_q[0]])
    assert np.linalg.norm((fk.position - pose.position).cpu().numpy()) < 0.0005
    assert (fk_rotation.inv() * rotation).magnitude() < 0.005
    entry["planning_mode"] = "Locally refined reference posture, collision-checked joint-space plan"
    entry["joint_goal"] = np.asarray(joint_goal).tolist()
    return planning_goal, planner.plan_single_js


def prepare_scene(args):
    """Build the selected scene with matching measured collision geometry."""
    import numpy as np

    from isaaclab_arena_environments.g2_workbench_environments import G2WorkbenchEnvironmentCfg

    factory = args.scene_factory
    geometry = json.loads((args.geometry_dir / "geometry_report.json").read_text())
    meshes = {name: dict(np.load(args.geometry_dir / f"{name}_mesh.npz")) for _, name, *_ in factory.object_layout}
    geometry["bin"] = copy.deepcopy(geometry["bins"][args.destination])
    (args.output_dir / "effective_geometry.json").write_text(json.dumps(geometry, indent=2))
    description = factory.build(G2WorkbenchEnvironmentCfg(enable_cameras=args.enable_cameras))
    if getattr(args, "configure_description", None):
        args.configure_description(description)
    return description, geometry, meshes


def ensure_curobo_warp_compat():
    """Expose Warp torch interop under the namespace expected by cuRobo."""
    import sys
    import types

    import warp as wp

    if not hasattr(wp, "torch"):
        interop = types.ModuleType("warp.torch")
        for name in (
            "from_torch",
            "to_torch",
            "device_from_torch",
            "device_to_torch",
            "dtype_from_torch",
            "dtype_to_torch",
            "stream_from_torch",
            "stream_to_torch",
        ):
            setattr(interop, name, getattr(wp, name))
        wp.torch = interop
        sys.modules["warp.torch"] = interop


def run_session(args, sequence):
    """Execute the supplied workcell sequence and retain physical step evidence."""
    import numpy as np
    import torch
    import yaml
    from scipy.spatial.transform import Rotation

    ensure_curobo_warp_compat()

    from curobo.geom.types import Cuboid, Mesh, WorldConfig
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CuPose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    from isaaclab_arena.embodiments.g2.g2 import G2JointPositionActionsCfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    description, geometry, meshes = prepare_scene(args)
    description.embodiment.action_config = G2JointPositionActionsCfg()
    if args.enable_cameras:
        from isaaclab_arena.embodiments.g2.g2 import G2CameraCfg

        description.embodiment.camera_config = G2CameraCfg()
    env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
    report = {
        "scope": (
            "Physical grasp, lift, collision-checked carry and release; planning attachments do not alter simulation"
            " objects"
        ),
        "collision_model": (
            "Vendor G2 collision spheres, room cuboids, all scene source collision meshes; no cooked PhysX hull"
            " equivalence claim"
        ),
        "checks": [],
        "fk_parity": [],
    }
    path = args.output_dir / "reach_report.json"
    frames = []
    phase = ["initialize"]
    last_logged_phase = [None]
    video_writer = None

    def save():
        path.write_text(json.dumps(report, indent=2))

    try:
        base = env.unwrapped
        robot = base.scene["robot"]
        names = {
            "right": [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)],
            "left": [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)],
        }
        ids = {side: [robot.joint_names.index(n) for n in joint_names] for side, joint_names in names.items()}
        action = torch.zeros((1, 16), device=base.device)
        tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
        planners = {}
        attachments = {}
        step_guard = [None]
        sequence_guard = [None]
        assert str(base.device).startswith("cuda"), base.device
        report["simulation_device"] = str(base.device)
        report["planning_device"] = str(tensor_args.device)
        report["local_ik_backend"] = "curobo_motion_gen"
        if getattr(args, "record_video", False):
            from isaaclab_arena_examples.g2_workcell.cameras import ThreeViewWriter

            video_writer = ThreeViewWriter(args.output_dir, 1 / base.step_dt)

        def step(count=1):
            for _ in range(count):
                current = (args.tool, phase[0])
                if current != last_logged_phase[0]:
                    print("WORKCELL_PHASE", *current, "step", len(frames), flush=True)
                    last_logged_phase[0] = current
                before = robot.data.joint_pos.torch[0].cpu().numpy().copy()
                commanded = action[0].cpu().numpy().copy()
                _, _, terminated, truncated, _ = env.step(action)
                if video_writer is not None:
                    video_writer.append(base.scene)
                if getattr(args, "record_steps", False):
                    frames.append({
                        "phase": phase[0],
                        "tool": getattr(args, "tool", ""),
                        "action": commanded,
                        "q_before": before,
                        "q_after": robot.data.joint_pos.torch[0].cpu().numpy().copy(),
                        "objects": np.stack([object_pose(n)[0] for n in meshes]),
                        "object_quaternions": np.stack([object_pose(n)[1].as_quat() for n in meshes]),
                        "object_velocities": np.stack(
                            [base.scene[n].data.root_vel_w.torch[0].cpu().numpy().copy() for n in meshes]
                        ),
                        "tcp": np.concatenate([
                            base.scene[k].data.target_pos_w.torch[0, 0].cpu().numpy()
                            for k in ["ee_frame", "left_ee_frame"]
                        ]),
                        "tcp_quaternions": np.stack([
                            base.scene[k].data.target_quat_w.torch[0, 0].cpu().numpy().copy()
                            for k in ["ee_frame", "left_ee_frame"]
                        ]),
                    })
                run_step_guards(sequence_guard, step_guard)
                assert not bool(terminated[0] or truncated[0]), "Unexpected automatic reset"

        def reset():
            report["reset_count"] = report.get("reset_count", 0) + 1
            env.reset()
            action[0, :7] = robot.data.joint_pos.torch[0, ids["right"]]
            action[0, 8:15] = robot.data.joint_pos.torch[0, ids["left"]]
            action[0, [7, 15]] = 1
            if args.enable_cameras:
                # Prime GPU graphics with a real control step, then start the recorded episode
                # at its resulting physical state. No restored-state images enter the dataset.
                env.step(action)
                base.recorder_manager.reset([0])
                base.recorder_manager.record_post_reset([0])
                report["unrecorded_camera_warmup_control_steps"] = 1
            step(45)

        def object_pose(name):
            obj = base.scene[name]
            return obj.data.root_pos_w.torch[0].cpu().numpy(), Rotation.from_quat(
                obj.data.root_quat_w.torch[0].cpu().numpy()
            )

        def world(exclude=None):
            root = robot.data.root_pos_w.torch[0].cpu().numpy()
            quat = robot.data.root_quat_w.torch[0].cpu().numpy()
            assert abs(quat[3]) > 0.99999, "This benchmark assumes an unrotated robot base"
            boxes = []
            for name, centre, dims in (
                ("table", [0, 0, -0.025], [0.6, 1.2, 0.05]),
                ("floor", [0, 0, -0.80], [5, 5, 0.1]),
                ("cabinet", [1.35, 0, -0.27], [0.75, 1.9, 0.96]),
                ("shelf", [1.4, 0, 0.9], [0.55, 1.9, 0.05]),
                ("shelf_back", [1.7, 0, 0.65], [0.05, 1.9, 0.55]),
                ("back_wall", [1.8, 0, 0.75], [0.12, 5, 3]),
                ("left_wall", [-0.2, 2.5, 0.75], [4, 0.12, 3]),
                ("sideboard", [0.9, 1.95, -0.30], [1.3, 0.55, 0.9]),
            ):
                boxes.append(
                    Cuboid(
                        name=name,
                        pose=[*(np.array(centre) - root).tolist(), 1, 0, 0, 0],
                        dims=dims,
                    )
                )
            obstacles = []
            for name, mesh in meshes.items():
                if name == exclude:
                    continue
                position, rotation = object_pose(name)
                q = rotation.as_quat()
                obstacles.append(
                    Mesh(
                        name=name,
                        pose=[*(position - root).tolist(), q[3], *q[:3]],
                        vertices=mesh["vertices"].tolist(),
                        faces=mesh["faces"].tolist(),
                    )
                )
            return WorldConfig(cuboid=boxes, mesh=obstacles)

        def planner_for(side):
            setup_started = time.perf_counter()
            created = side not in planners
            state = dict(zip(robot.joint_names, robot.data.joint_pos.torch[0].tolist()))
            if side not in planners:
                config = copy.deepcopy(yaml.safe_load(args.robot_yaml.read_text())["robot_cfg"])
                kin = config["kinematics"]
                kin.update(
                    use_usd_kinematics=False,
                    urdf_path=str(args.robot_urdf.resolve()),
                    asset_root_path=str(args.robot_urdf.resolve().parent),
                    ee_link=f"gripper_{side[0]}_center_link",
                    link_names=[],
                    mesh_link_names=[],
                )
                kin["collision_link_names"] = [n for n in kin["collision_link_names"] if "attached_object" not in n]
                kin["collision_spheres"] = {
                    n: s for n, s in kin["collision_spheres"].items() if "attached_object" not in n
                }
                kin["extra_links"] = {}
                kin["extra_collision_spheres"] = {}
                if getattr(args, "mode", "") in ("pick_place", "sequence"):
                    configure_payload(kin, side, getattr(args, "payload_capacities", {}).get(side, 384))
                movable = movable_joints(args.robot_urdf, [*kin["collision_link_names"], kin["ee_link"]])
                kin["lock_joints"] = {n: state.get(n, 0.0) for n in movable if n not in names[side]}
                kin["cspace"] = {
                    "joint_names": names[side],
                    "retract_config": [state[n] for n in names[side]],
                    "null_space_weight": [1.0] * 7,
                    "cspace_distance_weight": [1.0] * 7,
                    "max_jerk": 100.0,
                    "max_acceleration": 5.0,
                }
                save_planner_config(args, side, config)
                report.setdefault("planner_models", []).append(str(args.robot_urdf))
                planner = MotionGen(
                    MotionGenConfig.load_from_robot_config(
                        config,
                        world(),
                        tensor_args=tensor_args,
                        use_cuda_graph=False,
                        interpolation_dt=base.step_dt,
                        num_ik_seeds=32,
                        num_trajopt_seeds=12,
                        collision_cache={"obb": 16, "mesh": 8},
                        collision_activation_distance=0.005,
                        position_threshold=0.003,
                        rotation_threshold=0.03,
                    )
                )
                planners[side] = planner, config
            planner, config = planners[side]
            planner.update_locked_joints(
                {n: state.get(n, 0.0) for n in config["kinematics"]["lock_joints"]},
                config,
            )
            restore_payload(planner, attachments, side)
            current = JointState.from_position(
                tensor_args.to_device([[state[n] for n in names[side]]]),
                joint_names=names[side],
            )
            fk = planner.compute_kinematics(current).ee_pose
            sensor = base.scene["ee_frame" if side == "right" else "left_ee_frame"]
            sim_pos = sensor.data.target_pos_w.torch[0, 0].cpu().numpy() - robot.data.root_pos_w.torch[0].cpu().numpy()
            sim_rot = Rotation.from_quat(sensor.data.target_quat_w.torch[0, 0].cpu().numpy())
            q = fk.quaternion[0].cpu().numpy()
            fk_rot = Rotation.from_quat([*q[1:], q[0]])
            position_error = float(np.linalg.norm(fk.position[0].cpu().numpy() - sim_pos))
            rotation_error = float((fk_rot.inv() * sim_rot).magnitude())
            report["fk_parity"].append({
                "side": side,
                "position_m": position_error,
                "rotation_rad": rotation_error,
                "fk_minus_sim_xyz_m": (fk.position[0].cpu().numpy() - sim_pos).tolist(),
                "root_world_m": robot.data.root_pos_w.torch[0].cpu().numpy().tolist(),
                "base_body_world_m": (
                    robot.data.body_pos_w.torch[0, robot.body_names.index("base_link")].cpu().numpy().tolist()
                ),
            })
            assert position_error < 0.002 and rotation_error < 0.05, "Planner/simulator FK mismatch"
            report.setdefault("planner_setup_timings", []).append(
                {"side": side, "created": created, "wall_seconds": time.perf_counter() - setup_started}
            )
            return planner

        def goal(position, rotation):
            root = robot.data.root_pos_w.torch[0].cpu().numpy()
            q = rotation.as_quat()
            return CuPose(
                position=tensor_args.to_device([position - root]),
                quaternion=tensor_args.to_device([[q[3], *q[:3]]]),
            )

        def check_target(planner, side, name, position, rotation, execute, exclude=None, joint_goal=None):
            check_started = time.perf_counter()
            phase[0] = name
            planner.update_world(world(exclude))
            entry = {
                "name": name,
                "tool": getattr(args, "tool", ""),
                "side": side,
                "position_world_m": position.tolist(),
                "quaternion_xyzw": rotation.as_quat().tolist(),
                "executed": False,
                "excluded_contact_object": exclude,
            }
            report["checks"].append(entry)
            measured = robot.data.joint_pos.torch[:, ids[side]].to("cuda:0")
            limits = planner.kinematics.get_joint_limits().position
            bounded = measured.clamp(min=limits[0], max=limits[1])
            assert torch.max(abs(bounded - measured)).item() <= 1e-4, "Measured joint exceeds hard limits"
            start = JointState.from_position(bounded, joint_names=names[side])
            pose = goal(position, rotation)
            # Match stack bowls: MotionGen owns IK retries for executed segments.
            # A separate one-shot IK query must not veto its collision-checked plan.
            if not execute:
                result = planner.solve_ik(pose, retract_config=bounded, num_seeds=32, return_seeds=1)
                entry["ik_success"] = bool(result.success.any().item())
                entry["ik_position_error_m"] = float(result.position_error.min().item())
                entry["ik_rotation_error_rad"] = float(result.rotation_error.min().item())
            entry["query_setup_wall_seconds"] = time.perf_counter() - check_started
            if execute:
                planning_started = time.perf_counter()
                planning_goal, plan = select_planning_target(
                    planner, pose, rotation, joint_goal, names[side], tensor_args, entry
                )
                trajectory = plan(
                    start,
                    planning_goal,
                    MotionGenPlanConfig(
                        max_attempts=5,
                        timeout=30.0,
                        pose_cost_metric=(
                            PoseCostMetric(
                                hold_partial_pose=True,
                                hold_vec_weight=tensor_args.to_device([1, 1, 1, 1, 1, 0]),
                                project_to_goal_frame=False,
                            )
                            if name in {"approach", "lift", "lower_into_bin", "release_clearance"}
                            else None
                        ),
                        enable_graph=True,
                        enable_graph_attempt=2,
                        time_dilation_factor=(
                            0.25 if getattr(args, "tool", "") == "drill" and name == "carry_to_bin" else 0.5
                        ),
                        enable_opt=name not in getattr(args, "graph_phases", []),
                    ),
                )
                entry["planning_wall_seconds"] = time.perf_counter() - planning_started
                entry["plan_status"] = str(trajectory.status)
                entry["plan_success"] = bool(trajectory.success.item())
                if entry["plan_success"]:
                    execution_started = time.perf_counter()
                    tracked = {n: object_pose(n)[0].copy() for n in meshes}
                    sensor = base.scene["ee_frame" if side == "right" else "left_ee_frame"]
                    offset = 0 if side == "right" else 8
                    points = trajectory.get_interpolated_plan().get_ordered_joint_state(names[side]).position
                    points = prepare_execution_points(args, planner, points, name, base, entry)
                    for waypoint in points:
                        action[0, offset : offset + 7] = waypoint.to(base.device)
                        step()
                    step(10)
                    entry["execution_wall_seconds"] = time.perf_counter() - execution_started
                    entry["execution_control_steps"] = len(points) + 10
                    actual_pos = sensor.data.target_pos_w.torch[0, 0].cpu().numpy()
                    actual_rot = Rotation.from_quat(sensor.data.target_quat_w.torch[0, 0].cpu().numpy())
                    entry["executed"] = True
                    if joint_goal is not None:
                        actual_joints = robot.data.joint_pos.torch[0, ids[side]].cpu().numpy()
                        entry["joint_goal_error_rad"] = float(np.max(abs(actual_joints - joint_goal)))
                        assert entry["joint_goal_error_rad"] < 0.02, "Reference posture was not reached"
                    entry["actual_position_world_m"] = actual_pos.tolist()
                    entry["position_error_m"] = float(np.linalg.norm(actual_pos - position))
                    entry["rotation_error_deg"] = float(np.degrees((actual_rot.inv() * rotation).magnitude()))
                    entry["object_displacement_m"] = {
                        n: float(np.linalg.norm(object_pose(n)[0] - p)) for n, p in tracked.items()
                    }
                    entry["pose_pass"] = entry["position_error_m"] < getattr(
                        args, "position_tolerance", 0.01
                    ) and entry["rotation_error_deg"] < getattr(args, "rotation_tolerance", 5)
                    entry["objects_undisturbed"] = max(entry["object_displacement_m"].values()) < getattr(
                        args, "object_tolerance", 0.005
                    )
            save()
            print("CHECK", json.dumps(entry), flush=True)
            return entry

        from types import SimpleNamespace

        sequence(
            SimpleNamespace(
                args=args,
                base=base,
                robot=robot,
                action=action,
                step=step,
                phase=phase,
                ids=ids,
                names=names,
                meshes=meshes,
                planners=planners,
                attachments=attachments,
                step_guard=step_guard,
                sequence_guard=sequence_guard,
                world=world,
                goal=goal,
                geometry=geometry,
                report=report,
                reset=reset,
                planner_for=planner_for,
                object_pose=object_pose,
                check_target=check_target,
                save=save,
            )
        )
        report["complete"] = True
        report["step_dt"] = base.step_dt
        save()
        print("REACH_COMPLETE", flush=True)
    finally:
        save()
        if frames:
            np.savez_compressed(
                args.output_dir / "step_trace.npz",
                phase=np.array([f["phase"] for f in frames]),
                tool=np.array([f["tool"] for f in frames]),
                actions=np.stack([f["action"] for f in frames]),
                q_before=np.stack([f["q_before"] for f in frames]),
                q_after=np.stack([f["q_after"] for f in frames]),
                object_positions=np.stack([f["objects"] for f in frames]),
                object_quaternions=np.stack([f["object_quaternions"] for f in frames]),
                object_velocities=np.stack([f["object_velocities"] for f in frames]),
                tcp_positions=np.stack([f["tcp"] for f in frames]),
                tcp_quaternions=np.stack([f["tcp_quaternions"] for f in frames]),
                object_names=np.array(list(meshes)),
                joint_names=np.array(robot.joint_names),
                step_dt=base.step_dt,
            )
        if video_writer is not None:
            video_writer.close()
        env.close()
