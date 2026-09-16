# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plan G2 bowl stacking with cuRobo and record physically executed episodes."""

import argparse
import json
import math
import shutil
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from g2_dataset_io import sample_bowls

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher


def _planning_joint_names(urdf_path: Path, link_names: list[str]) -> set[str]:
    """Get independent movable joints in the requested URDF link chains.

    Args:
        urdf_path: Robot description file.
        link_names: End-effector and collision links included in the planner.

    Returns:
        Joint names excluding fixed and mimic joints.
    """
    child_joints = {
        joint.find("child").attrib["link"]: joint for joint in ET.parse(urdf_path).getroot().findall("joint")
    }
    movable = set()
    for link in link_names:
        while link in child_joints:
            joint = child_joints[link]
            if joint.attrib["type"] != "fixed" and joint.find("mimic") is None:
                movable.add(joint.attrib["name"])
            link = joint.find("parent").attrib["link"]
    return movable


def _close_video(writer):
    """Finalize an optional live video writer."""
    if writer is not None:
        writer.close()


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--robot_yaml", type=Path, required=True)
    parser.add_argument("--robot_urdf", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--table_height_m", type=float, default=0.75)
    parser.add_argument("--bowl_xy_noise_m", type=float, default=0.0)
    # Avoid AppLauncher's reserved `video` flag: we encode a sensor, not its viewport recorder.
    parser.add_argument("--record_video", action="store_true", help="Record the live overview camera to MP4")
    parser.add_argument("--rim_offset", type=float, default=0.065)
    parser.add_argument("--grasp_height", type=float, default=0.065)
    parser.add_argument("--grasp_yaw", type=float, default=0.0, help="Top-down rim grasp yaw in degrees")
    parser.add_argument("--stack_release_offset", type=float, default=0.070, help="Bowl centre separation at release")
    args = parser.parse_args()
    assert args.attempts > 0
    assert not args.bowl_xy_noise_m or args.attempts == 1, "Perturbed collection uses one seeded layout per process"
    args.bowl_positions = sample_bowls(args.seed, args.bowl_xy_noise_m)
    assert not args.record_video or args.enable_cameras, "--record_video requires --enable_cameras"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.robot_yaml, args.output_dir / "source_robot.yaml")
    shutil.copyfile(args.robot_urdf, args.output_dir / "robot.urdf")
    shutil.copyfile(__file__, args.output_dir / "collector_source.py")
    launcher = get_app_launcher(args)
    try:
        collect(args)
    finally:
        launcher.app.close()


def collect(args: argparse.Namespace):
    import torch
    import yaml

    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CuPose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from isaaclab.managers import DatasetExportMode, TerminationTermCfg
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.embodiments.g2.g2 import G2CollectionCameraCfg, G2JointPositionActionsCfg
    from isaaclab_arena.embodiments.g2.recorders import core_recorder_cfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg
    from isaaclab_arena_environments.g2_stack_bowls_environment import (
        G2StackBowlsEnvironment,
        G2StackBowlsEnvironmentCfg,
    )

    description = G2StackBowlsEnvironment().build(
        G2StackBowlsEnvironmentCfg(
            enable_cameras=args.enable_cameras,
            hdr=None,
            episode_length_s=600,
            table_height_m=args.table_height_m,
            bowl_positions=args.bowl_positions,
        )
    )
    description.embodiment.action_config = G2JointPositionActionsCfg()
    if args.enable_cameras:
        description.embodiment.camera_config = G2CollectionCameraCfg()
    original_callback = description.env_cfg_callback
    success_term = None
    failure_terms = {}

    def configure(cfg):
        nonlocal success_term
        cfg = original_callback(cfg)
        success_term = cfg.terminations.success
        # Evaluate success explicitly after opening the gripper and allowing the stack to settle.
        for name, term in vars(cfg.terminations).items():
            if isinstance(term, TerminationTermCfg):
                if name != "success" and not term.time_out:
                    failure_terms[name] = term
                setattr(cfg.terminations, name, None)
        cfg.recorders = ArenaEnvRecorderManagerCfg(
            dataset_export_dir_path=str(args.output_dir),
            dataset_filename="episodes",
            dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES,
            export_in_record_pre_reset=False,
        )
        if not args.enable_cameras:
            cfg.recorders.record_pre_step_flat_camera_observations = None
        cfg.recorders.g2_core = core_recorder_cfg()
        return cfg

    description.env_cfg_callback = configure
    args.num_envs = 1
    env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
    base = env.unwrapped
    robot = base.scene["robot"]
    action_config_path = "isaaclab_arena.embodiments.g2.g2:G2JointPositionActionsCfg"
    base.cfg.get_ep_meta = lambda: {
        "env_name": description.name,
        "action_config": action_config_path,
        "sim_args": {
            "dt": base.cfg.sim.dt,
            "decimation": base.cfg.decimation,
            "render_interval": base.cfg.sim.render_interval,
            "num_envs": 1,
        },
        "joint_names": robot.joint_names,
        "table_height_m": args.table_height_m,
        "bowl_positions": args.bowl_positions,
        "pose_convention": "right xyz+xyzw, left xyz+xyzw; metres; eef_pose is robot-base relative",
    }
    right = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
    left = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
    arm_names = right + left
    arm_ids = [robot.joint_names.index(name) for name in arm_names]
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    report = {
        "planner": "cuRobo MotionGen",
        "joint_names": robot.joint_names,
        "action_config": action_config_path,
        "step_dt": base.step_dt,
        "action_schema": "right_joint_positions[7],right_gripper,left_joint_positions[7],left_gripper",
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "attempts": [],
    }
    planners = {}
    attached_spheres = {}
    action = torch.zeros((1, 16), device=base.device)
    video_writer = None

    def step(count=1):
        for _ in range(count):
            _, _, terminated, truncated, _ = env.step(action)
            if video_writer is not None:
                frame = base.scene["overview_camera"].data.output["rgb"].torch[0, ..., :3].cpu().numpy()
                video_writer.append_data(frame)
            assert not bool(terminated[0] or truncated[0]), "Environment reset during trajectory execution"
            # Advance the sequential task's state without triggering an automatic reset.
            success_term.func(base, **success_term.params)
            for name, term in failure_terms.items():
                assert not bool(term.func(base, **term.params)[0]), f"Task failure: {name}"

    def position(name):
        return base.scene[name].data.root_pos_w.torch[0].clone()

    def world(excluded=()):
        root = robot.data.root_pos_w.torch[0].cpu().tolist()

        def box(name, center, dims):
            return Cuboid(name=name, pose=[*[center[i] - root[i] for i in range(3)], 1, 0, 0, 0], dims=dims)

        floor_z = -args.table_height_m
        boxes = [
            box("table", [0, 0, -0.025], [0.60, 1.20, 0.05]),
            box("floor", [0, 0, floor_z - 0.05], [5, 5, 0.1]),
            box("cabinet_and_counter", [1.35, 0, floor_z + 0.48], [0.75, 1.9, 0.96]),
            box("shelf", [1.4, 0, floor_z + 1.65], [0.55, 1.9, 0.05]),
            box("shelf_back", [1.7, 0, floor_z + 1.40], [0.05, 1.9, 0.55]),
            box("back_wall", [1.8, 0, floor_z + 1.5], [0.12, 5, 3]),
            box("left_wall", [-0.2, 2.5, floor_z + 1.5], [4, 0.12, 3]),
            box("sideboard", [0.9, 1.95, floor_z + 0.45], [1.3, 0.55, 0.9]),
        ]
        for name in ("bowl_1", "bowl_2", "bowl_3"):
            if name not in excluded:
                boxes.append(box(name, position(name).cpu().tolist(), [0.158, 0.158, 0.083]))
        return WorldConfig(cuboid=boxes)

    def get_planner(side):
        active = right if side == "right" else left
        state = dict(zip(robot.joint_names, robot.data.joint_pos.torch[0].cpu().tolist()))
        if side not in planners:
            config = yaml.safe_load(args.robot_yaml.read_text())["robot_cfg"]
            kin = config["kinematics"]
            kin.update(
                use_usd_kinematics=False,
                urdf_path=str(args.robot_urdf),
                asset_root_path=str(args.robot_urdf.parent),
                ee_link=f"gripper_{'r' if side == 'right' else 'l'}_center_link",
                link_names=[],
                mesh_link_names=[],
            )
            # The planner must use the simulator's actual waist and inactive-arm posture.
            movable = _planning_joint_names(args.robot_urdf, [*kin["collision_link_names"], kin["ee_link"]])
            kin["lock_joints"] = {name: state.get(name, 0.0) for name in movable if name not in active}
            kin["cspace"] = {
                "joint_names": active,
                "retract_config": [state[name] for name in active],
                "null_space_weight": [1.0] * 7,
                "cspace_distance_weight": [1.0] * 7,
                "max_jerk": 100.0,
                "max_acceleration": 5.0,
            }
            kin["extra_links"] = {
                "attached_object": {
                    "parent_link_name": kin["ee_link"],
                    "link_name": "attached_object",
                    "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
                    "joint_type": "FIXED",
                    "joint_name": "attach_joint",
                }
            }
            kin["extra_collision_spheres"] = {"attached_object": 17}
            kin["collision_link_names"] = [
                name for name in kin["collision_link_names"] if name != "left_attached_object"
            ]
            kin["collision_spheres"].pop("left_attached_object", None)
            config_path = args.output_dir / f"{side}_planner.yaml"
            config_path.write_text(yaml.safe_dump({"robot_cfg": config}))
            planner = MotionGen(
                MotionGenConfig.load_from_robot_config(
                    config,
                    world(),
                    tensor_args=tensor_args,
                    use_cuda_graph=False,
                    interpolation_dt=base.step_dt,
                    num_ik_seeds=32,
                    num_trajopt_seeds=4,
                    collision_cache={"obb": 20},
                    collision_activation_distance=0.005,
                )
            )
            planners[side] = (planner, config)
        planner, config = planners[side]
        locked = {name: state.get(name, 0.0) for name in config["kinematics"]["lock_joints"]}
        planner.update_locked_joints(locked, config)
        if side in attached_spheres:
            planner.attach_spheres_to_robot(sphere_tensor=attached_spheres[side])
        # Check URDF/TCP parity before trusting plans from a vendor configuration.
        current = JointState.from_position(
            tensor_args.to_device([[state[name] for name in active]]), joint_names=active
        )
        fk = planner.compute_kinematics(current).ee_pose.position[0].to(base.device)
        sensor = base.scene["ee_frame" if side == "right" else "left_ee_frame"]
        actual = sensor.data.target_pos_w.torch[0, 0] - robot.data.root_pos_w.torch[0]
        error = torch.norm(fk - actual).item()
        print(f"FK parity {side}: {error:.5f} m", flush=True)
        assert error < 0.01, f"G2 planner/simulator kinematics differ by {error:.4f} m"
        return planner

    def move(side, target, quat, excluded=()):
        planner = get_planner(side)
        planner.update_world(world(excluded))
        names = right if side == "right" else left
        ids = [robot.joint_names.index(name) for name in names]
        measured = robot.data.joint_pos.torch[:, ids].to("cuda:0")
        limits = planner.kinematics.get_joint_limits().position
        bounded = measured.clamp(min=limits[0], max=limits[1])
        violation = torch.max(torch.abs(measured - bounded)).item()
        # PhysX can exceed a hard stop by a few tens of microradians. Only sanitize
        # this numerical tolerance in the planning query; never overwrite physical state.
        assert violation <= 1e-4, f"Measured {side} joint limit violation: {violation:.6f} rad"
        if violation:
            print(f"Planning start limit tolerance {side}: {violation:.8f} rad", flush=True)
            report.setdefault("start_limit_tolerances", []).append({"side": side, "max_rad": violation})
        start = JointState.from_position(bounded, joint_names=names)
        goal = CuPose(
            position=(target - robot.data.root_pos_w.torch[0]).to("cuda:0").view(1, 3),
            quaternion=tensor_args.to_device([quat]),
        )
        result = planner.plan_single(
            start,
            goal,
            MotionGenPlanConfig(max_attempts=5, enable_graph=True, enable_graph_attempt=2, time_dilation_factor=0.5),
        )
        print(f"PLAN {side}: {target.tolist()} -> {result.status}, success={result.success.tolist()}", flush=True)
        assert bool(result.success.item()), f"Motion planning failed: {result.status}"
        trajectory = result.get_interpolated_plan().get_ordered_joint_state(names).position
        offset = 0 if side == "right" else 8
        for waypoint in trajectory:
            action[0, offset : offset + 7] = waypoint.to(base.device)
            step()
        step(10)
        sensor = base.scene["ee_frame" if side == "right" else "left_ee_frame"]
        error = torch.norm(sensor.data.target_pos_w.torch[0, 0] - target).item()
        report.setdefault("executed_targets", []).append(
            {"side": side, "target": target.cpu().tolist(), "error_m": error}
        )
        assert error < 0.025, f"Physical execution did not reach target: {error:.4f} m"

    try:
        for attempt in range(args.attempts):
            env.reset()
            for planner, _ in planners.values():
                planner.detach_spheres_from_robot()
            attached_spheres.clear()
            action[0, :7] = robot.data.joint_pos.torch[0, arm_ids[:7]]
            action[0, 8:15] = robot.data.joint_pos.torch[0, arm_ids[7:]]
            action[0, [7, 15]] = 1
            entry = {"attempt": attempt, "success": False, "phase": "settle"}
            report["attempts"].append(entry)
            if args.record_video:
                import imageio.v2 as imageio

                entry["video"] = f"attempt_{attempt:03d}.mp4"
                video_writer = imageio.get_writer(
                    str(args.output_dir / entry["video"]), fps=1.0 / base.step_dt, codec="libx264", quality=8
                )
            try:
                step(30)
                for side, source, destination in (("right", "bowl_1", "bowl_2"), ("left", "bowl_3", "bowl_1")):
                    entry["phase"] = f"{side}: approach {source}"
                    initial = position(source)
                    grasp = initial.clone()
                    yaw = math.radians(args.grasp_yaw)
                    grasp[0] -= args.rim_offset * math.cos(yaw)
                    grasp[1] -= args.rim_offset * math.sin(yaw)
                    grasp[2] = args.grasp_height
                    quat = [0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0]
                    up = torch.tensor([0, 0, 0.18], device=base.device)
                    move(side, grasp + up, quat, excluded=(source,))
                    # Intentional gripper/object contact; arm links still avoid the table and other bowls.
                    planner = get_planner(side)
                    move(side, grasp, quat, excluded=(source,))
                    action[0, 7 if side == "right" else 15] = -1
                    step(30)
                    entry["phase"] = f"{side}: lift {source}"
                    move(side, grasp + up, quat, excluded=(source,))
                    lift = (position(source)[2] - initial[2]).item()
                    entry[f"{source}_lift_m"] = lift
                    assert lift > 0.10, f"Grasp failed: {source} lifted only {lift:.4f} m"
                    sensor = base.scene["ee_frame" if side == "right" else "left_ee_frame"]
                    # A single enclosing sphere extends below the bowl and falsely hits the
                    # bottom bowl during the second placement. Use an overlapping ring plus
                    # centre sphere: conservative horizontally, only 4.7 cm in half-height.
                    offsets = [[0.0, 0.0, 0.0]] + [
                        [0.06 * math.cos(i * math.pi / 8), 0.06 * math.sin(i * math.pi / 8), 0.0] for i in range(16)
                    ]
                    bowl_offsets = torch.tensor(offsets, device=base.device)
                    centers = position(source) + quat_apply(
                        base.scene[source].data.root_quat_w.torch[0].expand(17, 4), bowl_offsets
                    )
                    local_centers = quat_apply_inverse(
                        sensor.data.target_quat_w.torch[0, 0].expand(17, 4),
                        centers - sensor.data.target_pos_w.torch[0, 0],
                    )
                    spheres = torch.cat((local_centers, local_centers.new_full((17, 1), 0.047)), dim=1)
                    attached_spheres[side] = spheres.to("cuda:0")
                    planner.attach_spheres_to_robot(sphere_tensor=attached_spheres[side])
                    entry["phase"] = f"{side}: place {source}"
                    delta = position(destination) - position(source)
                    delta[2] += args.stack_release_offset
                    place = grasp + up + delta
                    transfer = place.clone()
                    transfer[2] = max((grasp + up)[2].item(), place[2].item() + 0.02)
                    move(side, transfer, quat, excluded=(source, destination))
                    move(side, place, quat, excluded=(source, destination))
                    action[0, 7 if side == "right" else 15] = 1
                    step(20)
                    planner.detach_spheres_from_robot()
                    attached_spheres.pop(side)
                    move(side, transfer, quat, excluded=(source, destination))
                    step(30)
                    # Clear the shared central placement region before switching arms.
                    move(side, grasp + up, quat)
                entry["phase"] = "verify stack"
                for _ in range(10):
                    step()
                    assert bool(success_term.func(base, **success_term.params)[0]), "Task success predicate is false"
                    heights = [position(name)[2].item() for name in ("bowl_2", "bowl_1", "bowl_3")]
                    assert heights[0] < heights[1] < heights[2], "Bowls are not stacked in the intended order"
                entry["success"] = True
            except Exception as exc:
                entry["error"] = str(exc)
                traceback.print_exc()
            _close_video(video_writer)
            video_writer = None
            entry["final_bowl_positions"] = {
                name: position(name).cpu().tolist() for name in ("bowl_1", "bowl_2", "bowl_3")
            }
            base.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            base.recorder_manager.set_success_to_episodes([0], torch.tensor([[entry["success"]]], device=base.device))
            base.recorder_manager.export_episodes([0])
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
            print("ATTEMPT_RESULT", json.dumps(entry), flush=True)
    finally:
        _close_video(video_writer)
        env.close()


if __name__ == "__main__":
    main()
