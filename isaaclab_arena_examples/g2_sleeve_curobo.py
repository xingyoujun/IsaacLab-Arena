# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect physically executed G2 sleeve demonstrations with cuRobo MotionGen."""

import argparse
import json
import random
import shutil
import traceback
from pathlib import Path

from g2_stack_bowls_curobo import _ensure_curobo_warp_compat, _planning_joint_names

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--headless", action="store_true", help="Run without a viewer (also the GA default)")
    parser.add_argument("--robot_yaml", type=Path, required=True)
    parser.add_argument("--robot_urdf", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--table_height_m", type=float, default=0.75)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--grasp_x", type=float, default=0.0)
    parser.add_argument("--grasp_z", type=float, default=0.115)
    parser.add_argument("--sleeve_x", type=float, default=-0.10)
    parser.add_argument("--sleeve_y", type=float, default=0.0)
    parser.add_argument("--peg_x", type=float, default=0.05)
    parser.add_argument("--peg_y", type=float, default=-0.22)
    parser.add_argument("--xy_noise_m", type=float, default=0.0)
    parser.add_argument("--probe_grasp", action="store_true", help="Stop after lift; never export as successful")
    args = parser.parse_args()
    assert args.attempts > 0
    assert not args.record_video or args.enable_cameras, "--record_video requires --enable_cameras"
    assert 0 <= args.xy_noise_m <= 0.02
    rng = random.Random(args.seed)
    for key in ("peg_x", "peg_y", "sleeve_x", "sleeve_y"):
        setattr(args, key, getattr(args, key) + rng.uniform(-args.xy_noise_m, args.xy_noise_m))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for source, name in (
        (args.robot_yaml, "source_robot.yaml"),
        (args.robot_urdf, "robot.urdf"),
        (Path(__file__), "collector_source.py"),
    ):
        shutil.copyfile(source, args.output_dir / name)
    source_root = Path(__file__).resolve().parents[1]
    for relative in (
        "isaaclab_arena_environments/g2_sleeve_environment.py",
        "isaaclab_arena_environments/g2_workbench_environments.py",
        "isaaclab_arena/tasks/sleeve_task.py",
        "isaaclab_arena/embodiments/g2/g2.py",
        "isaaclab_arena/embodiments/g2/recorders.py",
        "isaaclab_arena_examples/g2_stack_bowls_curobo.py",
    ):
        target = args.output_dir / "provenance" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, target)
    with SimulationAppContext(args):
        assert collect(args), "No successful sleeve demonstration; inspect report.json and failed raw"


def _collision_world(base, table_height_m, excluded=()):
    from curobo.geom.types import Cuboid, WorldConfig

    robot = base.scene["robot"]

    def position(name):
        return base.scene[name].data.root_pos_w.torch[0]

    root = robot.data.root_pos_w.torch[0].cpu().tolist()

    def box(name, center, dims):
        return Cuboid(name=name, pose=[*[center[i] - root[i] for i in range(3)], 1, 0, 0, 0], dims=dims)

    floor_z = -table_height_m
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
    if "peg" not in excluded:
        center = position("peg").cpu().tolist()
        center[2] += 0.075
        boxes.append(box("peg", center, [0.026, 0.026, 0.15]))
    if "sleeve" not in excluded:
        center = position("sleeve").cpu().tolist()
        center[2] += 0.033
        boxes.append(box("sleeve", center, [0.12, 0.12, 0.084]))
    return WorldConfig(cuboid=boxes)


def _open_videos(args, attempt, step_dt):
    """Open aligned native-camera and overview streams for an optional live recording."""
    import imageio.v2 as imageio

    if not args.record_video:
        return {}
    return {
        camera: imageio.get_writer(
            str(args.output_dir / f"attempt_{attempt:03d}_{camera}.mp4"),
            fps=1.0 / step_dt,
            codec="libx264",
            quality=8,
        )
        for camera in ("overview_camera", "head_camera", "left_wrist_camera", "right_wrist_camera")
    }


def _close_videos(writers):
    """Finalize all open videos and clear their handles."""
    for writer in writers.values():
        writer.close()
    writers.clear()


def _reach_target(move_once, side, target, quat, excluded, tolerance):
    """Remeasure the wrist-dependent TCP model offset for up to three planned corrections."""
    for _ in range(3):
        error = move_once(side, target, quat, excluded)
        assert error < 0.025, f"Physical execution deviated by {error:.4f} m"
        if error < tolerance:
            return
    assert error < tolerance, f"Physical execution did not reach target: {error:.4f} m"


def collect(args: argparse.Namespace):
    _ensure_curobo_warp_compat()

    import torch
    import yaml

    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CuPose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from isaaclab.managers import DatasetExportMode, TerminationTermCfg
    from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply_inverse, subtract_frame_transforms

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.embodiments.g2.g2 import G2CameraCfg, G2CollectionCameraCfg, G2JointPositionActionsCfg
    from isaaclab_arena.embodiments.g2.recorders import G2CollectionSuccessTerm, core_recorder_cfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.tasks.sleeve_task import peg_is_inserted
    from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg
    from isaaclab_arena_environments.g2_sleeve_environment import G2SleeveEnvironment, G2SleeveEnvironmentCfg

    description = G2SleeveEnvironment().build(
        G2SleeveEnvironmentCfg(
            enable_cameras=args.enable_cameras,
            episode_length_s=600,
            table_height_m=args.table_height_m,
            sleeve_xy=[args.sleeve_x, args.sleeve_y],
            peg_xy=[args.peg_x, args.peg_y],
        )
    )
    description.embodiment.action_config = G2JointPositionActionsCfg()
    if args.enable_cameras:
        cameras = G2CollectionCameraCfg()
        cameras.head_camera = G2CameraCfg().head_camera
        description.embodiment.camera_config = cameras
    original_callback = description.env_cfg_callback
    failure_terms = {}

    def configure(cfg):
        cfg = original_callback(cfg)
        cfg.terminations.success.func = G2CollectionSuccessTerm
        # Evaluate success explicitly after release and physical settling.
        for name, term in vars(cfg.terminations).items():
            if isinstance(term, TerminationTermCfg) and name != "success":
                if not term.time_out:
                    failure_terms[name] = term
                setattr(cfg.terminations, name, None)
        cfg.recorders = ArenaEnvRecorderManagerCfg(
            dataset_export_dir_path=str(args.output_dir),
            dataset_filename="episodes",
            dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES,
            export_in_record_pre_reset=False,
        )
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
        "task": "Insert the cylindrical peg into the fixed upright sleeve",
        "pose_convention": "right xyz+xyzw, left xyz+xyzw; metres; eef_pose is robot-base relative",
    }
    right = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
    left = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
    arm_names = right + left
    arm_ids = [robot.joint_names.index(name) for name in arm_names]
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    report = {
        "planner": "cuRobo MotionGen",
        "task_variant": "movable_peg_fixed_sleeve",
        "joint_names": robot.joint_names,
        "action_config": action_config_path,
        "step_dt": base.step_dt,
        "action_schema": "right_joint_positions[7],right_gripper,left_joint_positions[7],left_gripper",
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "attempts": [],
    }
    planners = {}
    attached_spheres = {}
    planner_offsets = {}
    action = torch.zeros((1, 16), device=base.device)
    video_writers = {}
    frame_index = 0
    phase = "initialize"

    def step(count=1):
        nonlocal frame_index
        for _ in range(count):
            for camera, writer in video_writers.items():
                frame = base.scene[camera].data.output["rgb"].torch[0, ..., :3].cpu().numpy()
                writer.append_data(frame)
            _, _, terminated, truncated, _ = env.step(action)
            frame_index += 1
            assert not bool(terminated[0] or truncated[0]), "Environment reset during trajectory execution"
            for name, term in failure_terms.items():
                assert not bool(term.func(base, **term.params)[0]), f"Task failure: {name}"

    def set_phase(name):
        nonlocal phase
        phase = name
        report.setdefault("phases", []).append({"frame": frame_index, "name": name})
        print("PHASE", name, flush=True)

    def position(name):
        return base.scene[name].data.root_pos_w.torch[0].clone()

    def world(excluded=()):
        return _collision_world(base, args.table_height_m, excluded)

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
                    collision_activation_distance=0.002,
                    position_threshold=0.0005,
                    rotation_threshold=0.005,
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
        planner_offsets[side] = fk - actual
        print(f"FK parity {side}: {error:.5f} m", flush=True)
        assert error < 0.01, f"G2 planner/simulator kinematics differ by {error:.4f} m"
        return planner

    def move_once(side, target, quat, excluded=()):
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
        start = JointState.from_position(bounded, joint_names=names)
        goal = CuPose(
            position=(target - robot.data.root_pos_w.torch[0] + planner_offsets[side]).to("cuda:0").view(1, 3),
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
            {"side": side, "target": target.cpu().tolist(), "error_m": error, "start_limit_tolerance_rad": violation}
        )
        print(f"EXECUTED {side}: error={error:.6f} m", flush=True)
        return error

    def move(side, target, quat, excluded=(), tolerance=0.002):
        _reach_target(move_once, side, target, quat, excluded, tolerance)

    def tcp():
        sensor = base.scene["ee_frame"]
        return sensor.data.target_pos_w.torch[0, 0].clone(), sensor.data.target_quat_w.torch[0, 0].clone()

    def quat_wxyz(quat):
        return quat[[3, 0, 1, 2]].cpu().tolist()

    def attach_peg():
        # Cover the carried rod in the measured TCP frame for collision planning.
        local = torch.zeros((17, 3), device=base.device)
        local[:, 2] = torch.linspace(0.004, 0.146, 17, device=base.device)
        peg = base.scene["peg"].data
        centers = position("peg") + quat_apply(peg.root_quat_w.torch[0].expand(17, 4), local)
        p, q = tcp()
        centers = quat_apply_inverse(q.expand(17, 4), centers - p)
        spheres = torch.cat((centers, centers.new_full((17, 1), 0.014)), dim=1)
        attached_spheres["right"] = spheres.to("cuda:0")
        get_planner("right").attach_spheres_to_robot(sphere_tensor=attached_spheres["right"])

    try:
        for attempt in range(args.attempts):
            env.reset()
            for planner, _ in planners.values():
                planner.detach_spheres_from_robot()
            attached_spheres.clear()
            action[0, :7] = robot.data.joint_pos.torch[0, arm_ids[:7]]
            action[0, 8:15] = robot.data.joint_pos.torch[0, arm_ids[7:]]
            action[0, [7, 15]] = 1
            frame_index = 0
            entry = {"attempt": attempt, "success": False}
            report["attempts"].append(entry)
            video_writers = _open_videos(args, attempt, base.step_dt)
            try:
                set_phase("settle")
                step(30)
                initial = position("peg")
                sleeve_initial = position("sleeve")
                grasp = initial + torch.tensor([args.grasp_x, 0, args.grasp_z], device=base.device)
                # Approach from above and pinch the upper section of the upright rod.
                quat = [0, 1, 0, 0]
                up = torch.tensor([0, 0, 0.16], device=base.device)
                set_phase("approach peg")
                move("right", grasp + up, quat)
                move("right", grasp, quat, excluded=("peg",))
                set_phase("close gripper")
                action[0, 7] = -1
                step(30)
                set_phase("lift peg")
                move("right", grasp + up, quat, excluded=("peg",))
                entry["lift_m"] = (position("peg")[2] - initial[2]).item()
                entry["lift_quat_xyzw"] = base.scene["peg"].data.root_quat_w.torch[0].tolist()
                assert entry["lift_m"] > 0.15, f"Grasp failed: lift={entry['lift_m']:.4f} m"
                assert not args.probe_grasp, "Grasp probe complete; intentionally not a successful demonstration"
                attach_peg()
                set_phase("align peg above sleeve")
                desired_quat = torch.tensor([0.0, 0, 0, 1], device=base.device)

                def align(bottom_height, tolerance=0.001):
                    # Re-estimate the physical grasp transform at each insertion target.
                    rel_p, rel_q = subtract_frame_transforms(
                        position("peg"), base.scene["peg"].data.root_quat_w.torch[0], *tcp()
                    )
                    goal = position("sleeve") + torch.tensor([0, 0, bottom_height], device=base.device)
                    target, rotation = combine_frame_transforms(goal, desired_quat, rel_p, rel_q)
                    move("right", target, quat_wxyz(rotation), excluded=("sleeve", "peg"), tolerance=tolerance)

                align(0.11)
                set_phase("insert peg")
                # Assembly objects are excluded only from planning; SDF physics stays active.
                for height in (0.09, 0.075, 0.06, 0.045, 0.03, 0.015, 0.003, 0.0):
                    align(height)
                    entry.setdefault("insertion", []).append({
                        "peg_bottom_target_z": height,
                        "peg_position": position("peg").tolist(),
                        "peg_quat": base.scene["peg"].data.root_quat_w.torch[0].tolist(),
                    })
                set_phase("release peg")
                action[0, 7] = 1
                step(20)
                planners["right"][0].detach_spheres_from_robot()
                attached_spheres.clear()
                p, q = tcp()
                retreat = torch.tensor([0, 0, 0.10], device=base.device)
                move("right", p + retreat, quat_wxyz(q), excluded=("sleeve", "peg"))
                step(30)
                set_phase("verify inserted peg")
                for _ in range(15):
                    step()
                    assert bool(
                        peg_is_inserted(base, **description.task.success_params)[0]
                    ), "Peg is not fully inserted"
                    assert torch.norm(position("sleeve") - sleeve_initial) < 0.001, "Fixed sleeve moved"
                    assert torch.norm(base.scene["peg"].data.root_lin_vel_w.torch[0]) < 0.03, "Peg is still moving"
                entry["success"] = True
            except Exception as exc:
                entry["error"] = str(exc)
                traceback.print_exc()
            entry["phase"] = phase
            entry["frames"] = frame_index
            entry["final_positions"] = {name: position(name).tolist() for name in ("peg", "sleeve")}
            _close_videos(video_writers)
            base.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            base.recorder_manager.set_success_to_episodes([0], torch.tensor([[entry["success"]]], device=base.device))
            base.recorder_manager.export_episodes([0])
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
            print("ATTEMPT_RESULT", json.dumps(entry), flush=True)
    finally:
        _close_videos(video_writers)
        env.close()
    return any(entry["success"] for entry in report["attempts"])


if __name__ == "__main__":
    main()
