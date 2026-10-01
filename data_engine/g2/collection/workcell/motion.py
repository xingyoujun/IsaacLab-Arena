# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plan and physically execute G2 workcell grasp, carry and release motions."""

import copy
import json
import sys

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse


def run_step_guards(*guards):
    """Apply both retained-placement and current-motion checks after each simulator step."""
    for guard in guards:
        if guard[0] is not None:
            guard[0]()


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


def run_session(args, sequence):
    """Execute the supplied workcell sequence and retain physical step evidence."""
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from types import SimpleNamespace

    Cuboid = Mesh = WorldConfig = SimpleNamespace

    from isaaclab_arena.embodiments.g2.g2 import G2JointPositionActionsCfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    description, geometry, meshes = prepare_scene(args)
    from data_engine.motion.embodiments.g2 import configure_g2_placement

    configure_g2_placement(description)
    description.embodiment.action_config = G2JointPositionActionsCfg()
    if args.enable_cameras:
        from isaaclab_arena.embodiments.g2.g2 import G2CameraCfg

        description.embodiment.camera_config = G2CameraCfg()
    env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
    assert "curobo" not in sys.modules, "Unexpected legacy planner imported while building G2"
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
    recording_ready = [False]

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
        planners = {}
        attachments = {}
        step_guard = [None]
        sequence_guard = [None]
        assert str(base.device).startswith("cuda"), base.device
        report["simulation_device"] = str(base.device)
        report["planning_device"] = "native backend managed; CUDA execution not verified"
        report["local_ik_backend"] = "native_cumotion"
        if getattr(args, "record_video", False):
            from data_engine.g2.collection.workcell.cameras import ThreeViewWriter

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
                if video_writer is not None and recording_ready[0]:
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
            recording_ready[0] = False
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
            # Begin at a measured settled state, excluding inconsistent reset linkage poses.
            base.recorder_manager.reset([0])
            base.recorder_manager.record_post_reset([0])
            frames.clear()
            recording_ready[0] = True
            report["unrecorded_settle_control_steps"] = 45

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
                geometry=geometry,
                report=report,
                reset=reset,
                object_pose=object_pose,
                save=save,
            )
        )
        report["legacy_planner_imported"] = any(name == "curobo" or name.startswith("curobo.") for name in sys.modules)
        assert not report["legacy_planner_imported"], "Native G2 collection must not depend on cuRobo"
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
