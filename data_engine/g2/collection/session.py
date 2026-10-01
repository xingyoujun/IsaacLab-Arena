# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record native cuMotion G2 task recipes through Arena environment actions."""

import json
import numpy as np
import sys
import time
import torch
import traceback

from data_engine.motion.cumotion.executor import EnvActionExecutor
from data_engine.motion.cumotion.planner import CumotionArmPlanner
from data_engine.motion.embodiments.g2 import configure_g2_placement, create_g2_cumotion_cfg


class Session:
    """Own measured-state planning, collision geometry and recorded environment stepping."""

    def __init__(self, env, description, args, task):
        self.env, self.base, self.description = env, env.unwrapped, description
        self.args, self.task = args, task
        self.robot = self.base.scene["robot"]
        self.action = torch.zeros((1, 16), device=self.base.device)
        self.attachments = {}
        self.frames = 0
        self.report = dict(
            task_success=False,
            planner="native_cumotion",
            phases=[],
            executed_targets=[],
            planning_device="native backend managed; CUDA execution not verified",
        )
        self.names = {
            side: [f'idx{6 if side == "right" else 2}{i}_arm_{side[0]}_joint{i}' for i in range(1, 8)]
            for side in ("right", "left")
        }
        self.failure_terms = {}
        self.guard = None
        self.object_guard = None
        self.report["decisions"] = []
        from data_engine.planning.task_plan import load_plan

        self.tracking = load_plan("g2", args.task).get("tracking", {})

    def joint_names(self, term):
        return self.names[term.split("_")[0]]

    def set(self, term, values):
        offset = 0 if term.startswith("right") else 8
        self.action[0, offset : offset + 7] = torch.as_tensor(values, device=self.base.device)

    def set_scalar(self, term, value):
        self.action[0, 7 if term.startswith("right") else 15] = value

    def step(self, count=1):
        for _ in range(count):
            _, _, terminated, truncated, _ = self.env.step(self.action)
            self.frames += 1
            if self.guard is not None:
                self.guard.check_actual()
            if self.object_guard is not None:
                name, origin, limit = self.object_guard
                shift = float(torch.linalg.vector_norm(self.position(name) - origin))
                assert shift <= limit, f"Preclose object disturbance: {name}: {shift} m > {limit} m"
            if self.frames % 100 == 0:
                print("G2_STEPS", self.frames, flush=True)
            assert not bool(terminated[0] or truncated[0]), "Unexpected environment reset"
            for name, term in self.failure_terms.items():
                assert not bool(term.func(self.base, **term.params)[0]), f"Task failure: {name}"

    def phase(self, name):
        self.report["phases"].append(dict(frame=self.frames, name=name))
        print("G2_PHASE", name, flush=True)

    def decision(self, kind, **details):
        """Persist candidate choices and rejection reasons before executing any motion."""
        self.report["decisions"].append(dict(kind=kind, frame=self.frames, **details))
        (self.args.output_dir / "decisions.json").write_text(json.dumps(self.report["decisions"], indent=2))

    def position(self, name):
        return self.base.scene[name].data.root_pos_w.torch[0].clone()

    def add_world(self, planner, excluded):
        floor = -self.task["environment_config"]["table_height_m"]
        boxes = [
            ("table", [0, 0, -0.025], [0.60, 1.20, 0.05]),
            ("floor", [0, 0, floor - 0.05], [5, 5, 0.1]),
            ("cabinet_and_counter", [1.35, 0, floor + 0.48], [0.75, 1.9, 0.96]),
            ("shelf", [1.4, 0, floor + 1.65], [0.55, 1.9, 0.05]),
            ("shelf_back", [1.7, 0, floor + 1.40], [0.05, 1.9, 0.55]),
            ("back_wall", [1.8, 0, floor + 1.5], [0.12, 5, 3]),
            ("left_wall", [-0.2, 2.5, floor + 1.5], [4, 0.12, 3]),
            ("sideboard", [0.9, 1.95, floor + 0.45], [1.3, 0.55, 0.9]),
        ]
        objects = (
            [
                ("bowl_1", 0, [0.158, 0.158, 0.083]),
                ("bowl_2", 0, [0.158, 0.158, 0.083]),
                ("bowl_3", 0, [0.158, 0.158, 0.083]),
            ]
            if self.args.task == "stack_bowls"
            else [("peg", 0.075, [0.026, 0.026, 0.15]), ("sleeve", 0.033, [0.12, 0.12, 0.084])]
        )
        for name, z, dims in objects:
            if name not in excluded:
                center = self.position(name).cpu().numpy()
                center[2] += z
                boxes.append((name, center, dims))
        self.world_boxes = boxes
        for name, center, dims in boxes:
            planner.add_box_obstacle(name, np.asarray(center), np.asarray(dims), safety_tolerance_m=0.002)

    def world_contacts(self, planner, q):
        import yaml
        from pathlib import Path

        description = yaml.safe_load(Path(planner.cfg.lula_robot_description).read_text())
        contacts = []
        for entry in description["collision_spheres"]:
            for link, spheres in entry.items():
                pose = planner.kinematics.pose(q, link)
                rotation = np.asarray(pose.rotation.matrix())
                position = np.asarray(pose.translation) + planner.base_pos
                for sphere in spheres:
                    center = rotation @ np.asarray(sphere["center"]) + position
                    for name, origin, dims in self.world_boxes:
                        distance = np.linalg.norm(np.maximum(np.abs(center - origin) - np.asarray(dims) / 2, 0))
                        if distance < sphere["radius"] + 0.002:
                            contacts.append((link, name, float(sphere["radius"] + 0.002 - distance)))
        return contacts

    def move(self, side, target, quat, excluded=(), tolerance=0.002, cartesian=None, probe=False):
        # Rebuild from measured inactive joints and payload; stale locked arms are unsafe.
        cfg = create_g2_cumotion_cfg(
            self.base, side, payload=self.attachments.get(side), output=self.args.output_dir / "planning"
        )
        planner = CumotionArmPlanner(self.base, self.description.embodiment, arm=side, cfg=cfg)
        self.report.setdefault("motion_safety", []).append(planner.safety.report)
        from data_engine.motion.cumotion.robot_description import import_cumotion

        cm = import_cumotion()
        planner._planner.get_graph_planner_config().set_param("max_iterations", cm.MotionPlannerConfig.ParamValue(100))
        inspector = cm.create_robot_world_inspector(planner.robot_description)
        print(
            "G2_START_SELF_COLLISION",
            list(inspector.frames_in_self_collision(planner.joint_positions().reshape(-1, 1))),
            flush=True,
        )
        planner.tool_correction = np.eye(3)  # Recipes already specify actual TCP orientation.
        parity = planner.kinematics_error_m()
        assert parity < 0.002, f"G2 TCP/URDF mismatch: {parity}"
        self.add_world(planner, excluded)
        print("G2_START_WORLD", self.world_contacts(planner, planner.joint_positions()), flush=True)
        target_np = target.detach().cpu().numpy()
        executor = EnvActionExecutor(self.base, planner, self, f"{side}_arm", f"{side}_gripper")
        executor._gripper_target = float(self.action[0, 7 if side == "right" else 15])
        commanded_target = target_np.copy()
        original_corridor = None
        for correction in range(1 + self.tracking.get("max_corrections", 2)):
            measured = planner.joint_positions()
            q = np.clip(measured, planner.joint_limits[:, 0], planner.joint_limits[:, 1])
            violation = float(np.max(np.abs(q - measured)))
            assert violation <= 1e-4, f"Measured hard-stop violation: {violation}"
            started = time.perf_counter()
            candidate = (
                planner.plan_cartesian(q, commanded_target, np.asarray(quat), **cartesian)
                if cartesian is not None
                else planner.plan_pose(q, target_np, np.asarray(quat))
            )
            if cartesian is not None:
                if original_corridor is None:
                    original_corridor = planner.safety.path_constraint
                planner.safety.path_constraint = original_corridor
                if candidate is not None:
                    planner.safety.check_path(candidate.path.get_waypoints().numpy(), "original_cartesian_corridor")
            if cartesian is None and (candidate is None or np.max(np.abs(candidate.q_end - q)) > 2.6):
                if candidate is not None:
                    rejected = dict(
                        side=side, frame=self.frames, max_joint_travel_rad=float(np.max(np.abs(candidate.q_end - q)))
                    )
                    self.report.setdefault("branch_rejections", []).append(rejected)
                    print("G2_REJECT_BRANCH", rejected, flush=True)
                candidate = None
                from scipy.spatial.transform import Rotation

                from data_engine.motion.cumotion.robot_description import import_cumotion

                cm = import_cumotion()
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
                transform[:3, 3] = target_np - planner.base_pos
                seed_offsets = [(0, v) for v in (0.0, -0.3, 0.3, -0.6, 0.6)]
                # The redundant shoulder elevation can preserve the wrist branch
                # when a straight descent otherwise wraps through its hard stop.
                seed_offsets += [(1, v) for v in (-1.2, -0.6, 0.6, 1.2, -1.8, -2.4)]
                for joint, offset in seed_offsets:
                    seed = q.copy()
                    seed[joint] += offset
                    ik_cfg = cm.IkConfig()
                    ik_cfg.cspace_seeds = [
                        np.clip(seed, planner.joint_limits[:, 0] + 0.001, planner.joint_limits[:, 1] - 0.001)
                    ]
                    ik = cm.solve_ik(planner.kinematics, cm.Pose3(transform), planner.cfg.tool_frame, ik_cfg)
                    print(
                        "G2_IK",
                        side,
                        (joint, offset),
                        bool(ik.success),
                        "start",
                        q.tolist(),
                        "goal",
                        ik.cspace_position.tolist() if ik.success else None,
                        "goal_collision",
                        (
                            list(inspector.frames_in_self_collision(ik.cspace_position.reshape(-1, 1)))
                            if ik.success
                            else None
                        ),
                        flush=True,
                    )
                    if ik.success and np.max(np.abs(ik.cspace_position - q)) <= 2.6:
                        print("G2_GOAL_WORLD", self.world_contacts(planner, ik.cspace_position), flush=True)
                        candidate = planner.plan_config(q, ik.cspace_position)
                        if candidate is not None:
                            break
            self.decision(
                "motion_plan",
                side=side,
                target_world_m=target_np.tolist(),
                commanded_target_world_m=commanded_target.tolist(),
                quaternion_wxyz=list(quat),
                mode="cartesian_corridor" if cartesian is not None else "free_space",
                contact_objects=list(excluded),
                probe_only=probe,
                accepted=candidate is not None,
                rejection_evidence=list(planner.safety.report["failures"]),
            )
            assert candidate is not None, f"Native cuMotion failed safety/travel checks: {side} {target_np}"
            assert np.max(np.abs(candidate.q_end - q)) <= 2.6, "Discontinuous native IK branch"
            if probe:
                return planner, candidate
            planning_seconds = time.perf_counter() - started
            print(
                "G2_EXECUTE", side, "planning_seconds", planning_seconds, "goal", candidate.q_end.tolist(), flush=True
            )
            started = time.perf_counter()
            self.guard = planner.safety
            executor.follow(candidate.path, speed=0.35, settle_steps=10)
            execution_seconds = time.perf_counter() - started
            sensor = self.base.scene["ee_frame" if side == "right" else "left_ee_frame"]
            error = float(torch.norm(sensor.data.target_pos_w.torch[0, 0] - target))
            self.report["executed_targets"].append(
                dict(
                    side=side,
                    target=target_np.tolist(),
                    commanded_target=commanded_target.tolist(),
                    error_m=error,
                    fk_error_m=parity,
                    correction=correction,
                    planning_seconds=planning_seconds,
                    execution_seconds=execution_seconds,
                )
            )
            print("G2_TARGET", side, target_np.tolist(), error, flush=True)
            if error < tolerance:
                return None
            if cartesian is not None and self.tracking.get("max_compensation_m", 0) > 0:
                # Correct measured load deflection, while preserving the original goal and corridor.
                residual = target_np - sensor.data.target_pos_w.torch[0, 0].cpu().numpy()
                commanded_target += residual
                compensation = float(np.linalg.norm(commanded_target - target_np))
                assert (
                    compensation <= self.tracking["max_compensation_m"]
                ), "Tracking compensation exceeds configured bound"
        assert error < tolerance, f"Physical TCP missed target: {error} >= {tolerance}"
        return None


def collect(args, task):
    """Retain one successful or failed episode with the shared transition contract."""
    from isaaclab.managers import DatasetExportMode, TerminationTermCfg

    import isaaclab_arena_environments  # noqa: F401
    from data_engine.g2.collection import recipes
    from data_engine.recording.alignment import transition_metadata
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.embodiments.g2.g2 import G2JointPositionActionsCfg
    from isaaclab_arena.embodiments.g2.recorders import G2CollectionSuccessTerm, core_recorder_cfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg

    args.output_dir.mkdir(parents=True, exist_ok=False)
    factory = EnvironmentRegistry().get_component_by_name(task["environment"])()
    description = factory.build(factory._legacy_argparse_cfg_type(**task["environment_config"], episode_length_s=600))
    configure_g2_placement(description)
    description.embodiment.action_config = G2JointPositionActionsCfg()
    original, failure_terms = description.env_cfg_callback, {}

    def configure(cfg):
        cfg = original(cfg)
        cfg.sim.render_interval = cfg.decimation
        cfg.terminations.success.func = G2CollectionSuccessTerm
        for name, term in vars(cfg.terminations).items():
            if name != "success" and isinstance(term, TerminationTermCfg):
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
    env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
    s = Session(env, description, args, task)
    s.failure_terms = failure_terms
    contract = transition_metadata("g2", ["head_camera", "left_wrist_camera", "right_wrist_camera"])
    s.base.cfg.get_ep_meta = lambda: dict(
        collection_contract=contract,
        env_name=task["environment"],
        environment_config=task["environment_config"],
        joint_names=s.robot.joint_names,
        action_config="isaaclab_arena.embodiments.g2.g2:G2JointPositionActionsCfg",
        task=task,
        pose_convention="right then left xyz+xyzw; metres; pre-action observation",
    )
    try:
        env.reset()
        for side in ("right", "left"):
            ids = [s.robot.joint_names.index(n) for n in s.names[side]]
            s.set(side + "_arm", s.robot.data.joint_pos.torch[0, ids])
            s.set_scalar(side + "_gripper", 1)
        s.step(task["settle_steps"])
        s.base.recorder_manager.reset([0])
        s.base.recorder_manager.record_post_reset([0])
        s.frames = 0
        if args.task == "peg_into_sleeve":
            from scipy.spatial.transform import Rotation

            q = s.base.scene["peg"].data.root_quat_w.torch[0].cpu().numpy()
            axis_z = float(Rotation.from_quat(q).apply([0, 0, 1])[2])
            s.report["settled_peg_axis_z"] = axis_z
            assert axis_z > np.cos(np.deg2rad(1.0)), f"Peg is not upright after reset: {axis_z}"
        assert "curobo" not in sys.modules, "Unexpected legacy planner imported while building G2"
        getattr(recipes, task["recipe"])(s)
        s.report["legacy_planner_imported"] = any(
            name == "curobo" or name.startswith("curobo.") for name in sys.modules
        )
        assert not s.report["legacy_planner_imported"], "Native G2 collection must not depend on cuRobo"
        s.report["task_success"] = True
    except Exception:
        s.report["error"] = traceback.format_exc()
        raise
    finally:
        s.report.update(frames=s.frames, unrecorded_settle_control_steps=task["settle_steps"])
        s.base.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        s.base.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[s.report["task_success"]]], device=s.base.device)
        )
        s.base.recorder_manager.export_episodes([0])
        (args.output_dir / "report.json").write_text(json.dumps(s.report, indent=2))
        env.close()
