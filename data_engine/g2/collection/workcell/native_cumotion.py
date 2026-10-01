# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt data engine's native cuMotion planner and executor to G2 workcell actions."""

import numpy as np
import time
import yaml


def install(session):
    """Use native cuMotion while retaining the shared recorder and independent task guards."""
    from data_engine.motion.cumotion.executor import EnvActionExecutor
    from data_engine.motion.cumotion.planner import CumotionArmPlanner
    from data_engine.motion.embodiments.g2 import create_g2_cumotion_cfg

    s = session
    original_step = s.step
    active_guard = [None]

    def guarded_step(count=1):
        for _ in range(count):
            original_step()
            if active_guard[0] is not None:
                active_guard[0].check_actual()

    s.step = guarded_step
    motion_config = yaml.safe_load(s.args.config.read_text())
    embodiment = type("G2NativeWorkcell", (), {"name": "g2"})()
    s.report["planning_backend"] = "isaacsim.robot_motion.cumotion.GraphBasedMotionPlanner"
    s.report["local_ik_backend"] = "native_cumotion"
    s.report["planning_device"] = "native backend managed; CUDA execution not verified"
    s.report["collision_model"] = (
        "Vendor G2 spheres and measured locked joints; room boxes; tool bounding boxes; "
        "five conservative slabs per bin; carried-object sphere cover."
    )

    class Planner(CumotionArmPlanner):
        def detach_spheres_from_robot(self):
            # The shared placement driver immediately clears s.attachments; the next planner
            # is rebuilt from that measured state, without the payload geometry.
            pass

    class Interface:
        """Keep G2's binary gripper actions distinct from joint-space gripper targets."""

        def joint_names(self, term):
            return s.names[term.split("_")[0]]

        def set(self, term, values):
            import torch

            offset = 0 if term.startswith("right") else 8
            s.action[0, offset : offset + 7] = torch.as_tensor(values, device=s.base.device)

        def set_scalar(self, term, value):
            s.action[0, 7 if term.startswith("right") else 15] = value

        def step(self):
            s.step()

    interface = Interface()

    def planner_for(side):
        started = time.perf_counter()
        cfg = create_g2_cumotion_cfg(
            s.base,
            side,
            s.args.robot_yaml,
            s.args.robot_urdf,
            payload=s.attachments.get(side),
            output=s.args.output_dir / "planning",
        )
        planner = Planner(s.base, embodiment, arm=side, cfg=cfg)
        s.report.setdefault("motion_safety", []).append(planner.safety.report)
        error = planner.kinematics_error_m()
        assert error < 0.002, f"Native cuMotion/G2 FK mismatch: {error}"
        s.report["fk_parity"].append(dict(side=side, position_m=error, backend="native_cumotion"))
        s.report.setdefault("planner_setup_timings", []).append(
            dict(side=side, created=True, wall_seconds=time.perf_counter() - started)
        )
        s.planners[side] = planner
        return planner

    def add_world(planner, excluded):
        root = s.robot.data.root_pos_w.torch[0].cpu().numpy()
        for box in s.world(excluded).cuboid:
            planner.add_box_obstacle(box.name, np.array(box.pose[:3]) + root, box.dims, safety_tolerance_m=0.005)
        for name, mesh in s.meshes.items():
            if name == excluded:
                continue
            pos, rot = s.object_pose(name)
            quat = rot.as_quat()[[3, 0, 1, 2]]
            low, high = mesh["vertices"].min(0), mesh["vertices"].max(0)
            bounds = [(low, high)]
            if name in s.geometry["bins"]:
                geometry = s.geometry["bins"][name]
                inner_low, inner_high = np.asarray(geometry["inner_xy_bounds_m"])
                floor_high = high.copy()
                floor_high[2] = geometry["floor_z_m"]
                bounds = [(low.copy(), floor_high)]
                for axis in (0, 1):
                    lo, hi = low.copy(), high.copy()
                    hi[axis] = inner_low[axis]
                    bounds.append((lo, hi))
                    lo, hi = low.copy(), high.copy()
                    lo[axis] = inner_high[axis]
                    bounds.append((lo, hi))
            for index, (lo, hi) in enumerate(bounds):
                assert np.all(hi > lo), (name, lo, hi)
                planner.add_box_obstacle(
                    f"{name}_{index}", pos + rot.apply((lo + hi) / 2), hi - lo, quat, safety_tolerance_m=0.005
                )

    def check_target(planner, side, name, position, rotation, execute, exclude=None, joint_goal=None):
        s.phase[0] = name
        entry = dict(
            name=name,
            tool=s.args.tool,
            side=side,
            position_world_m=position.tolist(),
            quaternion_xyzw=rotation.as_quat().tolist(),
            executed=False,
            excluded_contact_object=exclude,
        )
        s.report["checks"].append(entry)
        # A candidate probe and its carry may share one planner. Build the world only once.
        if not planner._obstacles:
            add_world(planner, exclude)
        q = planner.joint_positions()
        quat = rotation.as_quat()[[3, 0, 1, 2]]
        # Targets already use the measured TCP convention, not the canonical grasp convention.
        planner.tool_correction = np.eye(3)
        if not execute:
            entry["ik_success"] = planner.ik_reachable(position, quat, q)
        else:
            started = time.perf_counter()
            candidate = (
                planner.plan_config(q, joint_goal) if joint_goal is not None else planner.plan_pose(q, position, quat)
            )
            travel_limits = np.full(7, 2.6)
            if name == "carry_to_bin":
                travel_limits = np.asarray(
                    motion_config["tools"][s.args.tool].get("carry_joint_travel_limits_rad", travel_limits)
                )
            if (
                (s.args.tool == "drill" and name == "carry_to_bin") or name in ("retreat_from_bin", "retreat_align")
            ) and (
                candidate is None
                or candidate.limit_margin_rad < 0.03
                or np.any(np.abs(candidate.q_end - q) > travel_limits)
            ):
                # The graph pose query can choose the opposite shoulder/elbow branch.
                # Seed native IK near the measured posture (or the deliberate drill roll),
                # then collision-plan to that joint goal. IK alone never authorizes motion.
                from data_engine.motion.cumotion.robot_description import import_cumotion

                cm = import_cumotion()
                transform = np.eye(4)
                transform[:3, :3] = rotation.as_matrix()
                transform[:3, 3] = position - planner.base_pos
                entry["native_ik_branch_search"] = []
                for shoulder in (-0.6, -0.3, 0.0, 0.3, 0.6):
                    seed = q.copy()
                    if name == "carry_to_bin":
                        seed[4] += np.pi
                    seed[0] += shoulder
                    seed = np.clip(seed, planner.joint_limits[:, 0] + 0.03, planner.joint_limits[:, 1] - 0.03)
                    ik_cfg = cm.IkConfig()
                    ik_cfg.cspace_seeds = [seed]
                    ik = cm.solve_ik(planner.kinematics, cm.Pose3(transform), planner.cfg.tool_frame, ik_cfg)
                    item = dict(shoulder_seed_offset=shoulder, ik_success=bool(ik.success), plan_success=False)
                    entry["native_ik_branch_search"].append(item)
                    if not ik.success or np.any(np.abs(ik.cspace_position - q) > travel_limits):
                        continue
                    alternative = planner.plan_config(q, ik.cspace_position)
                    if (
                        alternative is not None
                        and alternative.limit_margin_rad >= 0.03
                        and np.all(np.abs(alternative.q_end - q) <= travel_limits)
                    ):
                        item["plan_success"] = True
                        candidate = alternative
                        break
            entry["planning_wall_seconds"] = time.perf_counter() - started
            entry["plan_success"] = candidate is not None
            if candidate is not None:
                entry.update(limit_margin_rad=candidate.limit_margin_rad, max_travel_rad=candidate.max_travel_rad)
                joint_travel = np.abs(candidate.q_end - q)
                entry["joint_travel_rad"] = joint_travel.tolist()
                entry["joint_travel_limits_rad"] = travel_limits.tolist()
                assert travel_limits.shape == (7,) and np.isfinite(travel_limits).all()
                assert candidate.limit_margin_rad >= 0.03 and np.all(
                    joint_travel <= travel_limits
                ), f"Native path violates G2 margin/travel limits: {entry}"
                tracked = {n: s.object_pose(n)[0].copy() for n in s.meshes}
                executor = EnvActionExecutor(s.base, planner, interface, f"{side}_arm", f"{side}_gripper")
                executor._gripper_target = float(s.action[0, 7 if side == "right" else 15])
                started = time.perf_counter()
                active_guard[0] = planner.safety
                executor.follow(
                    candidate.path, speed=0.2 if s.args.tool == "drill" and name == "carry_to_bin" else 0.35
                )
                entry["execution_wall_seconds"] = time.perf_counter() - started
                from data_engine.g2.collection.workcell.grasp_motion import tcp_pose

                actual, actual_rotation = tcp_pose(s, side)
                entry.update(
                    executed=True,
                    actual_position_world_m=actual.tolist(),
                    position_error_m=float(np.linalg.norm(actual - position)),
                    rotation_error_deg=float(np.degrees((actual_rotation.inv() * rotation).magnitude())),
                    object_displacement_m={
                        n: float(np.linalg.norm(s.object_pose(n)[0] - p)) for n, p in tracked.items()
                    },
                )
                entry["pose_pass"] = (
                    entry["position_error_m"] < s.args.position_tolerance
                    and entry["rotation_error_deg"] < s.args.rotation_tolerance
                )
                entry["objects_undisturbed"] = max(entry["object_displacement_m"].values()) < s.args.object_tolerance
        s.save()
        print("NATIVE_CHECK", entry, flush=True)
        return entry

    s.planner_for = planner_for
    s.check_target = check_target
