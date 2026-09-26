# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run seeded, recorded physical qualification of the supplied pine_wm tasks."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="T001")
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--stage", choices=["preview", "stability", "collect"], default="preview")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--workspace", type=float, nargs=4, default=(-0.30, 0.30, -0.12, 0.25), metavar=("XMIN", "XMAX", "YMIN", "YMAX")
)
parser.add_argument("--audit-only", action="store_true")
parser.add_argument("--replay-results", type=Path, help="Diagnose one saved layout without resampling.")
parser.add_argument("--replay-trial", type=int, default=0)
parser.add_argument("--stack-count", type=int, choices=[3, 5], default=None)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--enable_cameras", action="store_true")
args = parser.parse_args()
from review_workflow import validate_run

validate_run(args.stage, [args.task])
args.headless = True
cameras_enabled = args.enable_cameras
launcher = AppLauncher(args)


def main():  # noqa: C901 - simulator lifetime encloses task callbacks and the trial loop.
    import json
    import numpy as np
    import os
    import torch
    import yaml
    from scipy.spatial.transform import Rotation

    import warp as wp
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.embodiments.ur7e.demo_recorders import ur7e_demo_recorder_cfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_cumotion.executor import EnvActionExecutor, JointActionInterface
    from isaaclab_arena_cumotion.grasps import quat_wxyz_from_matrix
    from isaaclab_arena_cumotion.planner import CumotionArmPlanner
    from isaaclab_arena_cumotion.robot_description import import_cumotion
    from isaaclab_arena_environments.pine_wm_first20_environment import (
        DEFAULT_PACKAGE,
        PineWmFirst20Environment,
        PineWmFirst20EnvironmentCfg,
    )

    assert (
        not (args.output / "results.json").exists() and not (args.output / "demos.hdf5").exists()
    ), f"Use a new output directory: {args.output}"
    args.output.mkdir(parents=True, exist_ok=True)
    review = json.loads(Path(__file__).with_name("review_layouts.json").read_text())
    sizes = review["tasks"][args.task].get("asset_sizes_mm", {})
    (args.output / "layout_snapshot.json").write_text(json.dumps(review["tasks"][args.task], indent=2))
    environment_config = {
        "task_id": args.task,
        "review_asset_sizes_mm": sizes,
        "package_root": os.environ.get("ARENA_PINE_WM_FIRST20_ROOT", DEFAULT_PACKAGE),
    }
    (args.output / "environment.json").write_text(json.dumps(environment_config, indent=2) + "\n")
    arena = PineWmFirst20Environment().build(
        PineWmFirst20EnvironmentCfg(task_id=args.task, enable_cameras=cameras_enabled, review_asset_sizes_mm=sizes)
    )

    def recording(cfg):
        cfg.recorders = ur7e_demo_recorder_cfg(with_cameras=False)
        cfg.recorders.dataset_export_dir_path = str(args.output)
        cfg.recorders.dataset_filename = "demos"
        cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
        cfg.env_name = "pine_wm_first20"
        cfg.sim.render_interval = cfg.decimation
        cfg.episode_length_s = 600.0
        return cfg

    if cameras_enabled:
        arena.embodiment.camera_config.set_use_tiled_camera(False)
    arena.env_cfg_callback = recording
    env = (
        ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(device=args.device, solve_relations=False, seed=args.seed))
        .make_registered()
        .unwrapped
    )
    env.reset()
    from live_preview import LivePreview

    preview = LivePreview(env, args.output) if cameras_enabled else None
    planner = CumotionArmPlanner(env, arena.embodiment)
    assert planner.kinematics_error_m() < 0.003
    inspector = import_cumotion().create_robot_world_inspector(planner.robot_description)
    interface = JointActionInterface(env)
    executor = EnvActionExecutor(env, planner, interface, "arm_action", "gripper_action")
    rng = np.random.default_rng(args.seed)
    bounds = {k: np.array(v) for k, v in arena.first20_bounds.items()}
    names = list(bounds)
    fixtures = {name for name, _, fixed in arena.first20_instances if fixed}
    current = {"phase": "reset", "checks": 0, "min_clearance": 1.0, "allowed": set(), "held": None}
    from contacts import ContactAudit
    from geometry import collider_transform

    contact_audit = ContactAudit(env, current, names)
    record_physics_step = env.recorder_manager.record_post_physics_decimation_step

    def record_and_audit_physics_step():
        record_physics_step()
        contact_audit.check()

    # Audit every 120 Hz physics substep, not only the 15 Hz action boundary.
    env.recorder_manager.record_post_physics_decimation_step = record_and_audit_physics_step

    sphere_links = yaml.safe_load(Path(planner.cfg.lula_robot_description).read_text())["collision_spheres"]
    geometries = []
    for item in sphere_links:
        for link, spheres in item.items():
            geometries.append(
                (link, np.array([s["center"] for s in spheres]), np.array([s["radius"] for s in spheres]))
            )
    # Measured USD colliders are decomposed individually: cavities remain empty.
    colliders = {}
    primitive_geometry = {}
    for name in names:
        asset = env.scene[name]
        print("ASSET_PATH", name, asset.cfg.prim_path, flush=True)
        root = env.sim.stage.GetPrimAtPath(
            asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
            .replace(".*", "0")
            .replace(".+", "0")
            .replace("[^/]+", "0")
        )
        cache = UsdGeom.XformCache()
        body_prims = []
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                body_prims.append(p)
        shapes = []
        primitive_geometry[name] = []
        for body in body_prims:
            for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():
                    box = (
                        UsdGeom.BBoxCache(0, ["default", "render", "proxy", "guide"], False, True)
                        .ComputeUntransformedBound(p)
                        .ComputeAlignedRange()
                    )
                    assert not box.IsEmpty(), f"Empty collider bounds: {p.GetPath()}"
                    T_B_C = collider_transform(cache, body, p)
                    shapes.append((body.GetName(), T_B_C, np.array(box.GetMin()), np.array(box.GetMax())))
                    kind = p.GetTypeName().lower()
                    axis = "XYZ".index(str(p.GetAttribute("axis").Get())) if kind == "cylinder" else None
                    if kind == "mesh":
                        points = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
                        half = (np.array(box.GetMax()) - np.array(box.GetMin())) / 2
                        middle = (np.array(box.GetMax()) + np.array(box.GetMin())) / 2
                        axis = (points - middle) / np.maximum(half, 1e-12)
                    primitive_geometry[name].append((kind, axis))
        assert shapes, f"No collision geometry: {name} {root.GetPath()}"
        colliders[name] = shapes
    protected_colliders = []
    for name, shapes in colliders.items():
        for body, _, _, _ in shapes:
            protected_colliders.append(name in {"drawer", "button"} and body in {"carcass", "base"})
    protected_colliders = np.array(protected_colliders)
    print("COLLIDERS", {k: len(v) for k, v in colliders.items()}, flush=True)
    gripper_shapes = []
    gripper_primitives = []
    root = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/ee_link/Robotiq_2F_85")
    cache = UsdGeom.XformCache()
    for body in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                box = (
                    UsdGeom.BBoxCache(0, ["default", "render", "proxy", "guide"], False, True)
                    .ComputeUntransformedBound(prim)
                    .ComputeAlignedRange()
                )
                assert not box.IsEmpty()
                T_B_C = collider_transform(cache, body, prim)
                gripper_shapes.append((body.GetName(), T_B_C, np.array(box.GetMin()), np.array(box.GetMax())))
                if prim.GetTypeName() == "Mesh":
                    half = (np.array(box.GetMax()) - np.array(box.GetMin())) / 2
                    middle = (np.array(box.GetMax()) + np.array(box.GetMin())) / 2
                    points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                    gripper_primitives.append(("mesh", (points - middle) / np.maximum(half, 1e-12)))
                else:
                    gripper_primitives.append(("box", None))
    assert gripper_shapes, "Missing gripper collision geometry"

    def vec(value):
        return wp.to_torch(value).detach().cpu().numpy()

    def pose(name):
        return vec(env.scene[name].data.root_pose_w)[0].copy()

    def put(name, position, rot=None):
        asset = env.scene[name]
        q = np.array([0, 0, 0, 1]) if rot is None else np.asarray(rot)
        asset.write_root_pose_to_sim(torch.tensor([[*position, *q]], device=env.device, dtype=torch.float32))
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))

    def center(name):
        p = pose(name)
        local = bounds[name].mean(axis=0)
        if name == "mug":
            local = np.array([0.0, 0.0, 0.035])
        return p[:3] + Rotation.from_quat(p[3:]).apply(local)

    def self_check(q):
        pairs = list(inspector.frames_in_self_collision(np.asarray(q, dtype=np.float64).reshape(-1, 1)))
        if pairs:
            raise RuntimeError(f"self_collision:{pairs}")

    def obstacle_boxes():
        boxes = []
        for name, shapes in colliders.items():
            a = env.scene[name]
            if name in env.scene.articulations:
                positions = vec(a.data.body_pos_w)[0]
                quats = vec(a.data.body_quat_w)[0]
                bodyposes = {
                    n: (p, Rotation.from_quat(q).as_matrix()) for n, p, q in zip(a.body_names, positions, quats)
                }
            else:
                pp = pose(name)
                bodyposes = {s[0]: (pp[:3], Rotation.from_quat(pp[3:]).as_matrix()) for s in shapes}
            for body, T, lo, hi in shapes:
                pos, rot = bodyposes[body]
                basis = rot @ T[:3, :3]
                scale = np.linalg.norm(basis, axis=0)
                basis = basis / scale
                c = pos + rot @ T[:3, 3] + basis @ (scale * (lo + hi) / 2)
                boxes.append((name, c, basis, scale * (hi - lo) / 2))
        return boxes

    def gripper_boxes(q):
        actual = planner.kinematics.pose(planner.joint_positions(), planner.cfg.tool_frame)
        planned = planner.kinematics.pose(np.asarray(q, dtype=np.float64), planner.cfg.tool_frame)
        delta_R = np.asarray(planned.rotation.matrix()) @ np.asarray(actual.rotation.matrix()).T
        a = np.asarray(actual.translation) + planner.base_pos
        b = np.asarray(planned.translation) + planner.base_pos
        cached = current.get("preflight_gripper")
        if cached is not None:
            return [(n, b + delta_R @ (c - a), delta_R @ r, h) for n, c, r, h in cached]
        robot = planner.robot
        positions = vec(robot.data.body_pos_w)[0]
        quats = vec(robot.data.body_quat_w)[0]
        poses = {n: (p, Rotation.from_quat(r).as_matrix()) for n, p, r in zip(robot.body_names, positions, quats)}
        result = []
        for name, T, lo, hi in gripper_shapes:
            p, R = poses[name]
            basis = R @ T[:3, :3]
            scale = np.linalg.norm(basis, axis=0)
            basis /= scale
            c = p + R @ T[:3, 3] + basis @ (scale * (lo + hi) / 2)
            result.append((name, b + delta_R @ (c - a), delta_R @ basis, scale * (hi - lo) / 2))
        return result

    def clearance(q, boxes):
        self_check(q)
        held = current["held"]
        if held is not None:
            actual = planner.kinematics.pose(planner.joint_positions(), planner.cfg.tool_frame)
            planned = planner.kinematics.pose(np.asarray(q, dtype=np.float64), planner.cfg.tool_frame)
            R = np.asarray(planned.rotation.matrix()) @ np.asarray(actual.rotation.matrix()).T
            a = np.asarray(actual.translation) + planner.base_pos
            b = np.asarray(planned.translation) + planner.base_pos
            boxes = [(n, b + R @ (c - a), R @ rot, h) if n == held else (n, c, rot, h) for n, c, rot, h in boxes]
        if held is not None:
            from geometry import validate_held_object

            supports = {"v_support"} if args.task == "T004" and current["phase"] == "pick" else set()
            validate_held_object(boxes, held, supports, primitive_geometry)
        from geometry import sphere_box_distances

        box_names = np.array([box[0] for box in boxes])
        intended_contact = (box_names == current.get("manipulated")) & ~protected_colliders
        box_centers = np.array([box[1] for box in boxes])
        box_rotations = np.array([box[2] for box in boxes])
        box_extents = np.array([box[3] for box in boxes])
        spheres_world = {}
        for link, cs, rs in geometries:
            T = planner.kinematics.pose(np.asarray(q, dtype=np.float64), link)
            pts = cs @ np.asarray(T.rotation.matrix()).T + np.asarray(T.translation) + planner.base_pos
            spheres_world[link] = (pts, rs)
            # Contact geometry is only permitted at the gripper, never cameras/arm.
            table = np.min(pts[:, 2] - rs) - 0.74
            if link not in {"base_link_inertia", "shoulder_link"} and table < -0.001:
                raise RuntimeError(f"table_clearance:{link}:{table:.4f}")
            distances = sphere_box_distances(pts, rs, box_centers, box_rotations, box_extents)
            if link == "pine_wm_gripper_base":
                distances[:, intended_contact] = np.inf
            minimum = float(distances.min())
            current["min_clearance"] = min(current["min_clearance"], minimum)
            if current.get("checking_actual"):
                current["min_executed_clearance"] = min(current.get("min_executed_clearance", 1.0), minimum)
            required_gap = 0.002 if current.get("checking_actual") else 0.004
            if minimum < required_gap:
                sphere_index, box_index = np.unravel_index(np.argmin(distances), distances.shape)
                raise RuntimeError(
                    f"obstacle_clearance:{link}:{box_names[box_index]}:{minimum:.4f}:sphere={sphere_index}:box_center={box_centers[box_index].round(4).tolist()}"
                )
        from geometry import penetration

        for g, primitive in zip(gripper_boxes(q), gripper_primitives):
            name, c, R, h = g
            extent = np.abs(R) @ h
            bottom = c[2] - extent[2]
            if primitive[0] == "mesh":
                bottom = c[2] + float(np.min((primitive[1] * h) @ R[2, :]))
            if bottom < 0.739:
                raise RuntimeError(f"gripper_table_clearance:{name}")
            for index, ob in enumerate(boxes):
                if intended_contact[index]:
                    continue
                _, oc, ort, oh = ob
                if np.any(np.abs(c - oc) > extent + np.abs(ort) @ oh + 0.002):
                    continue
                if penetration(g, ob, primitive) > -0.001:
                    raise RuntimeError(
                        f"gripper_obstacle_clearance:{name}:{ob[0]}:ob_center={oc.round(4).tolist()}:gripper_bottom={bottom:.4f}"
                    )
        cameras, camera_radii = spheres_world["tool0"]
        # Only the collar (index 2) is mechanically joined to the wrist.
        exposed = np.arange(len(camera_radii)) != 2
        for wrist in ("wrist_1_link", "wrist_2_link"):
            points, radii = spheres_world[wrist]
            gap = (
                np.linalg.norm(cameras[exposed, None, :] - points[None, :, :], axis=2)
                - camera_radii[exposed, None]
                - radii[None, :]
            )
            if float(gap.min()) < 0.002:
                raise RuntimeError(f"camera_wrist_clearance:{wrist}:{gap.min():.4f}")
        current["checks"] += 1

    def checked_step():
        if current["phase"] != "reset":
            contact_audit.check()
            held = current.get("held")
            if held is None:
                current["grasp_reference"] = None
            else:
                T = planner.kinematics.pose(planner.joint_positions(), planner.cfg.tool_frame)
                local = np.asarray(T.rotation.matrix()).T @ (
                    center(held) - np.asarray(T.translation) - planner.base_pos
                )
                reference = current.get("grasp_reference")
                if reference is None or reference[0] != held:
                    current["grasp_reference"] = (held, local.copy())
                else:
                    slip = float(np.linalg.norm(local - reference[1]))
                    current["max_grasp_slip"] = max(current.get("max_grasp_slip", 0.0), slip)
                    if slip > 0.03:
                        raise RuntimeError(f"grasp_slip:{held}:{slip:.4f}")
            for name in names:
                point = center(name)
                if name != current["held"] and (point[2] < 0.72 or np.any(np.abs(point[:2]) > 0.50)):
                    raise RuntimeError(f"object_left_workspace:{name}")
            for articulated in ("drawer", "button"):
                if articulated in names:
                    current.setdefault("joint_samples", {}).setdefault(articulated, []).append(
                        float(vec(env.scene[articulated].data.joint_pos)[0, 0])
                    )
            if "cube_40" in names:
                current["max_cube_z"] = max(current.get("max_cube_z", 0.0), float(center("cube_40")[2]))
            current["checking_actual"] = True
            try:
                clearance(planner.joint_positions(), obstacle_boxes())
            finally:
                current["checking_actual"] = False

    executor.on_step = checked_step

    class PathPoints:
        def __init__(self, qs):
            self.qs = np.array(qs)

        def get_waypoints(self):
            return self

        def numpy(self):
            return self.qs

    def checked_follow(path, speed=0.3):
        qs = path.get_waypoints().numpy().astype(np.float64)
        boxes = obstacle_boxes()
        current["validation_stage"] = "planned_trajectory"
        trajectory = planner.trajectory_generator.generate_trajectory_from_cspace_waypoints(qs)
        if trajectory is None:
            raise RuntimeError("trajectory_generation_failed")
        count = max(2, int(np.ceil(trajectory.duration / (executor.dt * speed))))
        commands = []
        for t in np.linspace(0, trajectory.duration, count):
            state = trajectory.get_target_state(float(t))
            if state is not None:
                commands.append(state.joints.positions.numpy().astype(np.float64))
        previous = planner.joint_positions()
        current["preflight_gripper"] = gripper_boxes(planner.joint_positions())
        try:
            for q in commands:
                for t in np.linspace(0, 1, max(2, int(np.max(abs(q - previous)) / 0.015) + 1)):
                    clearance(previous + (q - previous) * t, boxes)
                previous = q
        finally:
            current["preflight_gripper"] = None
        current["validation_stage"] = "execution"
        for q in commands:
            executor.step(arm_target=q)
        executor.step(arm_target=commands[-1], steps=executor.settle_steps)

    planner.add_box_obstacle("/table", np.array([0, 0.10, 0.68]), (1.0, 0.80, 0.12), safety_tolerance_m=0.0)
    world_shapes = set()

    def sync_world():
        boxes = obstacle_boxes()
        keys = [f"/p20/{name}/{i}" for i, (name, _, _, _) in enumerate(boxes)]
        positions = np.array([c for _, c, _, _ in boxes], dtype=np.float32)
        rotations = (
            Rotation.from_matrix(np.array([R for _, _, R, _ in boxes])).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
        )
        enabled = wp.array(
            [name not in current["allowed"] or protected_colliders[i] for i, (name, _, _, _) in enumerate(boxes)],
            dtype=wp.bool,
        )
        poses = (wp.array(positions, dtype=wp.float32), wp.array(rotations, dtype=wp.float32))
        if not world_shapes:
            planner.world.add_cubes(
                prim_paths=keys,
                sizes=wp.array(np.ones(len(keys)), dtype=wp.float32),
                scales=wp.array(np.array([2 * h for _, _, _, h in boxes]), dtype=wp.float32),
                safety_tolerances=wp.array(np.full(len(keys), 0.002), dtype=wp.float32),
                poses=poses,
                enabled_array=enabled,
            )
            world_shapes.update(keys)
        else:
            assert set(keys) == world_shapes
            planner.world.update_obstacle_transforms(keys, poses)
            planner.world.update_obstacle_enables(keys, enabled)

    unloaded_graph_planner = planner._planner
    payload_active = [None]

    def sync_payload():
        held = current["held"]
        if held == payload_active[0]:
            return
        if held is None:
            planner._planner = unloaded_graph_planner
            payload_active[0] = None
            return
        from isaacsim.robot_motion.cumotion import GraphBasedMotionPlanner
        from isaacsim.robot_motion.cumotion.impl.configuration_loader import CumotionRobot

        from isaaclab_arena_cumotion.robot_description import xrdf_from_lula

        xrdf = xrdf_from_lula(planner.cfg)
        T = planner.kinematics.pose(planner.joint_positions(), planner.cfg.tool_frame)
        p = pose(held)
        world_center = p[:3] + Rotation.from_quat(p[3:]).apply(bounds[held].mean(axis=0))
        local = np.asarray(T.rotation.matrix()).T @ (world_center - planner.base_pos - np.asarray(T.translation))
        radius = float(np.linalg.norm((bounds[held][1] - bounds[held][0]) / 2) + 0.003)
        xrdf["geometry"]["arena_collision_spheres"]["spheres"][planner.cfg.tool_frame].append(
            {"center": local.tolist(), "radius": radius}
        )
        description = import_cumotion().load_robot_from_memory(
            yaml.dump(xrdf), Path(planner.cfg.robot_urdf).read_text()
        )
        robot = CumotionRobot(
            directory=None,
            robot_description=description,
            kinematics=description.kinematics(),
            controlled_joint_names=[description.cspace_coord_name(i) for i in range(description.num_cspace_coords())],
        )
        planner._planner = GraphBasedMotionPlanner(robot, planner.world, tool_frame=planner.cfg.tool_frame)
        payload_active[0] = held

    def quat(yaw):
        return quat_wxyz_from_matrix(Rotation.from_euler("z", yaw).as_matrix() @ np.diag([-1, 1, -1]))

    def ik(pos, orientation, seed):
        cm = import_cumotion()
        T = np.eye(4)
        T[:3, :3] = cm.Rotation3(*planner.to_tool_frame(orientation)).matrix()
        T[:3, 3] = pos - planner.base_pos
        cfg = cm.IkConfig()
        cfg.cspace_seeds = [np.asarray(seed, dtype=np.float64)]
        result = cm.solve_ik(planner.kinematics, cm.Pose3(T), planner.cfg.tool_frame, cfg)
        if not result.success:
            raise RuntimeError("ik_unreachable")
        q = np.asarray(result.cspace_position).reshape(-1)
        if np.max(abs(q - seed)) > 0.6:
            raise RuntimeError("ik_branch_jump")
        return q

    def move(tcp, orientation, linear=False):
        current["last_tcp_target"] = np.asarray(tcp).tolist()
        print("MOVE", current["phase"], np.round(tcp, 3), "linear", linear, flush=True)
        R = Rotation.from_quat([*orientation[1:], orientation[0]]).as_matrix()
        pos = np.asarray(tcp) - R[:, 2] * 0.1628
        sync_world()
        sync_payload()
        if linear:
            start = planner.tool_position()
            qs = [planner.joint_positions()]
            for t in np.linspace(0, 1, max(3, int(np.linalg.norm(pos - start) / 0.008)))[1:]:
                qs.append(ik(start + (pos - start) * t, orientation, qs[-1]))
            checked_follow(PathPoints(qs), 0.3)
        else:
            candidate = planner.plan_pose(planner.joint_positions(), pos, orientation)
            if candidate is None or not candidate.is_executable():
                raise RuntimeError("no_executable_plan")
            checked_follow(candidate.path)
        error = np.linalg.norm(planner.tool_position() - pos)
        if error > 0.008:
            raise RuntimeError(f"tool_tracking_error:{error}")

    def approach(tcp, yaw):
        last = None
        name = current.get("manipulated") or ""
        offsets = (
            (0.0, np.pi, np.pi / 2, -np.pi / 2)
            if name.startswith("cube_") or name in {"sphere", "sign"}
            else (0.0, np.pi)
        )
        candidates = [(offset, 0) for offset in offsets]
        if args.task == "T044":
            # Lean the wrist towards the open front, away from the carcass roof.
            candidates = []
            for tilt in (20, 30, 10, 45, 0):
                for offset in (np.pi / 2, -np.pi / 2, 0.0, np.pi):
                    candidates.append((offset, tilt))
        for offset, tilt in candidates:
            matrix = (
                Rotation.from_euler("z", yaw).as_matrix()
                @ Rotation.from_euler("x", np.deg2rad(tilt)).as_matrix()
                @ np.diag([-1, 1, -1])
                @ Rotation.from_euler("z", offset).as_matrix()
            )
            orientation = quat_wxyz_from_matrix(matrix)
            try:
                move(np.asarray(tcp) + [0, 0, 0.13], orientation)
                move(np.asarray(tcp), orientation, True)
                return orientation, offset
            except RuntimeError as error:
                last = error
                current.setdefault("planning_rejections", []).append(
                    {"phase": current["phase"], "error": str(error), "yaw_offset": offset, "tilt_deg": tilt}
                )
                if (
                    str(error) not in {"no_executable_plan", "ik_unreachable", "ik_branch_jump"}
                    and current.get("validation_stage") != "planned_trajectory"
                ):
                    raise
        raise last

    def transfer(name, target, yaw=0, target_yaw=None, receptacle=None, grasp_offset=-0.004, grasp_opening=None):
        current["phase"] = "pick"
        current["allowed"] = {name}
        current["manipulated"] = name
        print("PICK", name, "target", target, flush=True)
        source = center(name)
        if source[2] < 0.73 or np.any(np.abs(source[:2]) > 0.49):
            raise RuntimeError(f"object_left_workspace:{name}")
        source[2] += grasp_offset
        if grasp_opening is None:
            executor.open_gripper()
        else:
            executor.set_gripper(grasp_opening)
        orientation, yaw_offset = approach(source, yaw)
        executor.close_gripper()
        executor.step(steps=8)
        current["held"] = name
        old = center(name).copy()
        move(source + [0, 0, 0.15], orientation, True)
        if center(name)[2] < old[2] + 0.08:
            raise RuntimeError("grasp_not_lifted")
        current["phase"] = "transport"
        grasp_rotation = Rotation.from_quat([*orientation[1:], orientation[0]]).as_matrix()
        actual_tcp = planner.tool_position() + grasp_rotation[:, 2] * 0.1628
        local_offset = grasp_rotation.T @ (center(name) - actual_tcp)
        if target_yaw is not None:
            orientation = quat(target_yaw + yaw_offset)
        orientations = [orientation]
        if args.task in {"T044", "T045"}:
            orientations = [quat(yaw + offset) for offset in (0, np.pi / 2, np.pi, -np.pi / 2)]
        last_error = None
        for candidate_orientation in orientations:
            destination_rotation = Rotation.from_quat(
                [*candidate_orientation[1:], candidate_orientation[0]]
            ).as_matrix()
            destination = np.array(target) - destination_rotation @ local_offset
            try:
                move(destination + [0, 0, 0.15], candidate_orientation)
                orientation = candidate_orientation
                break
            except RuntimeError as error:
                last_error = error
                if args.task not in {"T044", "T045"} or (
                    str(error) != "no_executable_plan" and current.get("validation_stage") != "planned_trajectory"
                ):
                    raise
                current["planning_rejections"].append({"phase": "transport", "error": str(error)})
        else:
            raise last_error
        # Correct measured in-hand offsets rather than assuming a perfectly centred grasp.
        actual_tcp = planner.tool_position() + destination_rotation[:, 2] * 0.1628
        destination = np.array(target) - (center(name) - actual_tcp)
        current["allowed"] = {name} | ({receptacle} if receptacle is not None else set())
        move(destination, orientation, True)
        before_release = center(name).tolist()
        current["held"] = None
        executor.open_gripper()
        executor.step(steps=15)
        current.setdefault("placements", []).append({
            "name": name,
            "target": np.asarray(target).tolist(),
            "before_release": before_release,
            "after_release": center(name).tolist(),
        })
        move(destination + [0, 0, 0.13], orientation, True)
        current["allowed"] = set()
        current["manipulated"] = None

    reports = []
    failed_exports = 0
    try:
        for trial in range(args.trials):
            current.update(
                phase="reset",
                checks=0,
                min_clearance=1.0,
                min_executed_clearance=1.0,
                planning_rejections=[],
                joint_samples={},
                placements=[],
                grasp_reference=None,
                max_grasp_slip=0.0,
                allowed=set(),
                held=None,
                manipulated=None,
                max_cube_z=0.0,
                validation_stage=None,
                last_tcp_target=None,
            )
            env.recorder_manager.reset()
            env.reset()
            interface.sync_from_robot()
            executor._gripper_target = planner.cfg.gripper_open_pos
            from layouts import sample_layout

            try:
                if args.replay_results is not None:
                    assert args.trials == 1, "Layout replay is a single-case diagnostic, not a random batch"
                    saved = json.loads(args.replay_results.read_text())[args.replay_trial]
                    assert saved["task"] == args.task, "Replay task mismatch"
                    proposal, goals = saved["proposal"], saved["goals"]
                    assert set(proposal) == set(names), "Replay asset mismatch"
                else:
                    proposal, goals = sample_layout(
                        args.task,
                        names,
                        bounds,
                        rng,
                        args.workspace,
                        stack_count=args.stack_count or (3 if trial % 2 == 0 else 5),
                    )
            except (RuntimeError, ValueError) as error:
                report = {
                    "task": args.task,
                    "trial": trial,
                    "seed": args.seed,
                    "workspace": args.workspace,
                    "success": False,
                    "phase": "layout_sampling",
                    "error": str(error),
                    "contact_events": 0,
                    "forbidden_contacts": [],
                }
                reports.append(report)
                (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
                print("TRIAL_RESULT", json.dumps(report), flush=True)
                continue
            yaw = goals["yaw"]
            for name, p in proposal.items():
                put(name, p[:3], p[3:])
            if "drawer" in names:
                drawer = env.scene["drawer"]
                q = torch.tensor([[goals["initial_opening"]]], device=env.device, dtype=torch.float32)
                drawer.write_joint_state_to_sim(q, torch.zeros_like(q))
            contact_audit.reset()
            try:
                from geometry import validate_table_footprints

                env.sim.forward()
                env.scene.update(0.0)
                current["phase"] = "layout_preflight"
                validate_table_footprints(obstacle_boxes())
                clearance(planner.joint_positions(), obstacle_boxes())
                current["phase"] = "layout_settling"
                executor.step(steps=30)
            except (RuntimeError, AssertionError) as error:
                report = {
                    "task": args.task,
                    "trial": trial,
                    "seed": args.seed,
                    "workspace": args.workspace,
                    "success": False,
                    "phase": current["phase"],
                    "error": str(error),
                    "proposal": proposal,
                    "goals": goals,
                    "initial": {n: pose(n).tolist() for n in names},
                    "contact_events": contact_audit.events,
                    "forbidden_contacts": list(contact_audit.forbidden),
                }
                reports.append(report)
                (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
                print("TRIAL_RESULT", json.dumps(report), flush=True)
                continue
            from isaaclab_arena_environments.pine_wm_first20_markers import place_markers

            place_markers(args.task, goals, pose, put)
            # The random layout is now settled. Exclude reset teleports and settling
            # from demonstrations, and record this actual layout as initial_state.
            contact_audit.reset()
            env.recorder_manager.reset([0])
            env.recorder_manager.record_post_reset([0])
            initial = {n: pose(n).tolist() for n in names}
            current["max_cube_z"] = float(center("cube_40")[2]) if "cube_40" in names else 0.0
            current["joint_samples"] = {}
            report = {
                "task": args.task,
                "trial": trial,
                "seed": args.seed,
                "workspace": args.workspace,
                "initial": initial,
                "proposal": proposal,
                "goals": goals,
                "success": False,
            }
            if args.replay_results is not None:
                report["layout_replay"] = {"results": str(args.replay_results), "trial_index": args.replay_trial}
            from dataset import instruction

            report["language_instruction"] = instruction(args.task, arena.first20_task["name"], goals)
            if preview is not None:
                preview.begin(trial)
            try:
                current["phase"] = "validate"
                from geometry import validate_table_footprints

                validate_table_footprints(obstacle_boxes())
                clearance(planner.joint_positions(), obstacle_boxes())
                if args.audit_only:
                    report["audit_only"] = True
                else:
                    from primitives import run_task

                    report["success"], report["metrics"] = run_task(dict(locals(), args=args))
                    report["success"] = bool(report["success"])
                    if not report["success"]:
                        report.update(error="success_predicate_false", phase="success_check")
                    if report["success"] and contact_audit.events == 0:
                        raise RuntimeError("contact_audit_no_robot_contact_events")
            except (RuntimeError, AssertionError) as error:
                report["success"] = False
                report["error"] = str(error)
                report["phase"] = current["phase"]
                report["last_tcp_target"] = current.get("last_tcp_target")
                report["validation_stage"] = current.get("validation_stage", "layout")
            report.update(
                contact_events=contact_audit.events,
                max_robot_contact_force_n=contact_audit.max_force,
                forbidden_contacts=contact_audit.forbidden[:10],
                planning_rejections=current.get("planning_rejections", []),
                placements=current.get("placements", []),
                max_grasp_slip_m=current.get("max_grasp_slip", 0.0),
                checks=current["checks"],
                min_clearance_m=current["min_clearance"],
                min_executed_arm_camera_sphere_clearance_m=current.get("min_executed_clearance"),
                final={n: pose(n).tolist() for n in names},
            )
            episode = env.recorder_manager._episodes.get(0)
            if not args.audit_only and episode is not None and "actions" in episode.data:
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[report["success"]]], device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                if not report["success"]:
                    report["failed_recording"] = {
                        "file": "demos_failed.hdf5",
                        "demo": f"demo_{failed_exports}",
                        "diagnostic_only": True,
                    }
                    failed_exports += 1
            if preview is not None:
                report["live_preview"] = preview.finish(report["success"])
            report["workflow_stage"] = args.stage
            reports.append(report)
            (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
            if cameras_enabled:
                from PIL import Image

                for camera_name in arena.embodiment.camera_config.camera_names():
                    rgb = env.scene[camera_name].data.output["rgb"][0].cpu().numpy()[..., :3]
                    Image.fromarray(rgb.astype(np.uint8)).save(args.output / f"trial_{trial:03d}_{camera_name}.png")
            print("TRIAL_RESULT", json.dumps(report), flush=True)
            if args.stage == "preview" and report["success"]:
                print("PREVIEW_SUCCESS: awaiting user review; no further trials", flush=True)
                break
    finally:
        if preview is not None:
            preview.abort()
        env.close()
    from dataset import annotate_and_validate

    annotate_and_validate(args.output, reports, args.task)


code = 0
try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    code = 1
finally:
    launcher.app.close(exit_code=code)
