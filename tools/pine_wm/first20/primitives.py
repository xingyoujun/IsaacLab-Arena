# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physical task sequences and measured success predicates for qualification."""

import numpy as np
from scipy.spatial.transform import Rotation


def run_task(ctx):  # noqa: C901 - dispatches the twenty distinct qualification sequences.
    """Execute one proposal; all motion goes through the runner's checked cuMotion paths."""
    task = ctx["args"].task
    center, pose, transfer = ctx["center"], ctx["pose"], ctx["transfer"]
    executor, move, quat = ctx["executor"], ctx["move"], ctx["quat"]
    current, env, vec = ctx["current"], ctx["env"], ctx["vec"]
    approach = ctx["approach"]
    goals, names, bounds = ctx["goals"], ctx["names"], ctx["bounds"]
    yaw = goals["yaw"]
    metrics = {}

    def obj_yaw(n):
        return float(Rotation.from_quat(pose(n)[3:]).as_euler("xyz")[2])

    def stable(n):
        return float(np.linalg.norm(vec(env.scene[n].data.root_lin_vel_w)[0])) < 0.01

    def inactive_cubes_undisturbed(active):
        displacements = {
            name: float(np.linalg.norm(pose(name)[:3] - ctx["initial"][name][:3]))
            for name in names
            if name.startswith("cube_") and name not in active
        }
        metrics["inactive_cube_displacements_m"] = displacements
        return max(displacements.values(), default=0.0) < 0.01

    def in_container(n, container):
        p = pose(container)
        local = Rotation.from_quat(p[3:]).inv().apply(center(n) - p[:3])
        half = np.abs((Rotation.from_quat(p[3:]).inv() * Rotation.from_quat(pose(n)[3:])).as_matrix()) @ (
            (bounds[n][1] - bounds[n][0]) / 2
        )
        ok = (
            abs(local[0]) + half[0] < bounds[container][1, 0] * (0.116 / 0.12)
            and abs(local[1]) + half[1] < bounds[container][1, 1] * (0.086 / 0.09)
            and 0.003 < local[2] < bounds[container][1, 2] + half[2] + 0.003
        )
        metrics[n] = {"local": local.tolist(), "stable": stable(n)}
        return bool(ok and stable(n))

    def drawer_grasp(grip, drawer_yaw):
        from isaaclab_arena_cumotion.grasps import quat_wxyz_from_matrix

        executor.set_gripper(0.35)
        last = None
        for tilt in (30, 45, 20, 55):
            for flip in (0, np.pi):
                matrix = (
                    Rotation.from_euler("z", drawer_yaw).as_matrix()
                    @ Rotation.from_euler("x", np.deg2rad(tilt)).as_matrix()
                    @ np.diag([-1, 1, -1])
                    @ Rotation.from_euler("z", flip).as_matrix()
                )
                orientation = quat_wxyz_from_matrix(matrix)
                try:
                    move(grip + [0, 0, 0.16], orientation)
                    move(grip, orientation, True)
                    metrics["grasp_tilt_deg"] = tilt
                    return orientation
                except RuntimeError as error:
                    last = error
                    if (
                        str(error) not in {"no_executable_plan", "ik_unreachable", "ik_branch_jump"}
                        and current.get("validation_stage") != "planned_trajectory"
                    ):
                        raise
        raise last

    def into(n, container, offset=(0, 0)):
        p = pose(container)
        h = (bounds[n][1, 2] - bounds[n][0, 2]) / 2
        dest = p[:3] + Rotation.from_quat(p[3:]).apply([*offset, 0.004 + h + 0.005])
        if container == "box":
            dest[2] = p[2] + bounds[container][1, 2] + h + 0.018
        grasp_offset = 0.004 if task == "T038" and n in {"cube_25", "cube_32"} else -0.004
        transfer(n, dest, obj_yaw(n), receptacle=container, grasp_offset=grasp_offset)

    if task in {"T001", "T002", "T004"}:
        name = {"T001": "cube_40", "T002": "bar", "T004": "cylinder"}[task]
        initial_fixture = {n: pose(n)[:3].copy() for n in ctx["fixtures"]}
        into(name, "tray")
        executor.step(steps=15)
        ok = in_container(name, "tray")
        for n, p in initial_fixture.items():
            drift = float(np.linalg.norm(pose(n)[:3] - p))
            metrics[n + "_drift_m"] = drift
            ok &= drift < 0.002
        return ok, metrics
    if task in {"T016", "T017"}:
        target = np.array([*goals["target"], 0.762])
        transfer("cube_40", target, yaw)
        executor.step(steps=15)
        error = float(np.linalg.norm(center("cube_40")[:2] - target[:2]))
        metrics["position_error_m"] = error
        return error < 0.015 and stable("cube_40"), metrics
    if task == "T014":
        original = center("mug").copy()
        transfer("mug", original, yaw, target_yaw=goals["target_yaw"], receptacle="coaster")
        executor.step(steps=15)
        angle = float(
            abs(np.arctan2(np.sin(obj_yaw("mug") - goals["target_yaw"]), np.cos(obj_yaw("mug") - goals["target_yaw"])))
        )
        error = float(np.linalg.norm(center("mug")[:2] - original[:2]))
        metrics.update(angle_error_rad=angle, position_error_m=error)
        return angle < np.deg2rad(10) and error < 0.012 and stable("mug"), metrics
    if task in {"T031", "T038", "T143"}:
        cubes = sorted([n for n in names if n.startswith("cube_")], key=lambda n: int(n.split("_")[1]))
        if task in {"T031", "T038"}:
            cubes = goals["active_cubes"]
        target = np.array(goals["target"])
        z = 0.74
        if task == "T031":
            below = None
            for n in reversed(cubes):
                h = float(n.split("_")[1]) / 1000
                transfer(n, [*target, z + h / 2 + (0.018 if below else 0.002)], obj_yaw(n), receptacle=below)
                z += h
                below = n
            executor.step(steps=30)
            z = 0.74
            ok = True
            for n in reversed(cubes):
                h = float(n.split("_")[1]) / 1000
                desired = np.array([*target, z + h / 2])
                error = float(np.linalg.norm(center(n) - desired))
                z += h
                metrics[n] = {"error_m": error}
                ok &= error < 0.01 and stable(n)
            return bool(ok and inactive_cubes_undisturbed(cubes)), metrics
        if task == "T038":
            for i, n in enumerate(cubes):
                into(
                    n,
                    "box",
                    offset=goals.get(
                        "container_offsets_m", [(-0.075 + (j % 3) * 0.075, -0.04 + (j // 3) * 0.08) for j in range(5)]
                    )[i],
                )
                for other in cubes[i + 1 :]:
                    if np.linalg.norm(center(other)[:2] - ctx["initial"][other][:2]) > 0.01:
                        raise RuntimeError("remaining_tower_disturbed")
            executor.step(steps=30)
            return all(in_container(n, "box") for n in cubes) and inactive_cubes_undisturbed(cubes), metrics
        for i, n in enumerate(cubes):
            h = float(n.split("_")[1]) / 1000
            transfer(n, [target[0] + (i - 2) * goals.get("row_pitch_m", 0.12), target[1], 0.742 + h / 2], obj_yaw(n))
        executor.step(steps=15)
        xs = [center(n)[0] for n in cubes]
        metrics["x_positions"] = xs
        errors = [
            float(
                np.linalg.norm(
                    center(n)
                    - [
                        target[0] + (i - 2) * goals.get("row_pitch_m", 0.12),
                        target[1],
                        0.74 + float(n.split("_")[1]) / 2000,
                    ]
                )
            )
            for i, n in enumerate(cubes)
        ]
        metrics["target_errors_m"] = errors
        return (
            all(xs[i + 1] - xs[i] > 0.05 for i in range(4)) and max(errors) < 0.015 and all(stable(n) for n in cubes),
            metrics,
        )
    if task == "T145":
        count = goals["count"]
        for i in range(count):
            into(
                f"cube_{i}",
                "box",
                offset=goals.get(
                    "container_offsets_m", [(-0.065 + (j % 3) * 0.065, -0.032 + (j // 3) * 0.064) for j in range(6)]
                )[i],
            )
        executor.step(steps=15)
        actual = sum(in_container(f"cube_{i}", "box") for i in range(6))
        metrics.update(requested_count=count, actual_count=actual)
        untouched = [
            float(np.linalg.norm(pose(f"cube_{i}")[:3] - ctx["initial"][f"cube_{i}"][:3])) for i in range(count, 6)
        ]
        metrics["unselected_displacement_m"] = untouched
        return (
            actual == count
            and all(in_container(f"cube_{i}", "box") for i in range(count))
            and max(untouched, default=0) < 0.01,
            metrics,
        )
    if task == "T142":
        for i, n in enumerate(["cube_40", "cylinder", "sphere"]):
            into(n, f"tray_{i}")
        executor.step(steps=15)
        return all(in_container(n, f"tray_{i}") for i, n in enumerate(["cube_40", "cylinder", "sphere"])), metrics
    if task == "T063":
        current["phase"] = "press"
        current["allowed"] = {"button"}
        current["manipulated"] = "button"
        button = env.scene["button"]
        p = pose("button")[:3]
        tip = p + [0, 0, 0.043]
        executor.close_gripper()
        orient = quat(yaw)
        move(tip + [0, 0, 0.12], orient)
        move(tip + [0, 0, -0.011], orient, True)
        q = max(current["joint_samples"].get("button", [0]))
        metrics["peak_press_m"] = q
        move(tip + [0, 0, 0.12], orient, True)
        executor.step(steps=15)
        reset = float(vec(button.data.joint_pos)[0, 0])
        metrics["reset_m"] = reset
        presses = 0
        active = False
        for position in current["joint_samples"].get("button", []):
            if not active and position >= 0.008:
                presses += 1
                active = True
            elif active and position <= 0.002:
                active = False
        metrics["press_count"] = presses
        return q >= 0.008 and reset <= 0.002 and presses == 1, metrics
    if task == "T021":
        current["phase"] = "push"
        current["allowed"] = {"cube_40", "push_mat"}
        current["manipulated"] = "cube_40"
        block = center("cube_40")
        orientation = quat(np.pi / 2)
        executor.close_gripper()
        # Closed pads approach the -X face; table contact is still forbidden.
        contact = block + [-0.027, 0, -0.005]
        orientation, _ = approach(contact, np.pi / 2)
        end = np.array([*goals["target"], contact[2]]) + [-0.027, 0, 0]
        move(end, orientation, True)
        move(end + [0, 0, 0.1], orientation, True)
        executor.step(steps=15)
        error = float(np.linalg.norm(center("cube_40")[:2] - goals["target"]))
        metrics["error_m"] = error
        lift = current.get("max_cube_z", 0.0) - ctx["initial"]["cube_40"][2]
        metrics["max_lift_m"] = lift
        return error < 0.02 and lift < 0.005 and stable("cube_40"), metrics
    if task in {"T041", "T042", "T043"}:
        current["phase"] = "drawer"
        current["allowed"] = {"drawer"}
        current["manipulated"] = "drawer"
        drawer = env.scene["drawer"]
        p = pose("drawer")
        R = Rotation.from_quat(p[3:])
        opening = R.apply([0, -1, 0])
        q0 = float(vec(drawer.data.joint_pos)[0, 0])
        target = goals["target_opening"]
        # New asset has a horizontal pull bar, not the old spherical knob.
        grip = p[:3] + R.apply([0, -0.177 - q0, 0.070])
        orientation = quat(yaw)
        executor.open_gripper()
        orientation = drawer_grasp(grip, yaw)
        executor.close_gripper()
        executor.step(steps=8)
        end = grip + opening * (target - q0)
        move(end, orientation, True)
        executor.set_gripper(0.35)
        move(end + [0, 0, 0.13], orientation, True)
        executor.open_gripper()
        executor.step(steps=15)
        q = float(vec(drawer.data.joint_pos)[0, 0])
        hold = current["joint_samples"].get("drawer", [])[-15:]
        drift = float(np.linalg.norm(pose("drawer")[:3] - ctx["initial"]["drawer"][:3]))
        metrics.update(opening_m=q, target_m=target, hold_openings_m=hold, carcass_displacement_m=drift)
        ok = q > 0.075 if task == "T041" else (q < 0.005 if task == "T042" else abs(q - target) < 0.005)
        ok &= drift < 0.002 and (task != "T043" or max(abs(value - target) for value in hold) < 0.005)
        return ok, metrics
    if task in {"T011", "T012"}:
        from isaaclab_arena_cumotion.grasps import quat_wxyz_from_matrix

        n = "bottle" if task == "T011" else "sign"
        current["phase"] = "reorient"
        current["allowed"] = {n}
        current["manipulated"] = n
        initial_q = Rotation.from_quat(pose(n)[3:])
        # Bottle is grasped high on its body so the horizontal final wrist clears the table.
        source = (
            pose(n)[:3] + initial_q.apply([0, 0, 0.095]) + [0, 0, -0.005]
            if task == "T011"
            else center(n) + [0, 0, 0.001]
        )
        orientation = quat(yaw + np.pi if task == "T011" else yaw)
        executor.open_gripper()
        orientation, _ = approach(source, yaw + np.pi if task == "T011" else yaw)
        executor.close_gripper()
        executor.step(steps=8)
        current["held"] = n
        move(source + [0, 0, 0.25], orientation, True)
        if center(n)[2] < 0.85:
            raise RuntimeError("reorientation_grasp_failed")
        lifted_q = Rotation.from_quat(pose(n)[3:])
        tool_R = Rotation.from_quat([*orientation[1:], orientation[0]])
        dest = np.array([*goals["target"], 0.855 if task == "T011" else 0.98])
        # A sign released directly above an inverted gripper would hit its palm.
        # Tilt past vertical; gravity completes the physical flip onto the table.
        metrics["commanded_sign_tilt_deg"] = 120 if task == "T012" else None
        last_error = None
        requested_target = np.asarray(goals["target"], dtype=float)
        placement_targets = [requested_target]
        if task == "T011" and goals.get("allow_target_relocation", True):
            # Uprighting has no marked destination in the supplied specification.
            # Keep broad initial sampling; choose a reachable free placement if needed.
            xmin, xmax, ymin, ymax = ctx["args"].workspace
            front_y = float(np.clip(0.12, ymin, ymax))
            placement_targets.extend([
                np.array([requested_target[0], front_y]),
                np.array([(xmin + xmax) / 2, ymin + 0.75 * (ymax - ymin)]),
            ])
            goals["requested_target"] = requested_target.tolist()
        found = False
        for placement_target in placement_targets:
            dest[:2] = placement_target
            for offset in (0, np.pi / 2, -np.pi / 2, np.pi, np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4):
                desired = Rotation.from_euler("z", yaw + offset)
                if task == "T012":
                    desired = desired * Rotation.from_euler("y", np.deg2rad(120))
                final_orientation = quat_wxyz_from_matrix((desired * lifted_q.inv() * tool_R).as_matrix())
                try:
                    candidate_dest = np.array([*placement_target, dest[2]])
                    move(candidate_dest + [0, 0, 0.12 if task == "T011" else 0], final_orientation)
                    edge_shift = float((bounds[n][1, 0] - bounds[n][0, 0]) / 2) * (1 + np.cos(np.deg2rad(120)))
                    landing_shift = (
                        Rotation.from_euler("z", yaw + offset).apply([edge_shift, 0, 0])
                        if task == "T012"
                        else np.zeros(3)
                    )
                    candidate_dest[:2] += placement_target - pose(n)[:2] - landing_shift[:2]
                    move(candidate_dest, final_orientation, True)
                    dest = candidate_dest
                    found = True
                    goals["target"] = placement_target.tolist()
                    break
                except RuntimeError as error:
                    last_error = error
                    current.setdefault("planning_rejections", []).append({
                        "phase": current["phase"],
                        "error": str(error),
                        "yaw_offset": offset,
                        "placement_target": placement_target.tolist(),
                    })
                    if (
                        str(error) not in {"no_executable_plan", "ik_unreachable", "ik_branch_jump"}
                        and current.get("validation_stage") != "planned_trajectory"
                    ):
                        raise
            if found:
                break
        if not found:
            raise last_error
        current["held"] = None
        executor.open_gripper()
        executor.step(steps=30)
        if task == "T011":
            move(dest + [0, 0, 0.14], final_orientation, True)
        else:
            axis = Rotation.from_quat(pose(n)[3:]).apply([0, 0, 1])
            if axis[2] > -np.cos(np.deg2rad(8)):
                raise RuntimeError("flip_face_not_achieved")
            metrics["pre_regrasp_error_m"] = float(np.linalg.norm(pose(n)[:2] - goals["target"]))
            transfer(n, [*goals["target"], center(n)[2]], obj_yaw(n), grasp_offset=0.001)
        body_axis = Rotation.from_quat(pose(n)[3:]).apply([0, 0, 1])
        angle = float(np.arccos(np.clip(body_axis[2] if task == "T011" else -body_axis[2], -1, 1)))
        error = float(np.linalg.norm(pose(n)[:2] - goals["target"]))
        metrics.update(angle_error_rad=angle, position_error_m=error)
        extent = np.abs(Rotation.from_quat(pose(n)[3:]).as_matrix()) @ ((bounds[n][1] - bounds[n][0]) / 2)
        bottom = pose(n)[2] if task == "T011" else center(n)[2] - extent[2]
        metrics["bottom_height_m"] = float(bottom)
        target_ok = error < 0.025
        if task == "T012":
            extent = np.abs(Rotation.from_quat(pose(n)[3:]).as_matrix()) @ ((bounds[n][1] - bounds[n][0]) / 2)
            target_ok = bool(np.all(np.abs(center(n)[:2] - goals["target"]) + extent[:2] < 0.0585))
            metrics["inside_target_frame"] = target_ok
        return angle < np.deg2rad(8) and target_ok and abs(bottom - 0.74) < 0.003 and stable(n), metrics
    if task in {"T044", "T045"}:
        drawer = env.scene["drawer"]
        p = pose("drawer")
        R = Rotation.from_quat(p[3:])
        q = float(vec(drawer.data.joint_pos)[0, 0])
        if task == "T044":
            tray = pose("tray")
            destination = tray[:3] + Rotation.from_quat(tray[3:]).apply([0, 0, 0.004 + 0.0125 + 0.005])
            transfer(
                "cube_25",
                destination,
                yaw,
                target_yaw=yaw,
                receptacle="tray",
                grasp_offset=0.010,
                grasp_opening=0.35,
            )
            executor.step(steps=15)
            return in_container("cube_25", "tray") and float(vec(drawer.data.joint_pos)[0, 0]) > 0.13, metrics
        destination = p[:3] + R.apply([0, -0.067 - q, 0.135])
        transfer("cube_25", destination, obj_yaw("cube_25"), target_yaw=yaw + np.pi / 2, receptacle="drawer")
        current["phase"] = "close_drawer"
        current["allowed"] = {"drawer"}
        current["manipulated"] = "drawer"
        grip = p[:3] + R.apply([0, -0.177 - q, 0.070])
        orientation = quat(yaw)
        executor.open_gripper()
        orientation = drawer_grasp(grip, yaw)
        executor.close_gripper()
        move(grip - R.apply([0, -q, 0]), orientation, True)
        executor.set_gripper(0.35)
        move(grip - R.apply([0, -q, 0]) + [0, 0, 0.13], orientation, True)
        executor.open_gripper()
        executor.step(steps=15)
        opening = float(vec(drawer.data.joint_pos)[0, 0])
        local = R.inv().apply(center("cube_25") - p[:3])
        metrics.update(opening_m=opening, object_local=local.tolist())
        return (
            opening < 0.005 and abs(local[0]) < 0.097 and -0.10 < local[1] < 0.066 and 0.04 < local[2] < 0.09,
            metrics,
        )
    raise RuntimeError(f"task_driver_pending:{task}")
