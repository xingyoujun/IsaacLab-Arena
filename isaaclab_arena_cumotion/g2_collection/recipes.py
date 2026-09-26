# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""G2 task recipes using the shared native cuMotion session."""

import math
import torch


def stack_bowls(s):
    """Stack three bowls with measured payloads and physical release checks."""
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    base, entry, action = s.base, s.report, s.action
    position, move, step = s.position, s.move, s.step
    attached_spheres = s.attachments
    for side, source, destination in (("right", "bowl_1", "bowl_2"), ("left", "bowl_3", "bowl_1")):
        s.phase(f"{side}: approach {source}")
        initial = position(source)
        grasp = initial.clone()
        yaw = math.radians(0.0)
        grasp[0] -= 0.065 * math.cos(yaw)
        grasp[1] -= 0.065 * math.sin(yaw)
        grasp[2] = 0.065
        quat = [0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0]
        up = torch.tensor([0, 0, 0.18], device=base.device)
        move(side, grasp + up, quat, excluded=(source,))
        # Intentional gripper/object contact; arm links still avoid the table and other bowls.
        move(side, grasp, quat, excluded=(source,))
        action[0, 7 if side == "right" else 15] = -1
        step(30)
        s.phase(f"{side}: lift {source}")
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
        attached_spheres[side] = spheres.to(base.device)
        s.phase(f"{side}: place {source}")
        delta = position(destination) - position(source)
        delta[2] += 0.070
        place = grasp + up + delta
        transfer = place.clone()
        transfer[2] = max((grasp + up)[2].item(), place[2].item() + 0.02)
        move(side, transfer, quat, excluded=(source, destination))
        move(side, place, quat, excluded=(source, destination))
        action[0, 7 if side == "right" else 15] = 1
        step(20)
        attached_spheres.pop(side)
        move(side, transfer, quat, excluded=(source, destination))
        step(30)
        # Clear the shared central placement region before switching arms.
        move(side, grasp + up, quat)
    s.phase("verify stack")
    for _ in range(10):
        step()
        assert bool(base.progress_tracker.is_complete()[0]), "Task success predicate is false"
        heights = [position(name)[2].item() for name in ("bowl_2", "bowl_1", "bowl_3")]
        assert heights[0] < heights[1] < heights[2], "Bowls are not stacked in the intended order"


def peg_into_sleeve(s):
    """Insert the peg in the fixed sleeve while retaining SDF contact physics."""
    from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply_inverse, subtract_frame_transforms

    from isaaclab_arena.tasks.sleeve_task import peg_is_inserted

    base, entry, action = s.base, s.report, s.action
    position, move, step, set_phase = s.position, s.move, s.step, s.phase
    attached_spheres = s.attachments

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
        attached_spheres["right"] = spheres.to(base.device)

    initial = position("peg")
    sleeve_initial = position("sleeve")
    grasp = initial + torch.tensor([0.0, 0, 0.115], device=base.device)
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
    attach_peg()
    set_phase("align peg above sleeve")
    desired_quat = torch.tensor([0.0, 0, 0, 1], device=base.device)

    def align(bottom_height, tolerance=0.001):
        # Re-estimate the physical grasp transform at each insertion target.
        rel_p, rel_q = subtract_frame_transforms(position("peg"), base.scene["peg"].data.root_quat_w.torch[0], *tcp())
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
    attached_spheres.clear()
    p, q = tcp()
    retreat = torch.tensor([0, 0, 0.10], device=base.device)
    move("right", p + retreat, quat_wxyz(q), excluded=("sleeve", "peg"))
    step(30)
    set_phase("verify inserted peg")
    for _ in range(15):
        step()
        assert bool(peg_is_inserted(base, **s.description.task.success_params)[0]), "Peg is not fully inserted"
        assert torch.norm(position("sleeve") - sleeve_initial) < 0.001, "Fixed sleeve moved"
        assert torch.norm(base.scene["peg"].data.root_lin_vel_w.torch[0]) < 0.03, "Peg is still moving"
