# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise sorting rules on synthetic states without altering the physical scene."""

import copy
import math
from types import SimpleNamespace


def check_sorting_contract(spec, corners):
    """Check correct/wrong bins, hovering, open grippers, stability and reset isolation."""
    import torch

    from isaaclab.utils.math import quat_apply

    from isaaclab_arena_environments.g2_clean_workcell_environment import sorted_objects

    def wrapped(value):
        return SimpleNamespace(torch=value)

    def body(position, quaternion):
        return SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=wrapped(position),
                root_quat_w=wrapped(quaternion),
                root_lin_vel_w=wrapped(torch.zeros(1, 3)),
                root_ang_vel_w=wrapped(torch.zeros(1, 3)),
            )
        )

    reference = spec["bin_geometry_reference"]
    ratio = torch.tensor(spec["bin_scale"]) / torch.tensor(reference["scale"])
    cavity = torch.tensor(reference["inner_xy_bounds_m"]) * ratio[:2]
    floor = reference["floor_z_m"] * ratio[2]
    scene = {}
    for name, item in spec["bins"].items():
        angle = math.radians(spec["bin_yaw_deg"]) / 2
        scene[name] = body(torch.tensor([[*item["xy"], 0.0]]), torch.tensor([[0, 0, math.sin(angle), math.cos(angle)]]))
    for name, item in spec["objects"].items():
        vertices = torch.tensor(corners[name])
        target = scene[item["destination"]].data
        offset = torch.cat(
            (cavity.mean(0) - (vertices.amin(0) + vertices.amax(0))[:2] / 2, (floor - vertices[:, 2].min()).reshape(1))
        )
        scene[name] = body(
            target.root_pos_w.torch + quat_apply(target.root_quat_w.torch, offset[None]),
            target.root_quat_w.torch.clone(),
        )
    scene["robot"] = SimpleNamespace(
        joint_names=[
            "idx41_gripper_l_outer_joint1",
            "idx31_gripper_l_inner_joint1",
            "idx81_gripper_r_outer_joint1",
            "idx71_gripper_r_inner_joint1",
        ],
        data=SimpleNamespace(joint_pos=wrapped(torch.tensor([[0.8, -0.8, 0.8, -0.8]]))),
    )
    template = SimpleNamespace(
        scene=scene, num_envs=1, device="cpu", step_dt=1 / 15, episode_length_buf=torch.tensor([2])
    )
    steps = math.ceil(spec["success"]["stable_seconds"] / template.step_dt)
    results = {}

    def run_case(name, change, expected):
        env = copy.deepcopy(template)
        change(env)
        for _ in range(steps):
            result = sorted_objects(env, spec, corners)
            env.episode_length_buf += 1
        assert result.item() == expected, name
        results[name] = True
        return env

    good = run_case("correct_bins_stable", lambda env: None, True)
    run_case(
        "wrong_bin_rejected",
        lambda env: env.scene["metal_stock"].data.root_pos_w.torch.__setitem__((0, 1), -0.37),
        False,
    )
    run_case(
        "hovering_rejected", lambda env: env.scene["metal_stock"].data.root_pos_w.torch.__setitem__((0, 2), 0.2), False
    )
    run_case("closed_gripper_rejected", lambda env: env.scene["robot"].data.joint_pos.torch.zero_(), False)
    run_case("moving_object_rejected", lambda env: env.scene["drill"].data.root_lin_vel_w.torch.fill_(0.1), False)
    run_case(
        "outside_wall_rejected", lambda env: env.scene["drill"].data.root_pos_w.torch.__setitem__((0, 0), 0.4), False
    )
    fresh = copy.deepcopy(template)
    assert not sorted_objects(fresh, spec, corners).item(), "One frame must not satisfy stability"
    good.episode_length_buf.zero_()
    assert not sorted_objects(good, spec, corners).item(), "Reset must clear prior success hold"
    results.update(one_frame_rejected=True, reset_clears_stability=True)
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena_environments.g2_clean_workcell_environment import CleanWorkcellTask

    cfg = CleanWorkcellTask(spec, corners).get_termination_cfg()
    env = copy.deepcopy(template)
    tracker = ProgressTracker(
        cfg.success,
        num_envs=1,
        device="cpu",
        env=env,
        desired_subtask_success_state=cfg.desired_subtask_success_state,
    )
    for _ in range(steps):
        tracker.step(env, step_index=env.episode_length_buf)
        count = env._workcell_settled_steps.clone()
        tracker.step(env, step_index=env.episode_length_buf)
        assert torch.equal(env._workcell_settled_steps, count), "Repeated evaluation advanced stability"
        env.episode_length_buf += 1
    assert tracker.is_complete().item(), "Managed task never completed"
    results.update(managed_completion=True, repeated_step_not_counted_twice=True)
    env.scene["drill"].data.root_lin_vel_w.torch.fill_(0.1)
    tracker.step(env, step_index=env.episode_length_buf)
    assert not tracker.is_complete().item(), "Managed task latched stale success"
    results["managed_success_remains_live"] = True
    env.scene["drill"].data.root_lin_vel_w.torch.zero_()
    env.episode_length_buf.zero_()
    tracker.reset(torch.tensor([0]))
    tracker.step(env, step_index=env.episode_length_buf)
    assert not tracker.is_complete().item(), "Managed reset retained success"
    results["managed_reset_clears_success"] = True
    return results
