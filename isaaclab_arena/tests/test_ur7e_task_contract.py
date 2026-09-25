# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the upstream termination migration preserves the RR benchmark."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_rr_termination_contract(simulation_app):
    import math
    import torch
    from types import SimpleNamespace

    import warp as wp

    from isaaclab_arena.embodiments.ur7e.demo_recorders import UR7E_ACTION_JOINT_NAMES, ur7e_demo_recorder_cfg
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
    from isaaclab_arena.tasks.ur7e_openable_task import Ur7eOpenableTask
    from isaaclab_arena_environments.ur7e_turn_toaster_knob_environment import (
        ArticraftToasterKnob,
        AstraToasterKnob,
        GptsolToasterKnob,
        UsdcraftToasterKnob,
    )
    from isaaclab_arena_environments.ur7e_workcell_environment import IdleTask

    threshold = math.radians(2)
    positions = torch.tensor([[0.0], [threshold], [-threshold], [threshold * 1.01], [-threshold * 1.01]])
    for asset_type in (UsdcraftToasterKnob, ArticraftToasterKnob, GptsolToasterKnob, AstraToasterKnob):
        asset = object.__new__(asset_type)
        env = SimpleNamespace()
        env.unwrapped = env
        env.scene = SimpleNamespace(
            articulations={
                asset.name: SimpleNamespace(
                    data=SimpleNamespace(joint_names=[asset.openable_joint_name], joint_pos=wp.from_torch(positions))
                )
            }
        )
        task = object.__new__(Ur7eOpenableTask)
        task.openable_object = asset
        task.target_joint_percentage_threshold = threshold
        task.episode_length_s = 22.0
        cfg = task.get_termination_cfg()
        assert isinstance(cfg, TaskTerminationCfg) and cfg.timeout_s == 22.0
        assert len(cfg.success) == 1 and len(cfg.success[0].predicate_sequence) == 1
        predicate = cfg.success[0].predicate_sequence[0]
        assert predicate.func.__self__ is asset
        assert predicate(env).tolist() == [False, False, False, True, True]

    idle = IdleTask(0.5).get_termination_cfg()
    assert idle.timeout_s == 0.5 and not idle.success and not idle.failures
    states_only = ur7e_demo_recorder_cfg(with_cameras=False)
    assert states_only.record_pre_step_flat_camera_observations is None
    assert states_only.record_post_step_joint_pos_target is not None
    assert ur7e_demo_recorder_cfg(with_cameras=True).record_pre_step_flat_camera_observations is not None
    assert len(UR7E_ACTION_JOINT_NAMES) == 7 and UR7E_ACTION_JOINT_NAMES[-1] == "finger_joint"
    return True


def test_rr_termination_contract():
    assert run_function_with_persistent_simulation_app(_test_rr_termination_contract)
