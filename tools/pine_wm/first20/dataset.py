# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Persist task conditions alongside the seven-joint state/action data."""

import h5py
import json
import numpy as np


def instruction(task, name, goals):
    """Resolve the language condition, including episode-specific counts and targets."""
    if task == "T031":
        return f"将最大的 {goals['stack_count']} 块积木按大件在下的次序搭在橙色目标框内，其余积木留在原处。"
    if task == "T038":
        return f"将塔上的 {goals['stack_count']} 块积木从顶向下逐个取走并放入盒子，其他物体留在原处。"
    if task == "T145":
        return f"从桌面六块积木中取恰好 {goals['count']} 块放入盒子，其余留在桌上。"
    if task == "T043":
        return f"将抽屉拉到 {goals['target_opening'] * 1000:.0f} 毫米开度，释放后保持。"
    if task == "T014":
        return "将杯柄对准橙色箭头，杯子仍放回原杯垫。"
    if task == "T012":
        return "翻转标牌，使原来朝下的一面朝上，并放入橙色目标框。"
    if task == "T017":
        return "把积木放到橙色标记盒子的后方（世界 +y 方向）。"
    if task == "T142":
        return "按几何形状分类：方块放左盘，圆柱放中盘，球放右盘；左右按世界 x 方向定义。"
    if task == "T143":
        return "将五块积木沿箭头按从小到大的顺序排列。"
    return name


def annotate_and_validate(output, results, task):
    """Check exported demonstrations against measured initial states and add conditions."""
    failed = [r for r in results if "failed_recording" in r]
    if failed:
        with h5py.File(output / "demos_failed.hdf5", "r+") as file:
            data = file["data"]
            metadata = json.loads(data.attrs["env_args"])
            metadata["env_cfg"] = json.loads((output / "environment.json").read_text())
            metadata["diagnostic_only"] = True
            data.attrs["env_args"] = json.dumps(metadata)
            assert len(data) == len(failed)
            for result in failed:
                demo = data[result["failed_recording"]["demo"]]
                demo.attrs["task_id"] = task
                demo.attrs["qualification_trial"] = result["trial"]
                demo.attrs["failure"] = result.get("error", "success_predicate_false")
                demo.attrs["goals_json"] = json.dumps(result["goals"], ensure_ascii=False)
    from isaaclab_arena.recording.alignment import pre_step_states, transition_metadata

    successes = [r for r in results if r["success"]]
    if not successes:
        return
    path = output / "demos.hdf5"
    with h5py.File(path, "r+") as file:
        data = file["data"]
        demos = sorted(data, key=lambda s: int(s.rsplit("_", 1)[1]))
        assert len(demos) == len(successes)
        metadata = json.loads(data.attrs["env_args"])
        metadata["env_cfg"] = json.loads((output / "environment.json").read_text())
        metadata["collection_contract"] = transition_metadata(
            "pine_wm_ur7e", ["realsense_d435_rgb", "wrist_a_rgb", "wrist_b_rgb"]
        )
        hashes = output / "source_hashes.json"
        if hashes.exists():
            metadata["qualification_source_hashes"] = json.loads(hashes.read_text())
        data.attrs["env_args"] = json.dumps(metadata)
        for name, result in zip(demos, successes):
            demo = data[name]
            pre_step_states(demo)
            actions = demo["actions"][:]
            assert actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all()
            assert np.allclose(actions, demo["joint_pos_target"][:], atol=1e-6)
            observed_joints = demo["obs/robot_joint_pos"][:]
            previous_states = np.concatenate([
                demo["initial_state/articulation/robot/joint_position"][:],
                demo["states/articulation/robot/joint_position"][:-1],
            ])
            assert np.allclose(observed_joints, previous_states, atol=1e-6), "Pre-step observation misalignment"
            demo.attrs["camera_state_alignment"] = "initial_state_then_previous_post_step_state"
            for group in demo["initial_state"]:
                for asset in demo["initial_state"][group]:
                    if asset in result["initial"]:
                        p = demo[f"initial_state/{group}/{asset}/root_pose"][0]
                        assert np.allclose(p, result["initial"][asset], atol=1e-6), asset

            def finite(key, item):
                if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
                    assert np.isfinite(item[()]).all(), key

            demo.visititems(finite)
            demo.attrs["task_id"] = task
            demo.attrs["language_instruction"] = result["language_instruction"]
            demo.attrs["goals_json"] = json.dumps(result["goals"], ensure_ascii=False)
            demo.attrs["success_metrics_json"] = json.dumps(result.get("metrics", {}), ensure_ascii=False)
            demo.attrs["qualification_trial"] = result["trial"]
            demo.attrs["qualification_seed"] = result["seed"]
            result["recording"] = {
                "demo": name,
                "steps": len(actions),
                "initial_state_checked": True,
                "joint_targets_checked": True,
                "observation_alignment_checked": True,
            }
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
