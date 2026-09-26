# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independently audit workcell raw timing, measured lifts, carries and retained placements."""

import argparse
import h5py
import json
import numpy as np
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation

from isaaclab_arena_examples.g2_workcell.carry_audit import audit_carry
from isaaclab_arena_examples.g2_workcell.placement import evaluate_placement


def audit(root):
    """Validate raw data and every completed placement; never promote a partial attempt."""
    report = json.loads((root / "reach_report.json").read_text())
    status = json.loads((root / "status.json").read_text())
    geometry = json.loads((root / "effective_geometry.json").read_text())
    config = yaml.safe_load((root / "motion_config.yaml").read_text())
    trace = dict(np.load(root / "step_trace.npz"))
    count = len(trace["actions"])
    assert report["reset_count"] == 1
    for key in (
        "actions",
        "q_before",
        "q_after",
        "object_positions",
        "object_quaternions",
        "tcp_positions",
        "tcp_quaternions",
    ):
        assert np.isfinite(trace[key]).all(), key
    np.testing.assert_allclose(trace["q_before"][1:], trace["q_after"][:-1], atol=1e-6, rtol=0)
    successful = bool(status["task_success"])
    raw = root / ("episodes.hdf5" if successful else "episodes_failed.hdf5")
    with h5py.File(raw) as file:
        demos = list(file["data"])
        assert len(demos) == 1
        demo = file["data"][demos[0]]
        assert bool(demo.attrs["success"]) == successful
        assert demo["actions"].shape == (count, 16)
        np.testing.assert_array_equal(demo["actions"][:], trace["actions"])
        np.testing.assert_allclose(demo["core/joint_position"][:], trace["q_before"], atol=1e-6, rtol=0)
        for name, width in (("joint_position", 46), ("joint_velocity", 46), ("eef_pose", 14), ("eef_pose_world", 14)):
            values = demo[f"core/{name}"][:]
            assert values.shape == (count, width) and np.isfinite(values).all(), name
        eef = demo["core/eef_pose_world"][:]
        np.testing.assert_allclose(eef[1:, :3], trace["tcp_positions"][:-1, :3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(eef[1:, 7:10], trace["tcp_positions"][:-1, 3:], atol=1e-6, rtol=0)
        for span in (slice(3, 7), slice(10, 14)):
            np.testing.assert_allclose(np.linalg.norm(eef[:, span], axis=1), 1, atol=1e-5, rtol=0)
        assert "initial_state" in demo and "states" in demo
    result = {
        "raw_audit_pass": True,
        "task_success": successful,
        "steps": count,
        "simulated_seconds": count * float(trace["step_dt"]),
        "objects": {},
    }
    names = list(trace["object_names"])
    result["maximum_bin_displacement_m"] = {
        name: float(
            np.linalg.norm(
                trace["object_positions"][:, names.index(name)] - trace["object_positions"][0, names.index(name)],
                axis=1,
            ).max()
        )
        for name in geometry["bins"]
    }
    for name in report["completed_tools"]:
        destination = config["mapping"][name]
        vertices = np.load(root / "geometry" / f"{name}_mesh.npz")["vertices"]
        oi, bi = names.index(name), names.index(destination)
        hold = np.flatnonzero((trace["tool"] == name) & (trace["phase"] == "hold"))
        assert len(hold) >= 30
        first = np.flatnonzero(trace["tool"] == name)[0]
        gain = float(trace["object_positions"][hold[-1], oi, 2] - trace["object_positions"][first, oi, 2])
        clearance = min(
            (Rotation.from_quat(trace["object_quaternions"][i, oi]).apply(vertices) + trace["object_positions"][i, oi])[
                :, 2
            ].min()
            for i in hold
        )
        assert gain > 0.07 and clearance > 0.05
        result["objects"][name] = {"lift_gain_m": gain, "minimum_clearance_m": float(clearance)}
        if report["collection_stage"] == "grasp_lift":
            continue
        result["objects"][name]["carry"] = audit_carry(trace, name, config["tools"][name]["arm"])
        verify = np.flatnonzero((trace["tool"] == name) & (trace["phase"] == "verify_placement"))
        assert len(verify) >= 30
        samples = []
        for i in range(verify[0], count):
            p, q = trace["object_positions"][i, oi], trace["object_quaternions"][i, oi]
            points = (
                Rotation.from_quat(trace["object_quaternions"][i, bi])
                .inv()
                .apply(Rotation.from_quat(q).apply(vertices) + p - trace["object_positions"][i, bi])
            )
            samples.append(
                dict(
                    position=p.tolist(),
                    quaternion_xyzw=q.tolist(),
                    minimum_bin=points.min(0).tolist(),
                    maximum_bin=points.max(0).tolist(),
                )
            )
        metrics = evaluate_placement(samples, geometry["bins"][destination])
        assert metrics["passed"], (name, metrics)
        result["objects"][name]["placement"] = metrics
    if successful:
        assert max(result["maximum_bin_displacement_m"].values()) < 0.005
        assert set(report["completed_tools"]) == set(config["mapping"])
        final = trace["phase"] == "verify_all_categories"
        assert final.sum() >= 30
        spec = yaml.safe_load((root / "sources" / "clean_workcell_table.yaml").read_text())
        limits = spec["success"]
        required = int(np.ceil(limits["stable_seconds"] / float(trace["step_dt"])))
        terminal = trace["object_velocities"][-required:]
        assert len(terminal) == required
        for name in spec["objects"]:
            velocity = terminal[:, names.index(name)]
            assert np.all(np.linalg.norm(velocity[:, :3], axis=1) < limits["max_linear_speed_m_s"]), name
            assert np.all(np.linalg.norm(velocity[:, 3:], axis=1) < limits["max_angular_speed_rad_s"]), name
        for name in spec["bins"]:
            velocity = terminal[:, names.index(name), :3]
            assert np.all(np.linalg.norm(velocity, axis=1) < limits["max_linear_speed_m_s"]), name
        result["terminal_stability_seconds"] = required * float(trace["step_dt"])
        for side, outer, inner, offset in (("l", "idx41", "idx31", 3), ("r", "idx81", "idx71", 0)):
            joints = list(trace["joint_names"])
            assert np.all(trace["q_after"][final, joints.index(f"{outer}_gripper_{side}_outer_joint1")] > 0.65)
            assert np.all(trace["q_after"][final, joints.index(f"{inner}_gripper_{side}_inner_joint1")] < -0.65)
            assert (
                trace["tcp_positions"][final, offset + 2].min()
                > max(b["rim_max_z_m"] for b in geometry["bins"].values()) + 0.07
            )
    (root / "collection_audit.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(audit(parser.parse_args().run_dir)), flush=True)
