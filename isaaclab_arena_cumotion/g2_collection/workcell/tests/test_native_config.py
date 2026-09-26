# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep inactive arm measurements separate between planner descriptions."""

import numpy as np
import xml.etree.ElementTree as ET
import yaml
from types import SimpleNamespace

import pytest

from isaaclab_arena_cumotion.g2 import create_g2_cumotion_cfg


def test_measured_inactive_arm_and_immutable_cache(tmp_path):
    names = [f"idx{prefix}{i}_arm_{side}_joint{i}" for side, prefix in (("l", 2), ("r", 6)) for i in range(1, 8)]
    root = ET.Element("robot", name="g2_test")
    for side in ("l", "r"):
        parent = "base_link"
        for i, name in enumerate([n for n in names if f"arm_{side}" in n], 1):
            child = f"arm_{side}_link{i}"
            joint = ET.SubElement(root, "joint", name=name, type="revolute")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            parent = child
        joint = ET.SubElement(root, "joint", name=f"tool_{side}", type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=f"gripper_{side}_center_link")
    urdf = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(urdf)
    config = tmp_path / "robot.yaml"
    config.write_text(
        yaml.safe_dump({
            "robot_cfg": {
                "kinematics": {
                    "collision_spheres": {"arm_l_link7": [{"center": [0, 0, 0], "radius": 0.01}], "arm_r_link7": []},
                    "self_collision_ignore": {},
                }
            }
        })
    )
    robot = SimpleNamespace(joint_names=names, data=SimpleNamespace(joint_pos=np.zeros((1, 14))))
    env = SimpleNamespace(scene={"robot": robot})
    first = create_g2_cumotion_cfg(env, "left", config, urdf, output=tmp_path)
    from pathlib import Path

    first_path = Path(first.lula_robot_description)
    saved = first_path.read_bytes()
    robot.data.joint_pos[0, 7:] = 0.4
    second = create_g2_cumotion_cfg(env, "left", config, urdf, output=tmp_path)
    assert second.lula_robot_description != first.lula_robot_description
    assert first_path.read_bytes() == saved
    description = yaml.safe_load(Path(second.lula_robot_description).read_text())
    assert {rule["value"] for rule in description["cspace_to_urdf_rules"]} == {0.4}
    assert first.gripper_action_space == "binary"
    assert first.arm_joint_names == names[:7]


def test_binary_gripper_rejects_direct_joint_execution():
    from isaaclab_arena_cumotion.executor import ArmExecutor

    executor = ArmExecutor.__new__(ArmExecutor)
    executor.planner = SimpleNamespace(cfg=SimpleNamespace(gripper_action_space="binary"))
    with pytest.raises(AssertionError, match="EnvActionExecutor"):
        executor.step(gripper_target=1)


def test_native_placement_preserves_geometry_and_source_config():
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena_cumotion.g2 import configure_g2_placement

    params = ObjectPlacerParams(
        enabled_checks={PlacementCheck.NO_OVERLAP, PlacementCheck.ON_RELATION, PlacementCheck.IK_REACHABLE},
        required_checks={PlacementCheck.NO_OVERLAP},
        resolve_on_reset=False,
        placement_seed=17,
    )
    description = SimpleNamespace(placer_params=params)
    configure_g2_placement(description)
    configured = description.placer_params
    assert configured.enabled_checks == {PlacementCheck.NO_OVERLAP, PlacementCheck.ON_RELATION}
    assert configured.required_checks == params.required_checks
    assert configured.resolve_on_reset is False and configured.placement_seed == 17
    assert PlacementCheck.IK_REACHABLE in params.enabled_checks


def test_native_placement_rejects_required_legacy_ik():
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena_cumotion.g2 import configure_g2_placement

    description = SimpleNamespace(placer_params=ObjectPlacerParams(required_checks={PlacementCheck.IK_REACHABLE}))
    with pytest.raises(AssertionError, match="required cuRobo placement check"):
        configure_g2_placement(description)
