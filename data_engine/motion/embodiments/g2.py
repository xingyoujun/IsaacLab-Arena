# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create a G2 arm description with the other joints locked at measured positions."""

import copy
import hashlib
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path

from data_engine.motion.cumotion.cumotion_embodiment_cfg import CumotionEmbodimentCfg
from isaaclab_arena.assets.g2_asset_paths import planning_path


def create_g2_cumotion_cfg(env, arm, robot_yaml=None, robot_urdf=None, payload=None, output=None):
    """Build a session-local description; G2 requires an environment action executor."""
    assert env is not None, "G2 configuration requires measured robot joint positions"
    assert arm in ("left", "right")
    robot_yaml = planning_path("g2_robot_yaml", robot_yaml)
    robot_urdf = planning_path("g2_tcp_urdf", robot_urdf)
    vendor = yaml.safe_load(robot_yaml.read_text())["robot_cfg"]["kinematics"]
    robot = env.scene["robot"]
    positions = robot.data.joint_pos
    positions = positions.torch if hasattr(positions, "torch") else positions
    state = dict(zip(robot.joint_names, positions[0].tolist()))
    names = [f"idx{6 if arm == 'right' else 2}{i}_arm_{arm[0]}_joint{i}" for i in range(1, 8)]
    tool = f"gripper_{arm[0]}_center_link"
    spheres = {
        name: copy.deepcopy(values) for name, values in vendor["collision_spheres"].items() if "attached" not in name
    }
    ignore = {}
    for name, values in vendor["self_collision_ignore"].items():
        if "attached" not in name:
            ignore[name] = [value for value in values if "attached" not in value]
    if payload is not None:
        values = payload.detach().cpu().numpy()
        spheres[tool] = [dict(center=p[:3].tolist(), radius=float(p[3])) for p in values if p[3] > 0]
        ignore[tool] = [name for name in spheres if name.startswith(f"gripper_{arm[0]}_") and name != tool]
    joints = {joint.find("child").attrib["link"]: joint for joint in ET.parse(robot_urdf).getroot().findall("joint")}
    movable = set()
    for link in [*spheres, tool]:
        while link in joints:
            joint = joints[link]
            if joint.attrib["type"] != "fixed" and joint.find("mimic") is None:
                movable.add(joint.attrib["name"])
            link = joint.find("parent").attrib["link"]
    description = dict(
        cspace=names,
        default_q=[state[name] for name in names],
        acceleration_limits=[5.0] * 7,
        jerk_limits=[100.0] * 7,
        cspace_to_urdf_rules=[
            dict(name=name, rule="fixed", value=state[name]) for name in sorted(movable) if name not in names
        ],
        collision_spheres=[{name: values} for name, values in spheres.items()],
    )
    text = yaml.safe_dump(description)
    directory = Path(output) if output else Path.home() / ".cache/isaaclab_arena/g2_planning"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{arm}_{hashlib.sha256(text.encode()).hexdigest()[:20]}.yaml"
    if not path.exists():
        path.write_text(text)
    return CumotionEmbodimentCfg(
        lula_robot_description=str(path),
        robot_urdf=str(robot_urdf),
        tool_frame=tool,
        arm_joint_names=names,
        gripper_joint_names=[name for name in state if f"gripper_{arm[0]}" in name],
        gripper_open_pos=1.0,
        gripper_closed_pos=-1.0,
        self_collision_ignore=ignore,
        gripper_ramp_seconds=0.0,
        gripper_action_space="binary",
    )


def configure_g2_placement(description):
    """Keep Arena geometry placement while reserving G2 reachability for native cuMotion."""
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    params = (
        copy.deepcopy(description.placer_params)
        if description.placer_params
        else ObjectPlacerParams(solver_params=RelationSolverParams(verbose=False, save_position_history=False))
    )
    assert PlacementCheck.IK_REACHABLE not in (
        params.required_checks or set()
    ), "G2 uses native runtime reachability; a required cuRobo placement check is incompatible"
    enabled = set(PlacementCheck) if params.enabled_checks is None else set(params.enabled_checks)
    params.enabled_checks = enabled - {PlacementCheck.IK_REACHABLE}
    params.validate()
    description.placer_params = params
