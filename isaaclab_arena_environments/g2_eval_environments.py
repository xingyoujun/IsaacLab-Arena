# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G2-Eval environments: one per task of the locked G2-Eval pack (``isaaclab_arena/g2_eval/lock.json``).

Each environment is named after its task (``pick_apple``) and built from the pack alone: the G2 platform of
``common/platform.json`` (the robot the data engine collects with, the table and the room), the task's objects at
their resting poses, a layout drawn each episode, and the task's success relations and time limit (``G2EvalTask``).
The environments are registered from the lock; building one needs the pack (``python -m
isaaclab_arena.g2_eval.pack download``, or ``G2_EVAL_ASSETS_DIR``).

    python isaaclab_arena/evaluation/policy_runner.py --headless --policy_type zero_action --num_episodes 1 pick_apple
"""

from __future__ import annotations

import types
from dataclasses import dataclass, make_dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.g2_eval import pack

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class G2EvalEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure a G2-Eval environment; the task itself comes from the pack."""

    arm_mode: str = "dual_arm"
    """dual_arm, or left/right for a single arm."""
    teleop_device: str | None = None
    hdr: str | None = None
    light_intensity: float = 1000.0

    def __post_init__(self):
        assert self.arm_mode in ("dual_arm", "left", "right"), "Unsupported arm_mode"
        assert self.light_intensity >= 0, "light_intensity must be non-negative"


class _G2EvalEnvironment:
    """Build the G2-Eval task ``name`` on the pack's G2 platform."""

    name: str

    def build(self, cfg: G2EvalEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """The platform, the task's objects at their resting poses and the task."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.background import Background
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.embodiments.g2.g2 import G2JointPositionActionsCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.g2_eval.task import REST_LIFT_M, G2EvalTask, rest_pose
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        loaded = pack.load_task(self.name)
        task, platform, root, pack_root = loaded["task"], loaded["platform"], loaded["root"], loaded["pack_root"]
        floor_z = float(platform["robot"]["base_position_m"][2])

        def placed(entry: dict) -> Pose:
            return Pose(
                position_xyz=tuple(float(v) for v in entry["position_m"]),
                rotation_xyzw=tuple(float(v) for v in entry["quat_xyzw"]),
            )

        table = Background(
            name="g2_table",
            usd_path=str(pack_root / platform["table"]["asset"]),
            object_min_z=floor_z,
            initial_pose=placed(platform["table"]),
            scale=tuple(float(v) for v in platform["table"]["scale"]),
        )
        room = Background(
            name="g2_workroom",
            usd_path=str(pack_root / platform["room"]["asset"]),
            object_min_z=floor_z,
            initial_pose=placed(platform["room"]),
        )
        ground = self.asset_registry.get_asset_by_name("ground_plane")(
            initial_pose=Pose(position_xyz=(0.0, 0.0, floor_z - 0.08), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())

        objects = {}
        lift = float(task.get("spawn_lift_m", REST_LIFT_M))
        for name, spec in task["objects"].items():
            position, quat_xyzw = rest_pose(spec, lift)
            scale = spec["scale"]
            scale = tuple(float(v) for v in scale) if isinstance(scale, list) else (float(scale),) * 3
            objects[name] = Object(
                name=name,
                usd_path=str(root / spec["asset"]),
                object_type=ObjectType.RIGID,
                scale=scale,
                initial_pose=Pose(
                    position_xyz=tuple(float(v) for v in position),
                    rotation_xyzw=tuple(float(v) for v in quat_xyzw),
                ),
                spawn_cfg_addon=spawn_settings(spec),
            )

        embodiment = self.asset_registry.get_asset_by_name("g2")(
            enable_cameras=cfg.enable_cameras, arm_mode=ArmMode(cfg.arm_mode)
        )
        # Joint-position control of both arms at 15 Hz, as the tasks' training data were recorded.
        assert platform["control"]["action"] == "joint_position", platform["control"]
        embodiment.action_config = G2JointPositionActionsCfg()
        # The robot the data engine collected the task's training data with, from the pack.
        robot = embodiment.scene_config.robot
        robot.spawn = robot.spawn.replace(usd_path=str(pack_root / platform["robot"]["asset"]))
        embodiment.set_initial_pose(placed({
            "position_m": platform["robot"]["base_position_m"],
            "quat_xyzw": platform["robot"]["base_quat_xyzw"],
        }))  # fmt: skip
        device = self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device else None

        def configure(env_cfg):
            from isaaclab_physx.physics import PhysxCfg

            env_cfg.viewer = ViewerCfg(eye=(1.25, -1.7, 1.1), lookat=(-0.3, 0.0, 0.2))
            # Apply external forces every solver iteration, as the Isaac Lab the tasks were collected with did by
            # default: without it TGS reports noisy velocities of resting bodies (a drill still in its bin read
            # 0.26 rad/s), and the collections' "at rest" checks never hold.
            physics = env_cfg.sim.physics or PhysxCfg()
            assert isinstance(physics, PhysxCfg), f"G2-Eval runs on PhysX, not {type(physics).__name__}"
            env_cfg.sim.physics = physics.replace(enable_external_forces_every_iteration=True)
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[table, room, ground, light, *objects.values()]),
            task=G2EvalTask(self.name, objects),
            teleop_device=device,
            env_cfg_callback=configure,
        )


def spawn_settings(spec: dict) -> dict:
    """The spawn settings of a task object (``Object``'s ``spawn_cfg_addon``): its ``physics`` (``kinematic``,
    ``mass_kg``, ``contact_offset_m``, ``rest_offset_m``, ``rigid_body``: ``high_precision``) and ``color_rgb``."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object_utils import RIGID_BODY_PROPS_HIGH_PRECISION

    physics = spec.get("physics") or {}
    unknown = set(physics) - {"kinematic", "mass_kg", "contact_offset_m", "rest_offset_m", "rigid_body"}
    assert not unknown, f"unknown physics settings {sorted(unknown)}"
    addons = {}
    rigid = None
    if physics.get("rigid_body") is not None:
        assert physics["rigid_body"] == "high_precision", f"unknown rigid_body {physics['rigid_body']!r}"
        rigid = RIGID_BODY_PROPS_HIGH_PRECISION.copy()
    if spec.get("fixed") or physics.get("kinematic"):
        rigid = (rigid or sim_utils.RigidBodyPropertiesCfg()).replace(kinematic_enabled=True)
    if rigid is not None:
        addons["rigid_props"] = rigid
    if physics.get("mass_kg") is not None:
        addons["mass_props"] = sim_utils.MassPropertiesCfg(mass=physics["mass_kg"])
    if physics.get("contact_offset_m") is not None or physics.get("rest_offset_m") is not None:
        addons["collision_props"] = sim_utils.CollisionPropertiesCfg(
            contact_offset=physics.get("contact_offset_m"), rest_offset=physics.get("rest_offset_m")
        )
    if spec.get("color_rgb") is not None:
        addons["visual_material"] = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(float(v) for v in spec["color_rgb"]), roughness=0.55, metallic=0.0
        )
    return addons


def _register(name: str) -> type:
    """Register the environment of the G2-Eval task ``name`` with a config type of its own."""
    title = "".join(part.capitalize() for part in name.split("_"))
    cfg_type = make_dataclass(f"G2Eval{title}EnvironmentCfg", [], bases=(G2EvalEnvironmentCfg,))
    cfg_type.__module__ = __name__
    cfg_type.__doc__ = f"Configure the G2-Eval environment {name}."
    attributes = {
        "name": name,
        "_legacy_argparse_cfg_type": cfg_type,
        "__module__": __name__,
        "__doc__": f"The G2-Eval task {name} on the G2 platform.",
    }
    factory_type = types.new_class(
        f"G2Eval{title}Environment",
        (_G2EvalEnvironment, ArenaEnvironmentFactory[cfg_type]),
        exec_body=lambda namespace: namespace.update(attributes),
    )
    globals()[cfg_type.__name__] = cfg_type
    globals()[factory_type.__name__] = factory_type
    return register_environment(factory_type)


for _name in pack.lock()["tasks"]:
    _register(_name)
