# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pine workcell tasks for the annotated kettle and toaster latch mechanisms."""

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from data_engine.assets.interactions import load_interactions
from data_engine.assets.latches import reset_latches
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.assets.usdcraft_scene import resolve_asset
from isaaclab_arena.embodiments.ur7e.ur7e import UR7E_READY_JOINT_POS
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.pine_wm_environment import PineWmEnvironment, PineWmEnvironmentCfg


@dataclass
class PineWmKettleEnvironmentCfg(PineWmEnvironmentCfg):
    episode_length_s: float = 120.0
    object_position: tuple[float, float, float] = (0.0, 0.04, 0.742)
    object_yaw_deg: float = 180.0


@dataclass
class PineWmToasterEnvironmentCfg(PineWmKettleEnvironmentCfg):
    object_position: tuple[float, float, float] = (0.03, 0.07, 0.742)
    object_yaw_deg: float = -90.0


def build_appliance(factory, cfg, kind):
    """Compose original geometry and drives with a fixed test fixture and an explicit latch runtime."""
    from scipy.spatial.transform import Rotation

    arena = PineWmEnvironment().build(cfg)
    arena.name = factory.name
    arena.embodiment.set_joint_initial_pos(UR7E_READY_JOINT_POS)
    path = resolve_asset(f"usdcraft_v2_{kind}", cfg.asset_root)
    annotations = load_interactions(path)
    assert annotations is not None and annotations["latches"]
    rotation = tuple(float(value) for value in Rotation.from_euler("z", cfg.object_yaw_deg, degrees=True).as_quat())
    root_link_offset = next(
        link for link in annotations["asset"]["links"] if link["prim_path"] == annotations["asset"]["articulation_root"]
    )["rest_pose_in_root"]["position_m"]
    root_position = tuple(
        float(value) for value in (Rotation.from_quat(rotation).apply(root_link_offset) + cfg.object_position)
    )
    obj = Object(
        name=kind,
        usd_path=str(path),
        object_type=ObjectType.ARTICULATION,
        initial_pose=Pose(position_xyz=root_position, rotation_xyzw=rotation),
        spawn_cfg_addon={
            "articulation_props": sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
            "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        },
    )
    arena.scene.add_asset(obj)
    arena.interaction_annotations = annotations

    def configure(env_cfg):
        setattr(
            env_cfg.events,
            "appliance_latches",
            EventTermCfg(
                func=reset_latches, mode="reset", params={"asset_name": kind, "rules": annotations["latches"]}
            ),
        )
        if kind == "kettle":
            # The disconnected power base is already spawned by the same USD reference.
            # Register its exact prim separately so Arena records and resets it as a rigid body.
            env_cfg.scene.kettle_power_base = RigidObjectCfg(
                prim_path=obj.prim_path + "/Links/power_base",
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(pos=cfg.object_position, rot=rotation),
            )
            from isaaclab.envs.mdp import reset_root_state_uniform

            env_cfg.events.kettle_power_base_reset = EventTermCfg(
                func=reset_root_state_uniform,
                mode="reset",
                params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("kettle_power_base")},
            )
        return env_cfg

    arena.env_cfg_callback = configure
    return arena


@register_environment
class PineWmKettleEnvironment(ArenaEnvironmentFactory[PineWmKettleEnvironmentCfg]):
    name = "pine_wm_kettle_release"
    _legacy_argparse_cfg_type = PineWmKettleEnvironmentCfg

    def build(self, cfg):
        return build_appliance(self, cfg, "kettle")


@register_environment
class PineWmToasterEnvironment(ArenaEnvironmentFactory[PineWmToasterEnvironmentCfg]):
    name = "pine_wm_toaster_cancel"
    _legacy_argparse_cfg_type = PineWmToasterEnvironmentCfg

    def build(self, cfg):
        return build_appliance(self, cfg, "toaster")
