# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G2 inserts a movable Factory peg into a fixed upright sleeve."""

from dataclasses import dataclass, field

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.g2_workbench_environments import G2WorkbenchEnvironmentCfg, _G2WorkbenchEnvironment


@dataclass
class G2SleeveEnvironmentCfg(G2WorkbenchEnvironmentCfg):
    """Configure insertion of an upright peg into a fixed sleeve."""

    sleeve_xy: list[float] = field(default_factory=lambda: [-0.10, 0.0])
    """Fixed sleeve centre on the tabletop, in metres."""

    peg_xy: list[float] = field(default_factory=lambda: [0.05, -0.22])
    """Initial movable peg centre on the tabletop, in metres."""

    radial_tolerance_m: float = 0.001
    """Maximum lateral seating error; the scaled radial clearance is about 1.5 mm."""
    seating_tolerance_m: float = 0.003
    """Maximum axial gap between the peg bottom and the bore floor."""
    axis_tolerance_deg: float = 1.0
    """Maximum deviation between the upright peg and sleeve Z axes."""
    episode_length_s: float = 120.0

    def __post_init__(self):
        super().__post_init__()
        assert len(self.sleeve_xy) == 2, "sleeve_xy must contain x and y"
        assert abs(self.sleeve_xy[0]) < 0.24 and abs(self.sleeve_xy[1]) < 0.54, "Sleeve must fit on the table"
        assert len(self.peg_xy) == 2, "peg_xy must contain x and y"
        assert abs(self.peg_xy[0]) < 0.28 and abs(self.peg_xy[1]) < 0.58, "Peg must fit on the table"
        assert sum((a - b) ** 2 for a, b in zip(self.peg_xy, self.sleeve_xy)) > 0.10**2, "Objects must start separated"
        assert 0 < self.radial_tolerance_m < 0.0015, "Radial tolerance must fit the sleeve clearance"
        assert 0 < self.seating_tolerance_m < 0.075, "Seating tolerance must be smaller than the bore depth"
        assert 0 < self.axis_tolerance_deg < 90, "Axis tolerance must be between 0 and 90 degrees"
        assert self.episode_length_s > 0, "episode_length_s must be positive"


def spawn_free_sleeve(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a sleeve and disable its authored world joint in the simulation stage."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from isaaclab.sim.utils import clone
    from pxr import Usd, UsdPhysics

    @clone
    def spawn_one(prim_path, cfg, translation=None, orientation=None, **kwargs):
        prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(child)
                if not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets():
                    joint.GetJointEnabledAttr().Set(False)
        return prim

    return spawn_one(prim_path, cfg, translation, orientation, **kwargs)


@register_environment
class G2SleeveEnvironment(_G2WorkbenchEnvironment, ArenaEnvironmentFactory[G2SleeveEnvironmentCfg]):
    """Build a two-object sleeve task with G2 dual-arm control."""

    name = "g2_sleeve"
    _legacy_argparse_cfg_type = G2SleeveEnvironmentCfg
    object_layout = (
        ("peg", "peg", 0.05, -0.22, 0.0),
        ("hole", "sleeve", -0.10, 0.0, 0.0),
    )

    def build(self, cfg: G2SleeveEnvironmentCfg):
        import copy

        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.tasks.sleeve_task import SleeveTask
        from isaaclab_arena.utils.pose import Pose

        environment = super().build(cfg)
        for name in ("peg", "sleeve"):
            template = environment.scene.assets[name]
            spawn = copy.deepcopy(template.spawn_cfg_addon)
            spawn["collision_props"] = sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)
            if name == "peg":
                spawn["rigid_props"].kinematic_enabled = False
                pose = Pose(position_xyz=(*cfg.peg_xy, 0.003), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            else:
                spawn["func"] = spawn_free_sleeve
                # Replace the authored world joint with a kinematic fixture; preserve its open bore.
                spawn["rigid_props"].kinematic_enabled = True
                pose = Pose(position_xyz=(*cfg.sleeve_xy, 0.009), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            environment.scene.assets[name] = Object(
                name=name,
                usd_path=template.usd_path,
                scale=template.scale,
                object_type=ObjectType.RIGID,
                initial_pose=pose,
                spawn_cfg_addon=spawn,
            )
        environment.task = SleeveTask(
            sleeve=environment.scene.assets["sleeve"],
            peg=environment.scene.assets["peg"],
            radial_tolerance_m=cfg.radial_tolerance_m,
            seating_tolerance_m=cfg.seating_tolerance_m,
            axis_tolerance_deg=cfg.axis_tolerance_deg,
            episode_length_s=cfg.episode_length_s,
        )
        return environment
