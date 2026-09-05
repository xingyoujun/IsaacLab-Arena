# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared stage, configuration and controller set-up for the Agibot table-top benchmark tasks.

Every Agibot task in this package stands the robot at the same spot in front of the same RoboDojo
table, in the same lit room, and drives it through the same teleoperation stack. This module is
the single place those facts live, so a new task cannot drift from the others by accident and a
change to the standard applies to all of them at once. A task environment subclasses
``AgibotTabletopEnvironmentCfg`` for its configuration, builds its stage with
``build_tabletop_stage``, and calls ``install_agibot_control_stack`` first thing in its
``env_cfg_callback``; everything task-specific (objects, layout, jitter, success) is its own.

The process these defaults come out of is written up in ``docs/agibot/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

# --- The stage --------------------------------------------------------------------------------

ROBOT_POSITION_XYZ = (-0.60, 0.0, 0.0)
"""Agibot base. Same standoff as tabletop_place_upright, the reference Agibot environment."""

TABLE_TOP_Z = 0.6232
"""Work-surface height, measured (a block dropped on the table comes to rest with its bottom
here), at the arm's natural working height. Every object's placement is expressed against it."""

TABLE_POSITION_X = -0.365 + 0.5 * 1.1
"""x of the table's centre: the near edge sits at -0.365, where the robot is known to clear it,
and the slab is 1.1 m deep."""

REACH_X_BAND_M = (0.35, 0.45)
"""The x band both arms were measured to reach at table height with a leaned top-down grasp.
The band has an inner limit as well as an outer one: an object at 0.30 is too close for the arm
to fold in onto. Lay graspable objects out inside it, and clamp jitter into it."""

DOME_LIGHT_HDR = "brown_photostudio_robolab"
DOME_LIGHT_INTENSITY = 1000.0
"""RoboDojo lights every task with this HDRI at 1000; Arena's stock grey dome at 500 leaves the
table nearly unreadable from the head view (mean pixel 25/255 against 113/255 here)."""

# --- The controller stack ---------------------------------------------------------------------

DEFAULT_ARM_EFFORT_LIMIT = 300.0
"""Torque ceiling for both arms, in N m, applied at the task level (the embodiment keeps the
shipped 1000-2000). See ``AgibotTabletopEnvironmentCfg.arm_effort_limit``."""

DEFAULT_GRIPPER_RAMP_SECONDS = 200 / 120
"""Full open-to-closed travel time of the teleop grippers' target; equals the cuMotion executor's
ramp so a teleoperated close and a scripted close load the object the same way."""


@dataclass
class AgibotTabletopEnvironmentCfg(ArenaEnvironmentCfg):
    """Configuration every Agibot table-top task shares. Subclass it and add the task's own fields.

    The values are the standard, not suggestions: they were each measured (see ``docs/agibot``),
    and a task that needs to move one should say why in its own field docstring. In particular
    the robot itself is never tuned per task -- ``arm_effort_limit`` is the one sanctioned knob,
    and its default is the same for all tasks.
    """

    background: str = "robodojo_table"
    """The RoboDojo mahogany table (1.1 x 1.4 x 0.05 m slab, friction 0.8), authored locally."""

    embodiment: str = "agibot"

    teleop_device: str | None = "dual_arm_keyboard"
    """Must emit as many values as the arm mode consumes: ``dual_arm_keyboard`` for two arms
    (14), plain ``keyboard`` for one (7). None disables teleoperation (scripted runs)."""

    arm_mode: str = "dual"
    """Which arm(s) to drive: ``"left"``, ``"right"`` or ``"dual"``. The benchmark is bimanual."""

    teleop_pos_sensitivity: float = 0.03
    """Metres of commanded end-effector motion per held key, per control step.

    Below the device default of 0.05. The arms are position-servoed with very high stiffness and
    zero damping, and RMPFlow runs with ``ignore_robot_state_updates``, so the arm does not yield
    when it touches something -- it drives through at whatever rate it was commanded. Descending
    onto a bowl at the default rate knocks it 18 mm sideways at 0.20 m/s and the grasp misses.

    This is per *step*, so the resulting speed follows the control rate: 0.45 m/s at Arena's
    15 Hz default."""

    teleop_rot_sensitivity: float = 0.1
    """Radians of commanded end-effector rotation per held key, per step. Five times the
    translation sensitivity: rotation aims the gripper rather than driving it into things, and
    at 0.02 reorienting the hand was the slow part of every grasp."""

    head_view: bool = True
    """Put the viewport on the robot's head, so teleop is driven from the robot's own view."""

    room: bool = True
    """Stage the task in RoboDojo's room instead of on a bare ground plane."""

    surface_guard: bool = True
    """Clamp each arm's commanded descent so the gripper cannot be driven into the table.

    Pressing the stiff arm into the table and then closing is one of the four measured causes of
    objects being thrown out of the gripper (finger speeds 1.0 m/s against 0.57 free); the clamp
    removes it without touching any actuator. Teleop path only -- joint-space (recording) terms
    are untouched. See ``isaaclab_arena/utils/surface_guard.py``."""

    gripper_ramp_seconds: float = DEFAULT_GRIPPER_RAMP_SECONDS
    """Ramp time of the teleop grippers' open/close target, then held constant.

    The stock binary action steps the target, which drives the fingers into the object at
    ~0.9 m/s and spat an SDF-collided sleeve out in 5/5 controlled grasps; ramped over this
    time, 5/5 held at 0.35 m/s. 0 disables the ramp. Joint-space (recording) terms are untouched:
    the executor ramps those itself."""

    arm_effort_limit: float | None = DEFAULT_ARM_EFFORT_LIMIT
    """Torque ceiling for both arms, in N m. None keeps the shipped 1000-2000.

    The Agibot arms have damping 0 and stiffness 2e4-1e7, so they do not yield when the gripper
    reaches the table -- they hold position and saturate torque instead. Measured pressing into
    the tabletop, by effort limit, with stiffness and damping left stock:

        stock (1000-2000)   2000 N m saturated   settles 0.2 mm   rebound 0.107 m/s
        500                  500 N m             0.2 mm           0.057 m/s
        300  (this default)  300 N m             0.2 mm           0.093 m/s
        200                  200 N m             0.2 mm           0.153 m/s
        100                  100 N m             ARM COLLAPSES -- droops 332 mm and stays there

    200-500 all keep tracking precision, finger penetration and release behaviour identical to
    stock while cutting the saturated torque several-fold; 300 is the middle of that band.

    Do **not** reach for the ARX X5 triple (stiffness 4400 / damping 40 / effort 100) that
    RoboDojo runs. It hits the same 100 N m ceiling but, measured: the idle arm settles to 2.4 mm
    instead of 0.2 mm, the fingers rebound at 0.647 m/s instead of 0.107, and in teleoperation
    the softer arm couples with the gripper into an oscillation that drops a bowl mid-lift."""

    arm_stiffness: float | None = None
    arm_damping: float | None = None
    """Left at the shipped values; see ``arm_effort_limit`` for why the X5 gains are not used.
    Exposed for measurement only -- never set them in a task's defaults."""


def arm_mode_from_cfg(cfg: AgibotTabletopEnvironmentCfg) -> ArmMode:
    """Map the config's ``arm_mode`` string onto the embodiment's ``ArmMode``."""
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode

    modes = {"left": ArmMode.LEFT, "right": ArmMode.RIGHT, "dual": ArmMode.DUAL_ARM}
    assert cfg.arm_mode in modes, f"arm_mode must be one of {sorted(modes)}, got {cfg.arm_mode!r}"
    return modes[cfg.arm_mode]


def build_tabletop_stage(
    factory: ArenaEnvironmentFactory, cfg: AgibotTabletopEnvironmentCfg, table_top_z: float = TABLE_TOP_Z
) -> tuple[ObjectBase, ObjectBase, ObjectBase]:
    """Build the table, the surroundings and the light every Agibot table-top task stands in.

    Args:
        factory: The environment factory, for its registries.
        cfg: The environment configuration (``background``, ``room``).
        table_top_z: World height of the work surface. Defaults to the measured standard.

    Returns:
        ``(table, surroundings, light)``, ready to be put into the ``Scene``.
    """
    import isaaclab.sim as sim_utils

    from isaaclab_arena.utils.pose import Pose

    table_asset = factory.asset_registry.get_asset_by_name(cfg.background)
    table = table_asset()
    table.set_initial_pose(
        Pose(
            position_xyz=(TABLE_POSITION_X, 0.0, table_top_z - table_asset.HALF_THICKNESS_M),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )

    # A fresh DomeLightCfg per instance: the asset's default is a class attribute, so reusing it
    # would leak the HDR texture into every other environment built in the same process.
    light = factory.asset_registry.get_asset_by_name("light")(
        spawner_cfg=sim_utils.DomeLightCfg(intensity=DOME_LIGHT_INTENSITY),
        hdr=factory.hdr_registry.get_hdr_by_name(DOME_LIGHT_HDR)(),
    )

    if cfg.room:
        # The room brings its own floor, so it replaces the default grid ground plane.
        room_asset = factory.asset_registry.get_asset_by_name("robodojo_simple_room")
        surroundings = room_asset()
        surroundings.object_cfg.spawn.scale = room_asset.SCALE
    else:
        surroundings = factory.asset_registry.get_asset_by_name("ground_plane")()

    return table, surroundings, light


def build_agibot(factory: ArenaEnvironmentFactory, cfg: AgibotTabletopEnvironmentCfg) -> EmbodimentBase:
    """Build the Agibot embodiment at the standard base position, in the configured arm mode."""
    from isaaclab_arena.utils.pose import Pose

    embodiment = factory.asset_registry.get_asset_by_name(cfg.embodiment)(
        enable_cameras=cfg.enable_cameras, arm_mode=arm_mode_from_cfg(cfg)
    )
    embodiment.set_initial_pose(Pose(position_xyz=ROBOT_POSITION_XYZ, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    return embodiment


def build_teleop_device(factory: ArenaEnvironmentFactory, cfg: AgibotTabletopEnvironmentCfg):
    """Build the configured teleop device with the standard sensitivities, or None."""
    if cfg.teleop_device is None:
        return None
    return factory.device_registry.get_device_by_name(cfg.teleop_device)(
        pos_sensitivity=cfg.teleop_pos_sensitivity,
        rot_sensitivity=cfg.teleop_rot_sensitivity,
    )


def apply_arm_gains(env_cfg, cfg: AgibotTabletopEnvironmentCfg) -> None:
    """Overwrite the arm actuators' gains on the compiled config, before the articulation exists.

    This has to happen on ``env_cfg.scene.robot.actuators`` rather than through
    ``write_joint_*_to_sim`` at run time: those setters reach PhysX but leave Isaac Lab's
    ``ImplicitActuator`` holding its original gains, so the two disagree and ``applied_torque``
    keeps reporting the old numbers. Patching the config keeps both sides consistent.

    Args:
        env_cfg: The compiled environment configuration, patched in place.
        cfg: The environment configuration carrying the overrides.
    """
    # ``joint_effort_limit`` is the solver-level ceiling (Isaac Lab 3.0 GA name; the pre-GA
    # ``effort_limit_sim`` is a deprecated alias). Implicit actuators derive their model-facing
    # ``actuator_effort_limit`` from it when that is left None, so one field is enough.
    overrides = {
        "stiffness": cfg.arm_stiffness,
        "damping": cfg.arm_damping,
        "joint_effort_limit": cfg.arm_effort_limit,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    if not overrides:
        return

    for name, actuator in env_cfg.scene.robot.actuators.items():
        if not name.endswith("_arm"):
            continue
        for key, value in overrides.items():
            setattr(actuator, key, value)
        print(f"[arm gains] {name}: " + ", ".join(f"{k}={v:g}" for k, v in overrides.items()))


def install_agibot_control_stack(env_cfg, cfg: AgibotTabletopEnvironmentCfg, surface_z: float = TABLE_TOP_Z) -> None:
    """Install the standard Agibot controller stack on a compiled environment config.

    Call it first in every Agibot task's ``env_cfg_callback``. In order: the idle-arm target hold
    (an RMPFlow arm commanded zero otherwise walks 88 mm off its reset pose), the surface guard,
    the ramped gripper, and the arm gain overrides. The order matters -- the guard subclasses the
    target-holding term -- and every installer leaves joint-space terms alone, so an environment
    rebuilt with ``AgibotDualArmJointActionsCfg`` for scripted recording is unaffected.

    Args:
        env_cfg: The compiled environment configuration, patched in place.
        cfg: The environment configuration carrying the knobs.
        surface_z: World height of the work surface the guard protects.
    """
    from isaaclab_arena.utils.arm_target_hold import install_arm_target_hold
    from isaaclab_arena.utils.ramped_gripper import install_ramped_gripper
    from isaaclab_arena.utils.surface_guard import install_surface_guard

    install_arm_target_hold(env_cfg)
    if cfg.surface_guard:
        install_surface_guard(env_cfg, surface_z)
    if cfg.gripper_ramp_seconds > 0:
        install_ramped_gripper(env_cfg, cfg.gripper_ramp_seconds)
    apply_arm_gains(env_cfg, cfg)
