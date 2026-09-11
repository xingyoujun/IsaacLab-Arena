# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""UR7e arm with a Robotiq 2F-85 gripper, replicating the real workcell captured in ``rr_ur/scene.py``.

The robot USD is the Robot Assembler output of a URDF-imported UR7e and the Isaac Robotiq 2F-85
asset. Until that USD and its sub-layers are available on this host, ``select_ur_robot_spec``
falls back to the Isaac Sim UR5e asset with the same gripper variant so the rest of the pipeline
can be exercised; the fallback is announced with a warning. Set ``ARENA_UR7E_USD`` to point at a
different copy of the UR7e USD.
"""

from __future__ import annotations

import math
import os
from abc import ABC
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.contrib.stack.mdp import franka_stack_events

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.smooth_joint_actions import SmoothJointPositionActionCfg
from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.ur7e.observations import (
    GRIPPER_CLOSED_JOINT_POS,
    GRIPPER_DRIVE_JOINT,
    UR_ARM_JOINT_NAMES,
    arm_joint_pos,
    ee_pos,
    ee_quat,
    gripper_pos,
)
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

# ---------------------------------------------------------------------------
# Robot USD selection
# ---------------------------------------------------------------------------

UR7E_ROBOTIQ_USD_PATH = "/home/ubuntu/playground/rr_ur/ur7e_usd/ur7e_gripper/Collected_ur7e_gripper/ur7e_gripper.usd"
"""Robot Assembler output: URDF-imported UR7e + Isaac Robotiq 2F-85. Sub-layers live next to it."""

_UR7E_REQUIRED_SUBLAYER = "ur7e/ur/configuration/ur_physics.usd"
"""One of the UR7e sub-layers the top-level USD payloads; used to detect an incomplete copy."""


@dataclass(frozen=True)
class UrRobotSpec:
    """Where a UR + Robotiq 2F-85 robot USD lives and how its prims are laid out."""

    usd_path: str
    """USD file to spawn."""
    variants: dict[str, str] | None
    """Variant selections to apply on the root prim, or None."""
    gripper_prim: str
    """Path of the Robotiq_2F_85 prim relative to the spawned robot root."""
    label: str
    """Human-readable description used in log messages."""
    gripper_base_body_name: str = "base_link"
    """Articulation body name of the Robotiq base link, used as the IK control body.

    Isaac Lab de-duplicates body names, so on assets whose arm also has a ``base_link`` body the
    gripper's becomes ``base_link_0``.
    """
    articulation_root_prim_path: str = "/root_joint"
    """Prim carrying the arm's ArticulationRootAPI, relative to the robot root.

    Both the URDF-imported UR7e and the Isaac UR assets put it on ``root_joint``. It must be named
    explicitly because the attached Robotiq asset carries a second (disabled) articulation root.
    """


UR7E_SPEC = UrRobotSpec(
    usd_path=UR7E_ROBOTIQ_USD_PATH,
    variants=None,
    gripper_prim="ee_link/Robotiq_2F_85",
    label="UR7e + Robotiq 2F-85 (rr_ur assembler USD)",
)

UR5E_STAND_IN_SPEC = UrRobotSpec(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
    variants={"Gripper": "Robotiq_2f_85"},
    gripper_prim="Gripper/Robotiq_2F_85",
    label="UR5e + Robotiq 2F-85 (Isaac asset, stand-in for the missing UR7e sub-layers)",
    gripper_base_body_name="base_link_0",
)


def select_ur_robot_spec() -> UrRobotSpec:
    """Return the UR7e spec when its USD and sub-layers are present, otherwise the UR5e stand-in."""
    override = os.environ.get("ARENA_UR7E_USD")
    if override:
        assert os.path.isfile(override), f"ARENA_UR7E_USD points at a missing file: {override}"
        return UrRobotSpec(usd_path=override, variants=None, gripper_prim=UR7E_SPEC.gripper_prim, label=override)
    sublayer = os.path.join(os.path.dirname(UR7E_ROBOTIQ_USD_PATH), _UR7E_REQUIRED_SUBLAYER)
    if os.path.isfile(UR7E_ROBOTIQ_USD_PATH) and os.path.isfile(sublayer):
        return UR7E_SPEC
    print(
        f"[ur7e] WARNING: {UR7E_ROBOTIQ_USD_PATH} or its sub-layer {_UR7E_REQUIRED_SUBLAYER} is missing; "
        f"spawning the stand-in instead: {UR5E_STAND_IN_SPEC.label}"
    )
    return UR5E_STAND_IN_SPEC


# ---------------------------------------------------------------------------
# Joint layout and calibrated poses (from rr_ur/scene.py)
# ---------------------------------------------------------------------------

_UR7E_JOINT_NAMES = UR_ARM_JOINT_NAMES + (
    GRIPPER_DRIVE_JOINT,
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
)
"""Joint order accepted by ``set_initial_joint_pose``: six arm joints, then the Robotiq joints."""

UR7E_POSE_A_JOINT_POS: dict[str, float] = {
    "shoulder_pan_joint": 1.4309711456298828,
    "shoulder_lift_joint": -1.6231142483153285,
    "elbow_joint": 2.4302189985858362,
    "wrist_1_joint": -2.3047520122923792,
    "wrist_2_joint": -1.5249694029437464,
    "wrist_3_joint": -0.06301623979677373,
    "finger_joint": 0.009650588235294117,
}
"""Calibration pose ``pose_a`` measured on the real UR7e (SN 254622076156) on 2026-08-22."""

UR7E_READY_JOINT_POS: dict[str, float] = {
    "shoulder_pan_joint": 1.4309711456298828,
    "shoulder_lift_joint": -1.95,
    "elbow_joint": 2.05,
    "wrist_1_joint": -1.67,
    "wrist_2_joint": -1.5249694029437464,
    "wrist_3_joint": -0.06301623979677373,
    "finger_joint": 0.0,
}
"""``pose_a`` with the arm folded up: same base yaw, gripper still pointing down, tool raised well clear
of the table so objects can be placed under the calibrated camera without touching the robot."""

_ROBOTIQ_PASSIVE_JOINT_POS: dict[str, float] = {
    "right_outer_knuckle_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
}

TCP_OFFSET_FROM_GRIPPER_BASE_M = 0.1628
"""Tool centre point along the gripper base_link +Z, Robotiq 2F-85 closed-finger length from the spec sheet."""


# ---------------------------------------------------------------------------
# Embodiments
# ---------------------------------------------------------------------------


class Ur7eRobotiqEmbodimentBase(EmbodimentBase, ABC):
    """Abstract base for the UR7e + Robotiq 2F-85 workcell embodiment.

    Subclasses set ``self.action_config``. ``initial_pose`` / ``set_initial_pose`` place the robot
    base (the UR ``base_link`` mounting face) in the world frame.
    """

    name = "ur7e_robotiq"
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        collision_mode: CollisionMode | str | None = None,
        robot_spec: UrRobotSpec | None = None,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
            collision_mode=collision_mode,
        )
        self.robot_spec = robot_spec if robot_spec is not None else select_ur_robot_spec()
        self.scene_config = make_ur7e_scene_cfg(self.robot_spec)
        self.action_config = None
        self.camera_config = Ur7eCameraCfg()
        self.observation_config = Ur7eObservationsCfg()
        self.event_config = Ur7eEventCfg()
        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)
        self.reward_config = None
        self.mimic_env = None
        self.add_camera_variations(self.camera_config)

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        """Set the spawn and reset joint positions in ``_UR7E_JOINT_NAMES`` order."""
        expected_joint_count = len(_UR7E_JOINT_NAMES)
        assert (
            len(initial_joint_pose) == expected_joint_count
        ), f"expected {expected_joint_count} joint positions, got {len(initial_joint_pose)}"
        robot = self.scene_config.robot
        robot.init_state = robot.init_state.replace(joint_pos=dict(zip(_UR7E_JOINT_NAMES, initial_joint_pose)))

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_command_body_name(self) -> str:
        return self.robot_spec.gripper_base_body_name


@register_asset
class Ur7eRobotiqJointPositionEmbodiment(Ur7eRobotiqEmbodimentBase):
    """UR7e + Robotiq 2F-85 driven by absolute arm joint positions and a binary gripper command."""

    name = "ur7e_robotiq_joint_pos"
    tags = ["embodiment", "default"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_config = Ur7eJointPositionActionsCfg()


@register_asset
class Ur7eRobotiqDifferentialIKEmbodiment(Ur7eRobotiqEmbodimentBase):
    """UR7e + Robotiq 2F-85 driven by relative tool-pose commands through differential IK."""

    name = "ur7e_robotiq_ik"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_config = Ur7eDifferentialIKActionsCfg()
        self.action_config.arm_action.body_name = self.robot_spec.gripper_base_body_name


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def make_ur7e_robot_cfg(spec: UrRobotSpec) -> ArticulationCfg:
    """Build the articulation config for a UR + Robotiq 2F-85 robot spawned from ``spec``.

    Arm gains follow Isaac Lab's ``UR10e_CFG``; gripper gains follow ``UR10e_ROBOTIQ_2F_85_CFG``.
    """
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        articulation_root_prim_path=spec.articulation_root_prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,
            variants=spec.variants,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={**UR7E_POSE_A_JOINT_POS, **_ROBOTIQ_PASSIVE_JOINT_POS},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*"],
                stiffness=1320.0,
                damping=72.6636085,
                friction=0.0,
                armature=0.0,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                stiffness=600.0,
                damping=34.64101615,
                friction=0.0,
                armature=0.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*"],
                stiffness=216.0,
                damping=29.39387691,
                friction=0.0,
                armature=0.0,
            ),
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=[GRIPPER_DRIVE_JOINT],
                joint_effort_limit=10.0,
                joint_velocity_limit=1.0,
                stiffness=11.25,
                damping=0.1,
                friction=0.0,
                armature=0.0,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                joint_effort_limit=1.0,
                joint_velocity_limit=1.0,
                stiffness=0.2,
                damping=0.001,
                friction=0.0,
                armature=0.0,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
                joint_effort_limit=1.0,
                joint_velocity_limit=1.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
            ),
        },
    )


def make_ur7e_scene_cfg(spec: UrRobotSpec):
    """Return a configclass instance holding the robot and its tool-centre-point frame sensor."""
    gripper_root = "{ENV_REGEX_NS}/Robot/" + spec.gripper_prim

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    @configclass
    class Ur7eSceneCfg:
        """Additions to the scene configuration coming from the UR7e embodiment."""

        robot: ArticulationCfg = make_ur7e_robot_cfg(spec)

        ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path=gripper_root + "/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=gripper_root + "/base_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, TCP_OFFSET_FROM_GRIPPER_BASE_M)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=gripper_root + "/right_inner_finger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=gripper_root + "/left_inner_finger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
            ],
        )

    return Ur7eSceneCfg()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Binary gripper action where inputs above 0.5 close the gripper."""

    class_type = BinaryJointPositionZeroToOneAction


def _gripper_action_cfg() -> ActionTermCfg:
    return BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_DRIVE_JOINT],
        open_command_expr={GRIPPER_DRIVE_JOINT: 0.0},
        close_command_expr={GRIPPER_DRIVE_JOINT: GRIPPER_CLOSED_JOINT_POS},
    )


@configclass
class Ur7eJointPositionActionsCfg:
    """Absolute arm joint positions (six values, ``UR_ARM_JOINT_NAMES`` order) plus a binary gripper."""

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(UR_ARM_JOINT_NAMES),
        preserve_order=True,
        use_default_offset=False,
    )
    gripper_action: ActionTermCfg = _gripper_action_cfg()


@configclass
class Ur7eJointRecordingActionsCfg:
    """Joint-space actions for scripted (cuMotion) demonstration recording.

    The arm term is a first-order hold so the stiff arm is not jolted at every 15 Hz control step;
    the gripper term is a plain zero-order hold on ``finger_joint`` so the executor's ramp reaches
    the drive unchanged (a binary term would collapse it to open/closed). Action vector:
    ``[6 arm joints in UR_ARM_JOINT_NAMES order, finger_joint]``.
    """

    arm_action: ActionTermCfg = SmoothJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(UR_ARM_JOINT_NAMES),
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )
    gripper_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_DRIVE_JOINT],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class Ur7eDifferentialIKActionsCfg:
    """Relative tool-pose deltas resolved by damped-least-squares differential IK, plus a binary gripper."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(UR_ARM_JOINT_NAMES),
        body_name="base_link",  # overwritten per robot spec by the embodiment
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, TCP_OFFSET_FROM_GRIPPER_BASE_M)),
    )
    gripper_action: ActionTermCfg = _gripper_action_cfg()


# ---------------------------------------------------------------------------
# Observations and events
# ---------------------------------------------------------------------------


@configclass
class Ur7eObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive state of the arm and gripper."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        robot_joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(func=gripper_pos)
        eef_pos = ObsTerm(func=ee_pos)
        eef_quat = ObsTerm(func=ee_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Ur7eEventCfg:
    """Reset events. The joint jitter defaults to zero so resets reproduce the calibrated pose exactly."""

    reset_robot_joints = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ---------------------------------------------------------------------------
# Camera: RealSense D435 replica, fixed in the world (rr_ur/scene.py)
# ---------------------------------------------------------------------------

D435_CAMERA_POS = (-0.5537, -0.1099, 1.3671)
"""Camera position in the world frame; base-frame calibration plus the base at (0, -0.425, 0.75)."""
D435_CAMERA_LOOK_AT = (-0.0517, -0.1160, 0.7443)
"""A point 0.8 m along the optical axis."""
D435_CAMERA_UP = (0.7786, 0.0070, 0.6275)
"""World direction that maps to image-up."""


def look_at_quat_xyzw(
    eye: tuple[float, float, float], target: tuple[float, float, float], up: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """Return the OpenGL-convention camera orientation (looks along -Z, +Y up) as an xyzw quaternion.

    Args:
        eye: Camera position.
        target: Point the optical axis passes through.
        up: World direction that should appear as image-up; re-orthogonalised against the view axis.
    """

    def _sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def _dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def _norm(a):
        n = math.sqrt(_dot(a, a))
        assert n > 1e-9, "degenerate vector"
        return tuple(x / n for x in a)

    def _cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    forward = _norm(_sub(target, eye))
    z_axis = tuple(-x for x in forward)
    up_proj = _sub(up, tuple(_dot(up, z_axis) * z for z in z_axis))
    y_axis = _norm(up_proj)
    x_axis = _cross(y_axis, z_axis)
    # Column-major rotation matrix [x_axis y_axis z_axis] -> quaternion (Shepperd's method).
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return (x, y, z, w)


SCENE_CAMERA_POS = (1.7, -1.5, 1.7)
"""Third-person overview camera position, front-left of the robot looking down at the table."""
SCENE_CAMERA_LOOK_AT = (0.0, 0.0, 0.8)


@configclass
class Ur7eCameraCfg(ArenaCameraCfg):
    """The calibrated RealSense D435 plus a third-person overview camera for videos and debugging.

    ``realsense_d435`` is the colour stream (640x480) of the real camera on the workcell.
    ``scene_cam`` is not part of the real setup; it stands in for the Kit viewport, which cannot be
    recorded headless, so `--record_camera_video` yields an overview clip as well.

    Intrinsics come from the measured colour intrinsics fx=606.5 fy=605.6 cx=328.3 cy=238.3
    (aperture 20.955 mm); the horizontal aperture offset is negated because the RTX renderer applies
    it mirrored, as measured in ``rr_ur/scene.py``.
    """

    realsense_d435: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RealSenseD435",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=19.858321,
            horizontal_aperture=20.955,
            vertical_aperture=15.740976,
            horizontal_aperture_offset=-0.272296,
            vertical_aperture_offset=-0.057256,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=D435_CAMERA_POS,
            rot=look_at_quat_xyzw(D435_CAMERA_POS, D435_CAMERA_LOOK_AT, D435_CAMERA_UP),
            convention="opengl",
        ),
    )

    scene_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneCam",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(
            pos=SCENE_CAMERA_POS,
            rot=look_at_quat_xyzw(SCENE_CAMERA_POS, SCENE_CAMERA_LOOK_AT, (0.0, 0.0, 1.0)),
            convention="opengl",
        ),
    )
