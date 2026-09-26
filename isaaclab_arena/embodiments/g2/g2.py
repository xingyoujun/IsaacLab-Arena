# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""AgiBot Genie G2 (omnipicker gripper) embodiment for upper-body tabletop manipulation.

The robot asset is the ``robot/G2_omnipicker`` folder of the ``agibot-world/GenieSimAssets`` HuggingFace
dataset (the robot used by every ``*_g2_op`` Genie Sim benchmark task). The base is fixed to the world by the
``robot_fix.usda`` layer, so only the upper body moves. Runtime resolves the repackaged robot
from the pinned USDCraft-Scene manifest ID ``g2``.
"""

import math
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
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
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.contrib.pick_place.mdp import get_robot_joint_state
from isaaclab_tasks.contrib.stack.mdp import ee_frame_pose_in_base_frame, franka_stack_events

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.usdcraft_scene import resolve_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.franka import FrankaMimicEnv
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

# Genie Sim's default G2 manipulation posture (``G2_DEFAULT_STATES`` in geniesim_benchmark): torso bent over the
# table, both arms in front of the body, grippers open.
G2_DEFAULT_JOINT_POS: dict[str, float] = {
    # waist
    "idx01_body_joint1": -0.83423,
    "idx02_body_joint2": 1.2172,
    "idx03_body_joint3": 0.10025,
    "idx04_body_joint4": 0.0,
    "idx05_body_joint5": 0.0,
    # head
    "idx11_head_joint1": 0.0,
    "idx12_head_joint2": 0.0,
    "idx13_head_joint3": 0.11464,
    # left arm
    "idx21_arm_l_joint1": 0.739033,
    "idx22_arm_l_joint2": -0.717023,
    "idx23_arm_l_joint3": -1.524419,
    "idx24_arm_l_joint4": -1.537612,
    "idx25_arm_l_joint5": 0.27811,
    "idx26_arm_l_joint6": -0.925845,
    "idx27_arm_l_joint7": -0.839257,
    # right arm
    "idx61_arm_r_joint1": -0.739033,
    "idx62_arm_r_joint2": -0.717023,
    "idx63_arm_r_joint3": 1.524419,
    "idx64_arm_r_joint4": -1.537612,
    "idx65_arm_r_joint5": -0.27811,
    "idx66_arm_r_joint6": -0.925845,
    "idx67_arm_r_joint7": 0.839257,
    # grippers: the outer and inner finger joints mirror each other; the remaining four-bar joints are passive
    # and follow through the ``/genie/loop_joints`` spherical constraints
    "idx41_gripper_l_outer_joint1": 0.785,
    "idx81_gripper_r_outer_joint1": 0.785,
    "idx31_gripper_l_inner_joint1": -0.785,
    "idx71_gripper_r_inner_joint1": -0.785,
    "idx.._gripper_._(inner|outer)_joint[034]": 0.0,
    # chassis wheels, unused on the fixed base
    "idx1.._chassis_.*": 0.0,
}

G2_GRIPPER_OPEN: float = 0.785
"""Omnipicker finger joint angle [rad] for an open gripper (45 deg, the USD joint limit); the inner joint is negated."""

G2_GRIPPER_CLOSED: float = 0.0
"""Omnipicker finger joint angle [rad] for a closed gripper."""

# PD gains follow the vendor ``robot_fix.usda`` layer; effort limits are the URDF-derived drive max forces.
G2_OMNIPICKER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="",  # Resolved from the HF bundle when G2Embodiment is constructed.
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # Genie Sim grasps rely on high-friction contacts combined with "max"; the vendor USD binds no physics
        # material to the finger pads, so apply one to the whole robot here.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="max"
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=G2_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["idx2._arm_l_joint.", "idx6._arm_r_joint."],
            effort_limit_sim=60.0,
            velocity_limit_sim=math.pi,
            stiffness=10000.0,
            damping=1000.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["idx0._body_joint."],
            effort_limit_sim=100.0,
            stiffness=100000.0,
            damping=10000.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["idx1._head_joint."],
            effort_limit_sim=50.0,
            stiffness=500.0,
            damping=50.0,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["idx[48]1_gripper_._outer_joint1", "idx[37]1_gripper_._inner_joint1"],
            effort_limit_sim=15.0,
            stiffness=100.0,
            damping=20.0,
        ),
        # Passive four-bar links closed by the ``/genie/loop_joints`` spherical constraints.
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["idx.._gripper_._(inner|outer)_joint[034]"],
            effort_limit_sim=10.0,
            stiffness=0.0,
            damping=0.02,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["idx1.._chassis_.*"],
            stiffness=100000.0,
            damping=10000.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


@register_asset
class G2Embodiment(EmbodimentBase):
    """AgiBot Genie G2 with omnipicker grippers and relative differential IK for one or both arms."""

    name = "g2"
    default_arm_mode = ArmMode.RIGHT

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
        )
        if self.arm_mode == ArmMode.RIGHT:
            self.scene_config = G2RightArmSceneCfg()
            self.action_config = G2RightArmActionsCfg()
        elif self.arm_mode == ArmMode.LEFT:
            self.scene_config = G2LeftArmSceneCfg()
            self.action_config = G2LeftArmActionsCfg()
        elif self.arm_mode == ArmMode.DUAL_ARM:
            self.scene_config = G2DualArmSceneCfg()
            self.action_config = G2DualArmActionsCfg()
        else:
            raise NotImplementedError(f"Unsupported G2 arm mode: {self.arm_mode}.")
        self.scene_config.robot.spawn.usd_path = str(resolve_asset("g2"))
        self.observation_config = (
            G2DualArmObservationsCfg() if self.arm_mode == ArmMode.DUAL_ARM else G2ObservationsCfg()
        )
        self.event_config = G2EventCfg()
        self.camera_config = G2CameraCfg()
        self.mimic_env = G2MimicEnv
        self.add_camera_variations(self.camera_config)

    def get_command_body_name(self) -> str:
        assert self.arm_mode != ArmMode.DUAL_ARM, "Select a specific arm for a single-body command"
        return self.action_config.arm_action.body_name

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "left_end_effector" if arm_mode == ArmMode.LEFT else "right_end_effector"

    def get_ee_frame_transformer_names(self) -> list[str]:
        return ["ee_frame", "left_ee_frame"] if self.arm_mode == ArmMode.DUAL_ARM else ["ee_frame"]

    def get_teleop_target_frame_prim_path(self) -> str | None:
        return "{ENV_REGEX_NS}/Robot/base_link"


@configclass
class G2SceneCfg:
    """Scene configuration for the G2."""

    robot = G2_OMNIPICKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = MISSING

    def __post_init__(self):
        # Add a marker to the end-effector frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class G2RightArmSceneCfg(G2SceneCfg):
    """Scene configuration for the G2 right arm."""

    # ``gripper_r_center_link`` is the omnipicker TCP frame, so no extra offset is needed.
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_r_center_link",
                name="right_end_effector",
            ),
        ],
    )


@configclass
class G2LeftArmSceneCfg(G2SceneCfg):
    """Scene configuration for the G2 left arm."""

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_l_center_link",
                name="left_end_effector",
            ),
        ],
    )


@configclass
class G2DualArmSceneCfg(G2RightArmSceneCfg):
    """Right and left TCP sensors for independent arm observations."""

    left_ee_frame = G2LeftArmSceneCfg().ee_frame.copy()


@configclass
class G2RightArmActionsCfg:
    """Action configuration for the G2 right arm: relative differential IK plus a binary gripper."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["idx6._arm_r_joint."],
        body_name="gripper_r_center_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["idx81_gripper_r_outer_joint1", "idx71_gripper_r_inner_joint1"],
        open_command_expr={
            "idx81_gripper_r_outer_joint1": G2_GRIPPER_OPEN,
            "idx71_gripper_r_inner_joint1": -G2_GRIPPER_OPEN,
        },
        close_command_expr={
            "idx81_gripper_r_outer_joint1": G2_GRIPPER_CLOSED,
            "idx71_gripper_r_inner_joint1": G2_GRIPPER_CLOSED,
        },
    )


@configclass
class G2LeftArmActionsCfg:
    """Action configuration for the G2 left arm: relative differential IK plus a binary gripper."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["idx2._arm_l_joint."],
        body_name="gripper_l_center_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["idx41_gripper_l_outer_joint1", "idx31_gripper_l_inner_joint1"],
        open_command_expr={
            "idx41_gripper_l_outer_joint1": G2_GRIPPER_OPEN,
            "idx31_gripper_l_inner_joint1": -G2_GRIPPER_OPEN,
        },
        close_command_expr={
            "idx41_gripper_l_outer_joint1": G2_GRIPPER_CLOSED,
            "idx31_gripper_l_inner_joint1": G2_GRIPPER_CLOSED,
        },
    )


@configclass
class G2DualArmActionsCfg:
    """Actions ordered as right pose/gripper, then left pose/gripper (14 values)."""

    right_arm_action = G2RightArmActionsCfg().arm_action.copy()
    right_gripper_action = G2RightArmActionsCfg().gripper_action.copy()
    left_arm_action = G2LeftArmActionsCfg().arm_action.copy()
    left_gripper_action = G2LeftArmActionsCfg().gripper_action.copy()


@configclass
class G2JointPositionActionsCfg:
    """Planner/replay actions: right joints (7), gripper (1), left joints (7), gripper (1)."""

    right_arm = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)],
        use_default_offset=False,
        preserve_order=True,
    )
    right_gripper = G2RightArmActionsCfg().gripper_action.copy()
    left_arm = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)],
        use_default_offset=False,
        preserve_order=True,
    )
    left_gripper = G2LeftArmActionsCfg().gripper_action.copy()


@configclass
class G2CameraCfg(ArenaCameraCfg):
    """The G2's on-board cameras, read from the camera prims authored in the robot USD.

    Prim paths, resolutions and the aspect ratios follow the Genie Sim ``G2_omnipicker.json`` camera list
    (head 640x400, wrist cameras 1280x1056), the wrist cameras halved to keep tiled rendering light.
    """

    head_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_link3/head_front_Camera",
        height=400,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )
    left_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_l_base_link/Left_Camera",
        height=528,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )
    right_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_r_base_link/Right_Camera",
        height=528,
        width=640,
        data_types=["rgb"],
        spawn=None,
    )


@configclass
class G2CollectionCameraCfg(G2CameraCfg):
    """Wrist cameras plus a fixed external view for inspecting the task."""

    head_camera: CameraCfg | None = None

    overview_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CollectionCamera",
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-1.0, -1.2, 0.9),
            # Isaac Lab camera offsets use xyzw, unlike cuRobo's wxyz goal poses.
            rot=(0.47902577, -0.16666203, -0.28319755, 0.81397618),
            convention="opengl",
        ),
    )


@configclass
class G2ObservationsCfg:
    """Observation configuration for the G2 robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # Since the robot may not be located at the environment origin, the EEF pose is expressed in the base frame.
        eef_pos = ObsTerm(func=ee_frame_pose_in_base_frame, params={"return_key": "pos"})
        eef_quat = ObsTerm(func=ee_frame_pose_in_base_frame, params={"return_key": "quat"})
        left_gripper_pos = ObsTerm(func=get_robot_joint_state, params={"joint_names": ["idx41_gripper_l_outer_joint1"]})
        right_gripper_pos = ObsTerm(
            func=get_robot_joint_state, params={"joint_names": ["idx81_gripper_r_outer_joint1"]}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class G2DualArmObservationsCfg(G2ObservationsCfg):
    """Expose both TCP poses; existing eef terms continue to describe the right arm."""

    @configclass
    class PolicyCfg(G2ObservationsCfg.PolicyCfg):
        left_eef_pos = ObsTerm(
            func=ee_frame_pose_in_base_frame,
            params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame"), "return_key": "pos"},
        )
        left_eef_quat = ObsTerm(
            func=ee_frame_pose_in_base_frame,
            params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame"), "return_key": "quat"},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class G2EventCfg:
    """Event configuration for the G2 robot."""

    # Drive every joint back to ``G2_DEFAULT_JOINT_POS`` on reset (zero mean and std: no randomization), so the
    # relative IK controller always starts from the same manipulation posture.
    reset_g2_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


class G2MimicEnv(FrankaMimicEnv):
    """Mimic environment for the G2; object poses are expressed in the robot base frame like the observations."""

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Gets the pose of each rigid and articulated object in the robot base frame.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to 4x4 object pose matrices in the robot base frame.
        """
        if env_ids is None:
            env_ids = slice(None)

        scene_state = self.scene.get_state(is_relative=True)
        rigid_object_states = scene_state["rigid_object"]
        articulation_states = scene_state["articulation"]

        robot_root_pose = articulation_states["robot"]["root_pose"]
        root_pos = robot_root_pose[env_ids, :3]
        root_quat = robot_root_pose[env_ids, 3:7]

        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat, obj_state["root_pose"][env_ids, :3], obj_state["root_pose"][env_ids, 3:7]
            )
            object_pose_matrix[obj_name] = PoseUtils.make_pose(pos_obj_base, PoseUtils.matrix_from_quat(quat_obj_base))

        for art_name, art_state in articulation_states.items():
            if art_name == "robot":
                continue
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat, art_state["root_pose"][env_ids, :3], art_state["root_pose"][env_ids, 3:7]
            )
            object_pose_matrix[art_name] = PoseUtils.make_pose(pos_obj_base, PoseUtils.matrix_from_quat(quat_obj_base))

        return object_pose_matrix
