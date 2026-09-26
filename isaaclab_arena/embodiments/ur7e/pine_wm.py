# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""UR7e with the pine_wm mounting hardware and September 2026 camera calibration."""

import json
import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.usdcraft_scene import bundle_root, resolve_asset
from isaaclab_arena.embodiments.ur7e.appearance import apply_black_gripper
from isaaclab_arena.embodiments.ur7e.pine_wm_mounts import build_camera_mounts
from isaaclab_arena.embodiments.ur7e.ur7e import (
    Ur7eCameraCfg,
    Ur7eJointRecordingActionsCfg,
    Ur7eRobotiqJointPositionEmbodiment,
    UrRobotSpec,
    look_at_quat_xyzw,
)
from isaaclab_arena.utils.cameras import ArenaCameraCfg

CALIBRATION_DIR = Path(__file__).parent / "calibration" / "pine_wm"
TOOL_PATH = "wrist_3_link/flange/tool0"
ROBOT_POSITION = (0.0, -0.425, 0.75)
GRIPPER_SPACER_M = 0.013


def load_calibration(name: str) -> dict:
    """Read the packaged, unchanged source calibration JSON."""
    return json.loads((CALIBRATION_DIR / f"{name}.json").read_text())


def _wrist_camera(name: str) -> CameraCfg:
    calibration = load_calibration("wrist_cams")["cameras"][name]
    # T_T_C maps OpenCV optical camera C into tool0 T. Isaac Lab accepts ROS optical offsets.
    T_T_C = np.asarray(calibration["T_tool0_cam"])
    intr = calibration["intrinsics"]
    width, height = intr["width"], intr["height"]
    aperture = 20.955
    vertical_aperture = aperture * height / width
    # Preserve scene.py's pinhole projection exactly (including its fx-based vertical aperture).
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{TOOL_PATH}/{name}",
        width=width,
        height=height,
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=intr["fx"] * aperture / width,
            horizontal_aperture=aperture,
            vertical_aperture=vertical_aperture,
            horizontal_aperture_offset=-(intr["ppx"] - width / 2) / width * aperture,
            vertical_aperture_offset=(intr["ppy"] - height / 2) / height * vertical_aperture,
            clipping_range=(0.01, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(T_T_C[:3, 3]),
            rot=tuple(Rotation.from_matrix(T_T_C[:3, :3]).as_quat()),
            convention="ros",
        ),
    )


def _d435_camera() -> CameraCfg:
    args = load_calibration("d435")["scene_args"]
    camera = Ur7eCameraCfg().realsense_d435.copy()
    camera.offset = CameraCfg.OffsetCfg(
        pos=tuple(args["cam_pos"]),
        rot=look_at_quat_xyzw(args["cam_pos"], args["cam_look"], args["cam_up"]),
        convention="opengl",
    )
    return camera


@configclass
class PineWmCameraCfg(ArenaCameraCfg):
    """Three calibrated 640x480 RGB sensors and a 1280x720 diagnostic overview."""

    realsense_d435: CameraCfg = _d435_camera()
    wrist_a: CameraCfg = _wrist_camera("wrist_a")
    wrist_b: CameraCfg = _wrist_camera("wrist_b")
    scene_cam: CameraCfg = Ur7eCameraCfg().scene_cam.copy()


def spawn_pine_wm_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the collected robot with the source spacer and collision-enabled D405 housings."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = prim.GetStage()
    calibration = load_calibration("wrist_cams")
    assert calibration["gripper_spacer_m"] == GRIPPER_SPACER_M
    for robot_path in sim_utils.find_matching_prim_paths(prim_path, stage=stage):
        apply_black_gripper(stage, robot_path)
        joint_path = f"{robot_path}/ee_link/Robotiq_2F_85/base_link/AssemblerFixedJoint"
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(joint_path))
        assert joint, f"Missing pine_wm gripper mounting joint: {joint_path}"
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, GRIPPER_SPACER_M))
        tool_path = f"{robot_path}/{TOOL_PATH}"
        assert stage.GetPrimAtPath(tool_path), f"Missing tool0: {tool_path}"
        spacer = UsdGeom.Cylinder.Define(stage, f"{tool_path}/GripperSpacer")
        spacer.CreateAxisAttr("Z")
        spacer.CreateHeightAttr(GRIPPER_SPACER_M)
        spacer.CreateRadiusAttr(0.0375)
        UsdGeom.Xformable(spacer).AddTranslateOp().Set(Gf.Vec3d(0, 0, GRIPPER_SPACER_M / 2))
        spacer.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.03, 0.03)])
        material = UsdShade.Material.Define(stage, f"{robot_path}/D405Alu")
        shader = UsdShade.Shader.Define(stage, f"{robot_path}/D405Alu/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.62, 0.63, 0.64))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        for name, camera in calibration["cameras"].items():
            body = UsdGeom.Xform.Define(stage, f"{tool_path}/{name}_body")
            T_T_C = np.asarray(camera["T_tool0_cam"])
            UsdGeom.Xformable(body).AddTransformOp().Set(Gf.Matrix4d(T_T_C.T.tolist()))
            box = UsdGeom.Cube.Define(stage, f"{body.GetPath()}/Housing")
            box.GetSizeAttr().Set(1.0)
            transform = UsdGeom.Xformable(box)
            transform.AddTranslateOp().Set(Gf.Vec3d(0.009, 0.0, -0.0125))
            transform.AddScaleOp().Set(Gf.Vec3f(0.042, 0.042, 0.023))
            UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material)
        build_camera_mounts(stage, tool_path, calibration)
        # These parts are attached to wrist_3, not independent rigid bodies.
        # Convex hulls conservatively fill the printed support windows for contact safety.
        for part in Usd.PrimRange(stage.GetPrimAtPath(tool_path)):
            if part.IsA(UsdGeom.Cube) or part.IsA(UsdGeom.Cylinder) or part.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(part)
                if part.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(part).CreateApproximationAttr("convexHull")
    return prim


@register_asset
class PineWmUr7eEmbodiment(Ur7eRobotiqJointPositionEmbodiment):
    """Collected UR7e with a -90 degree gripper clocking and three calibrated cameras."""

    name = "pine_wm_ur7e"

    def __init__(self, *args, asset_root: str | None = None, **kwargs):
        legacy_root = asset_root or os.environ.get("ARENA_PINE_WM_ASSET_ROOT")
        root = bundle_root(legacy_root)
        if (root / "manifest.json").is_file():
            usd_path = resolve_asset("pine_wm_ur7e", root)
        elif legacy_root:
            # Explicit old paths remain readable for historical recordings.
            usd_path = root / "assets/ur7e_gripper/ur7e_gripper.usd"
            assert usd_path.is_file(), f"Missing legacy Pine WM robot: {usd_path}"
        else:
            usd_path = resolve_asset("pine_wm_ur7e", root)
        spec = UrRobotSpec(
            usd_path=str(usd_path),
            variants=None,
            gripper_prim="ee_link/Robotiq_2F_85",
            label="pine_wm UR7e + Robotiq (-90 degree mount, 13 mm spacer)",
        )
        super().__init__(*args, robot_spec=spec, camera_config=PineWmCameraCfg(), **kwargs)
        # Preserve the source's independent render products and calibrated aperture offsets.
        self.camera_config.set_use_tiled_camera(False)
        self.scene_config.robot.spawn.func = spawn_pine_wm_robot
        # Seven absolute joint targets retain continuous finger positions for data collection.
        self.action_config = Ur7eJointRecordingActionsCfg()
