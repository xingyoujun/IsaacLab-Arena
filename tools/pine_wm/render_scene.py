# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render and smoke-check the registered pine_wm scene through Arena."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/pine_wm"))
parser.add_argument("--asset-root", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
launcher = AppLauncher(args)


def main():
    """Save all four cameras and validate robot holding, wrist following and reset."""
    import json
    import numpy as np
    import torch

    from PIL import Image, ImageDraw

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.embodiments.ur7e.observations import UR_ARM_JOINT_NAMES
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.pine_wm_environment import PineWmEnvironmentCfg

    factory = EnvironmentRegistry().get_component_by_name("pine_wm")()
    arena_env = factory.build(PineWmEnvironmentCfg(enable_cameras=True, asset_root=args.asset_root))
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(device=args.device, solve_relations=False))
    env = builder.make_registered()
    try:
        base = env.unwrapped
        env.reset()
        robot = base.scene["robot"]
        indices, _ = robot.find_joints([*UR_ARM_JOINT_NAMES, "finger_joint"], preserve_order=True)
        target = robot.data.default_joint_pos[:, indices].clone()
        names = arena_env.embodiment.camera_config.camera_names()
        for _ in range(45):
            obs, *_ = env.step(target)
        assert set(obs["camera_obs"]) == {f"{name}_rgb" for name in names}
        hold_error = (robot.data.joint_pos[:, indices] - target).abs().max().item()
        assert hold_error < 0.02, f"Robot drifted: {hold_error} rad"
        positions_before = {name: base.scene[name].data.pos_w.clone() for name in names}
        moved = target.clone()
        moved[:, 5] += 0.15
        for _ in range(20):
            env.step(moved)
        displacement = {}
        for name in names:
            position = base.scene[name].data.pos_w
            displacement[name] = torch.linalg.vector_norm(position - positions_before[name]).item()
        assert displacement["wrist_a"] > 0.001 and displacement["wrist_b"] > 0.001, displacement
        assert displacement["realsense_d435"] < 1e-6 and displacement["scene_cam"] < 1e-6, displacement
        env.reset()
        for _ in range(45):
            env.step(target)
        reset_error = {}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        panels = []
        camera_report = {}
        for name in names:
            sensor = base.scene[name]
            reset_error[name] = torch.linalg.vector_norm(sensor.data.pos_w - positions_before[name]).item()
            assert reset_error[name] < 0.001, reset_error
            rgb = sensor.data.output["rgb"][0].cpu().numpy()[..., :3]
            assert np.isfinite(rgb).all() and float(rgb.std()) > 5.0, f"Empty camera: {name}"
            image = Image.fromarray(rgb.astype(np.uint8))
            image.save(args.output_dir / f"{name}.png")
            panel = Image.new("RGB", (640, 510), (25, 25, 25))
            preview = image.copy()
            preview.thumbnail((640, 480))
            panel.paste(preview, ((640 - preview.width) // 2, 30 + (480 - preview.height) // 2))
            ImageDraw.Draw(panel).text((12, 8), name, fill="white")
            panels.append(panel)
            camera_report[name] = {
                "resolution": list(image.size),
                "prim_path": sensor.cfg.prim_path,
                "position_world_m": sensor.data.pos_w[0].cpu().tolist(),
                "quaternion_world_ros_xyzw": sensor.data.quat_w_ros[0].cpu().tolist(),
            }
        overview = Image.new("RGB", (1280, 1020))
        for index, panel in enumerate(panels):
            overview.paste(panel, ((index % 2) * 640, (index // 2) * 510))
        overview.save(args.output_dir / "all_views.png")
        # Additional inspection view for the photo-derived wrist mounting hardware.
        detail_camera = base.scene["scene_cam"]
        detail_camera.set_world_poses_from_view(
            eyes=torch.tensor([[-0.42, -0.40, 1.20]], device=base.device),
            targets=torch.tensor([[-0.10, -0.055, 0.94]], device=base.device),
        )
        for _ in range(12):
            env.step(target)
        detail = detail_camera.data.output["rgb"][0].cpu().numpy()[..., :3]
        Image.fromarray(detail.astype(np.uint8)).save(args.output_dir / "gripper_detail.png")
        report = {
            "environment": "pine_wm",
            "robot_usd": arena_env.embodiment.robot_spec.usd_path,
            "joint_names": [*UR_ARM_JOINT_NAMES, "finger_joint"],
            "joint_target_rad": target[0].cpu().tolist(),
            "max_hold_error_rad": hold_error,
            "camera_motion_test_m": displacement,
            "camera_reset_error_m": reset_error,
            "cameras": camera_report,
        }
        (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PINE_WM_VALIDATED: {args.output_dir.resolve()}", flush=True)
    finally:
        env.close()


exit_code = 0
try:
    main()
except BaseException:
    import traceback

    exit_code = 1
    traceback.print_exc()
    raise
finally:
    launcher.app.close(exit_code=exit_code)
