# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render G2 pre-action states with the shared bounded microstep protocol."""

import hashlib
import json
import time
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main():
    """Write three validated native camera streams without altering a source recording."""
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--raw_dir", type=Path)
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument(
        "--env", default="g2_clean_workcell_table", choices=["g2_clean_workcell_table", "g2_stack_bowls", "g2_sleeve"]
    )
    parser.add_argument("--env_config", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frame_indices", type=int, nargs="+", help="Diagnostic source frames; not a training episode")
    args = parser.parse_args()
    assert args.device.startswith("cuda") and args.enable_cameras and args.num_envs == 1
    assert bool(args.raw_dir) != bool(args.hdf5), "Choose raw_dir or hdf5"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with SimulationAppContext(args):
        import h5py
        import torch
        import yaml

        import imageio.v3 as iio

        from data_engine.g2.collection.workcell.cameras import CAMERA_NAMES, ThreeViewWriter
        from data_engine.recording.alignment import (
            REPLAY_DT,
            ReplayDrift,
            pre_step_states,
            transition_metadata,
            validate_video,
        )
        from isaaclab_arena.assets.registries import EnvironmentRegistry
        from isaaclab_arena.embodiments.g2.g2 import G2CameraCfg, G2JointPositionActionsCfg
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena_environments.g2_clean_workcell_environment import G2CleanWorkcellTableEnvironment

        factory = EnvironmentRegistry().get_component_by_name(args.env)()
        source_success = None
        if args.raw_dir:
            assert args.env == "g2_clean_workcell_table"
            status = json.loads((args.raw_dir / "status.json").read_text())
            source_success = status["task_success"]
            raw = args.raw_dir / ("episodes.hdf5" if source_success else "episodes_failed.hdf5")
            spec = yaml.safe_load((args.raw_dir / "sources/clean_workcell_table.yaml").read_text())
            factory = G2CleanWorkcellTableEnvironment(spec)
        else:
            raw = args.hdf5
            assert args.env_config, "Historical episodes require their saved environment config"
        overrides = json.loads(args.env_config.read_text()) if args.env_config else {}
        overrides["enable_cameras"] = True
        description = factory.build(factory._legacy_argparse_cfg_type(**overrides))
        from data_engine.motion.embodiments.g2 import configure_g2_placement

        configure_g2_placement(description)
        description.embodiment.camera_config = G2CameraCfg()
        description.embodiment.camera_config.set_use_tiled_camera(False)
        description.embodiment.action_config = G2JointPositionActionsCfg()
        original = description.env_cfg_callback

        def configure(cfg):
            if original:
                cfg = original(cfg)
            cfg.sim.dt = REPLAY_DT
            cfg.sim.render_interval = 1
            cfg.recorders = None
            return cfg

        description.env_cfg_callback = configure
        env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
        base = env.unwrapped
        fps = 15.0  # Source control clock; replay microsteps never redefine video timestamps.
        report = transition_metadata("g2", list(CAMERA_NAMES), fps)
        with raw.open("rb") as source_stream:
            report["source_hdf5_sha256"] = hashlib.file_digest(source_stream, "sha256").hexdigest()
        report["source_episode"] = args.episode
        report.update(
            validation_passed=False,
            exact_states=False,
            replay_mode="physics_microstep",
            physics_dt_s=REPLAY_DT,
            visual_sync_review_required=True,
            source_task_success=source_success,
            diagnostic_only=bool(args.frame_indices),
        )
        report_path = args.output_dir / "render_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        writer = ThreeViewWriter(args.output_dir, fps)
        started = time.perf_counter()
        try:
            base.reset()
            ids = torch.tensor([0], device=base.device, dtype=torch.int32)
            with h5py.File(raw, "r") as file:
                demo = file[f"data/{args.episode}"]
                report["source_task_success"] = bool(demo.attrs.get("success", False))
                states = pre_step_states(demo)
                count = len(demo["actions"])
            indices = args.frame_indices if args.frame_indices is not None else list(range(count))
            assert indices and indices == sorted(set(indices)) and 0 <= indices[0] <= indices[-1] < count
            drift = ReplayDrift()
            first, motion, contrast = {}, dict.fromkeys(CAMERA_NAMES, 0.0), dict.fromkeys(CAMERA_NAMES, 0.0)

            def restore(index):
                state = {}
                for kind, assets in states.items():
                    state[kind] = {}
                    for name, fields in assets.items():
                        state[kind][name] = {
                            key: torch.as_tensor(value[index : index + 1], device=base.device)
                            for key, value in fields.items()
                        }
                base.scene.reset_to(state, ids, is_relative=True)
                # Hold the restored joints while flushing PhysX -> Hydra; do not apply action[t].
                base.scene["robot"].set_joint_position_target(state["articulation"]["robot"]["joint_position"])
                base.scene.write_data_to_sim()
                base.sim.step(render=True)
                base.scene.update(base.sim.get_physics_dt())
                for name in CAMERA_NAMES:
                    base.scene[name].update(REPLAY_DT, force_recompute=True)

            for _ in range(20):
                restore(indices[0])
            for output_index, index in enumerate(indices):
                restore(index)
                drift.update(base, states, index)
                writer.append(base.scene)
                views = []
                for name in CAMERA_NAMES:
                    data = base.scene[name].data.output["rgb"]
                    data = data.torch if hasattr(data, "torch") else data
                    rgb = data[0, ..., :3]
                    pixels = rgb.float()
                    first.setdefault(name, pixels.clone())
                    motion[name] = max(motion[name], float((pixels - first[name]).abs().mean()))
                    contrast[name] = max(contrast[name], float(pixels.std()))
                    if output_index in (0, len(indices) // 2, len(indices) - 1):
                        if name == "head_camera":
                            padded = torch.zeros((528, 640, 3), device=rgb.device, dtype=rgb.dtype)
                            padded[64:464] = rgb
                            rgb = padded
                        views.append(rgb)
                if views:
                    iio.imwrite(args.output_dir / f"three_views_{index:06d}.png", torch.cat(views, dim=1).cpu().numpy())
                if output_index % 300 == 0:
                    print(f"STATE_RENDER {output_index}/{len(indices)} source={index}", flush=True)
            writer.close()
            videos = {
                name: validate_video(args.output_dir / f"{name}.mp4", len(indices), fps, base.scene[name].image_shape)
                for name in CAMERA_NAMES
            }
            report.update(drift.report())
            report["joint_names"] = {name: base.scene[name].joint_names for name in drift.joint_errors}
            report.update(
                frames=len(indices),
                source_frames=count,
                source_frame_indices=indices,
                source_timestamps_s=[index / fps for index in indices],
                videos=videos,
                camera_devices=writer.devices,
                image_motion=motion,
                image_contrast=contrast,
                image_validation_passed=all(v > 3 for v in contrast.values()) and all(v > 1 for v in motion.values()),
                wall_seconds=time.perf_counter() - started,
            )
            report["validation_passed"] = report["state_validation_passed"] and report["image_validation_passed"]
            report_path.write_text(json.dumps(report, indent=2))
            assert report["validation_passed"], f"Replay rejected; inspect {report_path}"
            print("STATE_RENDER_COMPLETE", json.dumps(report), flush=True)
        finally:
            writer.close()
            env.close()


if __name__ == "__main__":
    import os

    assert os.environ.get("ARENA_G2_FRAMEWORK") == "1", "Use data_engine/g2/cli.py"
    main()
