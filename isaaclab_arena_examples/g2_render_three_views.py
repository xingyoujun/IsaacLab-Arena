# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render the native G2 head view alongside recorded wrist views without changing a demonstration."""

import json
import shutil
from contextlib import ExitStack
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def render(args):
    """Export synchronized pre-action native head and wrist videos from a saved episode."""
    import h5py
    import numpy as np
    import torch

    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    from isaaclab_arena.embodiments.g2.g2 import G2CameraCfg
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.g2_stack_bowls_environment import (
        G2StackBowlsEnvironment,
        G2StackBowlsEnvironmentCfg,
    )

    description = G2StackBowlsEnvironment().build(
        G2StackBowlsEnvironmentCfg(enable_cameras=True, hdr=None, table_height_m=args.table_height_m)
    )
    rig = G2CameraCfg()
    rig.left_wrist_camera = None
    rig.right_wrist_camera = None
    description.embodiment.camera_config = rig
    args.num_envs = 1
    env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
    base = env.unwrapped
    try:
        env.reset()
        with ExitStack() as stack:
            dataset = stack.enter_context(h5py.File(args.dataset, "r"))
            episode = dataset[f"data/{args.episode}"]
            count = len(episode["actions"])
            fps = 1.0 / base.step_dt
            names = ("head", "left_wrist", "right_wrist", "three_views")
            writers = {
                name: stack.enter_context(
                    imageio.get_writer(str(args.output_dir / f"{name}.mp4"), fps=fps, codec="libx264", quality=8)
                )
                for name in names
            }
            env_ids = torch.tensor([0], dtype=torch.int32, device=base.device)
            wrists = {side: episode[f"camera_obs/{side}_wrist_camera_rgb"] for side in ("left", "right")}
            # Read aligned slabs instead of repeatedly decompressing multi-frame chunks.
            block_size = wrists["left"].chunks[0] if wrists["left"].chunks else 32
            wrist_frames = {}
            for index in range(count):
                if index % block_size == 0:
                    wrist_frames = {side: camera[index : index + block_size] for side, camera in wrists.items()}
                group = episode["initial_state"] if index == 0 else episode["states"]
                source_index = 0 if index == 0 else index - 1
                state = {
                    category: {
                        name: {
                            key: torch.tensor(value[source_index], device=base.device).unsqueeze(0)
                            for key, value in fields.items()
                        }
                        for name, fields in objects.items()
                    }
                    for category, objects in group.items()
                }
                base.scene.reset_to(state, env_ids=env_ids, is_relative=True)
                base.sim.forward()
                base.sim.render_context.reset_scene_state_cadence()
                base.scene.update(base.step_dt)
                base.sim.render()
                base.sim.render()
                observations = base.observation_manager.compute()
                frames = [observations["camera_obs"]["head_camera_rgb"][0].cpu().numpy()]
                frames.extend(wrist_frames[side][index % block_size] for side in ("left", "right"))
                canvas = np.zeros((560, 1920, 3), dtype=np.uint8)
                for column, (name, frame) in enumerate(zip(names, frames)):
                    writers[name].append_data(frame)
                    height, width = frame.shape[:2]
                    top = 32 + (528 - height) // 2
                    canvas[top : top + height, column * 640 : column * 640 + width] = frame
                labeled = Image.fromarray(canvas)
                draw = ImageDraw.Draw(labeled)
                for column, label in enumerate(("HEAD (native camera)", "LEFT WRIST", "RIGHT WRIST")):
                    draw.text((column * 640 + 12, 8), label, fill="white")
                writers["three_views"].append_data(np.asarray(labeled))
                if index in (0, count // 2, count - 1):
                    labeled.save(args.output_dir / f"preview_{index:04d}.png")
                if index % 100 == 0:
                    print("THREE_VIEW_RENDER", index, count, flush=True)
            metadata = {
                "source_dataset": str(args.dataset),
                "episode": args.episode,
                "frames": count,
                "fps": fps,
                "table_height_m": args.table_height_m,
                "head": "Native authored G2 head camera, re-rendered from recorded pre-action physical states",
                "wrists": "Original recorded pre-action RGB observations",
                "source_modified": False,
                "physics_stepped_during_replay": False,
                "camera_configuration": "isaaclab_arena.embodiments.g2.g2:G2CameraCfg",
            }
            (args.output_dir / "render_metadata.json").write_text(json.dumps(metadata, indent=2))
        print("THREE_VIEW_COMPLETE", count, flush=True)
    finally:
        env.close()


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--headless", action="store_true", help="Run without a viewer (also the GA default)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--table_height_m", type=float, default=0.75)
    args = parser.parse_args()
    assert args.enable_cameras, "Rendering requires --enable_cameras"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output_dir / "render_source.py")
    with SimulationAppContext(args):
        render(args)


if __name__ == "__main__":
    main()
