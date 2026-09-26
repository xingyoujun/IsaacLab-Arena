# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render G2 state-only trials with data engine's restore-and-physics-step method."""

import json
import time
import yaml
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main():
    """Write three native camera sidecars and quantify the extra physics step's pose error."""
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    assert args.device.startswith("cuda") and args.enable_cameras and args.num_envs == 1
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with SimulationAppContext(args):
        import h5py
        import numpy as np
        import torch

        import imageio.v3 as iio

        from isaaclab_arena.embodiments.g2.g2 import G2CameraCfg, G2JointPositionActionsCfg
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena_environments.g2_clean_workcell_environment import G2CleanWorkcellTableEnvironment
        from isaaclab_arena_environments.g2_workbench_environments import G2WorkbenchEnvironmentCfg
        from isaaclab_arena_examples.g2_workcell.cameras import CAMERA_NAMES, ThreeViewWriter

        status = json.loads((args.raw_dir / "status.json").read_text())
        raw = args.raw_dir / ("episodes.hdf5" if status["task_success"] else "episodes_failed.hdf5")
        scene_spec = yaml.safe_load((args.raw_dir / "sources" / "clean_workcell_table.yaml").read_text())
        description = G2CleanWorkcellTableEnvironment(scene_spec).build(G2WorkbenchEnvironmentCfg(enable_cameras=True))
        description.embodiment.camera_config = G2CameraCfg()
        description.embodiment.action_config = G2JointPositionActionsCfg()
        env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
        base = env.unwrapped
        trace = np.load(args.raw_dir / "step_trace.npz")
        writer = ThreeViewWriter(args.output_dir, 1 / base.step_dt)
        max_tcp_error = 0.0
        max_object_error = 0.0
        started = time.perf_counter()
        try:
            env.reset()
            ids = torch.tensor([0], device=base.device, dtype=torch.int32)
            with h5py.File(raw) as file:
                demo = file["data/demo_0"]
                actions = torch.as_tensor(demo["actions"][:], device=base.device)
                states = {
                    kind: {
                        name: {key: torch.as_tensor(value[:], device=base.device) for key, value in fields.items()}
                        for name, fields in objects.items()
                    }
                    for kind, objects in demo["states"].items()
                }
                count = len(actions)
                for index in range(count):
                    state = {
                        kind: {
                            name: {key: value[index : index + 1] for key, value in fields.items()}
                            for name, fields in objects.items()
                        }
                        for kind, objects in states.items()
                    }
                    base.scene.reset_to(state, ids, is_relative=True)
                    # Hold the recorded command during the one real physics tick needed to
                    # update CUDA graphics; measure the resulting deviation rather than hide it.
                    base.action_manager.process_action(actions[index : index + 1])
                    base.action_manager.apply_action()
                    base.scene.write_data_to_sim()
                    base.sim.step(render=True)
                    base.scene.update(base.sim.get_physics_dt())
                    writer.append(base.scene)
                    actual = torch.cat(
                        [base.scene[name].data.target_pos_w.torch[0, 0] for name in ("ee_frame", "left_ee_frame")]
                    )
                    expected = torch.as_tensor(trace["tcp_positions"][index], device=base.device)
                    max_tcp_error = max(
                        max_tcp_error, float((actual.reshape(2, 3) - expected.reshape(2, 3)).norm(dim=1).max())
                    )
                    for oi, name in enumerate(trace["object_names"]):
                        error = (
                            base.scene[str(name)].data.root_pos_w.torch[0]
                            - torch.as_tensor(trace["object_positions"][index, oi], device=base.device)
                        ).norm()
                        max_object_error = max(max_object_error, float(error))
                    if index in (0, count // 2, count - 1):
                        views = []
                        for name in CAMERA_NAMES:
                            rgb = base.scene[name].data.output["rgb"].torch[0, ..., :3]
                            if name == "head_camera":
                                padded = torch.zeros((528, 640, 3), device=rgb.device, dtype=rgb.dtype)
                                padded[64:464] = rgb
                                rgb = padded
                            views.append(rgb)
                        iio.imwrite(
                            args.output_dir / f"three_views_{index:06d}.png", torch.cat(views, dim=1).cpu().numpy()
                        )
                    if index % 300 == 0:
                        print(f"STATE_RENDER {index}/{count}", flush=True)
            result = dict(
                source_task_success=status["task_success"],
                frames=count,
                fps=1 / base.step_dt,
                image_time_convention="recorded post-action state advanced by one physics tick",
                physics_advance_seconds=base.sim.get_physics_dt(),
                maximum_tcp_error_m=max_tcp_error,
                maximum_object_position_error_m=max_object_error,
                exact_pre_action_alignment=False,
                render_and_encode_wall_seconds=time.perf_counter() - started,
                camera_devices=writer.devices,
            )
            (args.output_dir / "render_report.json").write_text(json.dumps(result, indent=2))
            print("STATE_RENDER_COMPLETE", json.dumps(result), flush=True)
        finally:
            writer.close()
            env.close()


if __name__ == "__main__":
    main()
