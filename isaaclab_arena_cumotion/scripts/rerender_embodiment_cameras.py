# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Re-render an embodiment's own (world-fixed) cameras for states-only demonstrations.

Demonstrations are recorded without images (see the cuMotion drivers); this replays each demo's
recorded states frame by frame and writes one mp4 per requested camera stream into a
``<hdf5>.cameras/`` sidecar directory, which ``convert_hdf5_to_lerobot.py`` copies into the LeRobot
dataset. Unlike ``rerender_demo_cameras.py`` (Agibot: head + wrist cameras posed by hand) this uses
the environment's own camera sensors at their configured resolution, so it works for any embodiment
whose cameras are static in the world (the UR7e workcell's calibrated D435 and its ``scene_cam``).

With ``--randomize`` the workcell's lights, table/floor materials and a few distractor objects are
randomized per demo (see ``ur7e_workcell_randomization.py``); the draw is seeded by the demo index,
so workers and relaunches render a demo identically, and the draw is logged next to the videos.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/rerender_embodiment_cameras.py \\
        --hdf5 /path/open_drawer.hdf5 --env ur7e_open_drawer --streams realsense_d435_rgb \\
        --randomize --skies-dir /home/ubuntu/playground/assets/skies
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5", type=str, nargs="+", required=True, help="Recorded HDF5 file(s).")
parser.add_argument("--env", type=str, required=True, help="Registered environment the demos were recorded in.")
parser.add_argument("--embodiment", type=str, default=None, help="Embodiment override (env default otherwise).")
parser.add_argument(
    "--streams", type=str, nargs="+", default=["realsense_d435_rgb"], help="Camera streams (<camera>_rgb) to render."
)
parser.add_argument("--demo-range", type=int, nargs=2, default=None, help="Half-open demo index range to render.")
parser.add_argument("--force", action="store_true", help="Re-render streams that already exist.")
parser.add_argument("--randomize", action="store_true", help="Randomize lights, materials and distractors per demo.")
parser.add_argument("--randomize-seed", type=int, default=0, help="Base seed of the per-demo randomization.")
parser.add_argument("--skies-dir", type=str, default=None, help="Directory of .hdr skies for the dome light.")
parser.add_argument("--max-distractors", type=int, default=6, help="Upper bound on distractor objects per demo.")
parser.add_argument(
    "--keep-clear-radius", type=float, default=0.28, help="No distractor within this radius (m) of a task object."
)
parser.add_argument("--warmup-frames", type=int, default=20, help="Camera renders per demo before frames are written.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
import pathlib  # noqa: E402
import torch  # noqa: E402

import imageio.v2 as iio  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402

FPS = 15  # one frame per control step at Arena's 15 Hz control rate; the videos play in real time

# ------------------------------------------------------------------------------------- env ---
arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
cfg_kwargs = {"enable_cameras": True}
if args.embodiment is not None:
    cfg_kwargs["embodiment"] = args.embodiment
arena_env = factory.build(factory._legacy_argparse_cfg_type(**cfg_kwargs))

camera_rig = arena_env.embodiment.camera_config
camera_keys = {}
for stream in args.streams:
    assert stream.endswith("_rgb"), f"stream '{stream}' must be a <camera>_rgb key"
    camera_keys[stream] = stream[: -len("_rgb")]
    assert (
        camera_keys[stream] in camera_rig.camera_names()
    ), f"camera '{camera_keys[stream]}' is not in the embodiment rig {camera_rig.camera_names()}"
# Untiled sensors are the ones whose per-camera output this script reads.
camera_rig.set_use_tiled_camera(False)

builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
env.sim.reset()
env.reset()
device = env.device
env_ids = torch.tensor([0], device=device)
cams = {stream: env.scene[camera_key] for stream, camera_key in camera_keys.items()}
for stream, cam in cams.items():
    print(f"{stream}: {cam.image_shape} from {type(cam).__name__}")

randomizer = None
if args.randomize:
    from isaaclab_arena_environments.ur7e_workcell_randomization import WorkcellVisualRandomizer

    randomizer = WorkcellVisualRandomizer(
        env.sim.stage, seed=args.randomize_seed, skies_dir=args.skies_dir, max_distractors=args.max_distractors
    )
    print(f"randomizing with seed {args.randomize_seed}, {len(randomizer.skies)} skies")


# -------------------------------------------------------------------------------- playback ---
def task_object_footprints(demo, radius: float) -> list[tuple[float, float, float]]:
    """``(x, y, radius)`` discs around every non-robot asset of the demo's initial state."""
    discs = []
    for kind in demo["initial_state"]:
        for asset in demo["initial_state"][kind]:
            if asset == "robot" or "root_pose" not in demo["initial_state"][kind][asset]:
                continue
            pose = np.array(demo["initial_state"][kind][asset]["root_pose"])[0]
            discs.append((float(pose[0]), float(pose[1]), radius))
    return discs


def rerender_demo(demo, out_dir: pathlib.Path, demo_name: str) -> int:
    """Replay one demo's states and write its camera streams as sidecar mp4s."""
    if randomizer is not None:
        index = int(demo_name.split("_")[-1])
        summary = randomizer.randomize(index, keep_clear=task_object_footprints(demo, args.keep_clear_radius))
        (out_dir / f"{demo_name}_randomization.json").write_text(json.dumps(summary, indent=1))
        print(f"    randomization: {summary}")
    group = demo["states"]
    states = {
        kind: {
            asset: {field: np.array(group[kind][asset][field]) for field in group[kind][asset]} for asset in group[kind]
        }
        for kind in group
    }
    num_steps = next(iter(states["articulation"]["robot"].values())).shape[0]

    writers = {
        # Default libx264 quality, matching the Agibot datasets: higher quality settings made each
        # video ~10x larger for no training benefit.
        name: iio.get_writer(out_dir / f"{demo_name}_{name}.part.mp4", fps=FPS, codec="libx264", macro_block_size=8)
        for name in cams
    }

    def frame_state(step: int) -> dict:
        return {
            kind: {
                asset: {
                    field: torch.tensor(array[step], dtype=torch.float32, device=device).unsqueeze(0)
                    for field, array in fields.items()
                }
                for asset, fields in assets.items()
            }
            for kind, assets in states.items()
        }

    # Settle the first frame's state and flush the renderer's temporal history (DLSS/denoiser accumulate over
    # frames, so the previous demo's scene ghosts into the first frames otherwise). Cameras render when they
    # are updated, not on sim.step(), so the warm-up must update them.
    env.scene.reset_to(frame_state(0), env_ids, is_relative=True)
    for _ in range(args.warmup_frames):
        env.sim.step(render=True)
        for cam in cams.values():
            cam.update(env.sim.get_physics_dt(), force_recompute=True)

    for step in range(num_steps):
        env.scene.reset_to(frame_state(step), env_ids, is_relative=True)
        env.sim.step(render=True)
        for name, cam in cams.items():
            cam.update(env.sim.get_physics_dt())
            writers[name].append_data(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))

    # Only completed renders carry the final name; a killed worker leaves .part files behind,
    # which the skip check ignores, so relaunching resumes cleanly.
    for name, writer in writers.items():
        writer.close()
        (out_dir / f"{demo_name}_{name}.part.mp4").rename(out_dir / f"{demo_name}_{name}.mp4")
    return num_steps


for path in args.hdf5:
    out_dir = pathlib.Path(f"{path}.cameras")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Read-only and unlocked so several workers may share one file.
    with h5py.File(path, "r", locking=False) as handle:
        names = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
        if args.demo_range is not None:
            names = names[args.demo_range[0] : args.demo_range[1]]
        print(f"{path}: {len(names)} demos -> {out_dir}")
        for i, name in enumerate(names):
            if not args.force and all((out_dir / f"{name}_{stream}.mp4").exists() for stream in cams):
                print(f"  [{i + 1}/{len(names)}] {name}: already rendered, skipping")
                continue
            steps = rerender_demo(handle[f"data/{name}"], out_dir, name)
            print(f"  [{i + 1}/{len(names)}] {name}: {steps} steps rendered", flush=True)

print("all files done")
simulation_app.close()
