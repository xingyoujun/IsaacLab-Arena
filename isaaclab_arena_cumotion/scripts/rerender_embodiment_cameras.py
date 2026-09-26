# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Re-render an embodiment's calibrated cameras for states-only demonstrations.

Demonstrations are recorded without images (see the cuMotion drivers); this replays each demo's
recorded states frame by frame and writes one mp4 per requested camera stream into a
``<hdf5>.cameras/`` sidecar directory, which ``convert_hdf5_to_lerobot.py`` copies into the LeRobot
dataset. Unlike ``rerender_demo_cameras.py`` (Agibot: head + wrist cameras posed by hand) this uses
the environment's own camera sensors at their configured resolution. Robot-mounted cameras
follow the replayed articulation states, including pine_wm's two calibrated wrist cameras.

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
parser.add_argument(
    "--env-config", type=str, default=None, help="JSON environment config overrides (e.g. first20 task_id)."
)
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
import os  # noqa: E402
import pathlib  # noqa: E402
import tempfile  # noqa: E402
import torch  # noqa: E402

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

import imageio.v2 as iio  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.recording.alignment import (  # noqa: E402
    REPLAY_DT,
    pre_step_states,
    transition_metadata,
    validate_video,
)

FPS = 15  # one frame per control step at Arena's 15 Hz control rate; the videos play in real time

# ------------------------------------------------------------------------------------- env ---
arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
cfg_kwargs = json.loads(pathlib.Path(args.env_config).read_text()) if args.env_config else {}
if args.env == "pine_wm_first20" and args.env_config is None:
    recorded_configs = []
    for hdf5_path in args.hdf5:
        with h5py.File(hdf5_path, "r") as dataset:
            metadata = json.loads(dataset["data"].attrs["env_args"])
            assert "env_cfg" in metadata, "First20 recordings require env_cfg metadata or --env-config"
            recorded_configs.append(metadata["env_cfg"])
    assert all(config == recorded_configs[0] for config in recorded_configs), "Render different tasks separately"
    cfg_kwargs = recorded_configs[0]

cfg_kwargs["enable_cameras"] = True
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

if args.env == "pine_wm_first20":
    original_callback = arena_env.env_cfg_callback

    def configure_microstep_replay(cfg):
        if original_callback is not None:
            cfg = original_callback(cfg)
        # PhysX must publish a completed step for Hydra to consume restored poses.
        # A microsecond step bounds motion drift while flushing those render buffers.
        cfg.sim.dt = REPLAY_DT
        cfg.sim.render_interval = 1
        return cfg

    arena_env.env_cfg_callback = configure_microstep_replay

builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
if args.env != "pine_wm_first20":
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
    if args.env == "pine_wm_first20":
        states = pre_step_states(demo)

    writers = {
        # Default libx264 quality, matching the Agibot datasets: higher quality settings made each
        # video ~10x larger for no training benefit.
        name: iio.get_writer(out_dir / f"{demo_name}_{name}.part.mp4", fps=FPS, codec="h264_nvenc", macro_block_size=8)
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

    bounded_replay = args.env == "pine_wm_first20"
    max_joint_error = 0.0
    max_position_error = 0.0
    max_root_rotation_error = 0.0
    joint_errors = {}
    first_images = {}
    image_motion = {name: 0.0 for name in cams}
    image_contrast = {name: 0.0 for name in cams}

    def render_state():
        env.sim.step(render=True)
        if bounded_replay:
            env.scene.update(env.sim.get_physics_dt())

    # Flush temporal image history from the same recorded initial state.
    env.scene.reset_to(frame_state(0), env_ids, is_relative=True)
    for _ in range(args.warmup_frames):
        if bounded_replay:
            env.scene.reset_to(frame_state(0), env_ids, is_relative=True)
        render_state()
        for cam in cams.values():
            cam.update(env.sim.get_physics_dt(), force_recompute=True)

    for step in range(num_steps):
        env.scene.reset_to(frame_state(step), env_ids, is_relative=True)
        render_state()
        for name, cam in cams.items():
            cam.update(env.sim.get_physics_dt(), force_recompute=bounded_replay)
            pixels = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            writers[name].append_data(pixels)
            if bounded_replay:
                if name not in first_images:
                    first_images[name] = pixels.astype(np.float32)
                image_motion[name] = max(image_motion[name], float(np.abs(pixels - first_images[name]).mean()))
                image_contrast[name] = max(image_contrast[name], float(pixels.std()))
        if bounded_replay:
            for kind, assets in states.items():
                for asset_name, fields in assets.items():
                    asset = env.scene[asset_name]
                    # Read the physics backend, not the reset_to() write-through caches.
                    transforms = (
                        asset.root_view.get_root_transforms()
                        if kind == "articulation"
                        else asset.root_view.get_transforms()
                    )
                    actual_pose = wp.to_torch(transforms).cpu().numpy()[0]
                    actual_position = actual_pose[:3]
                    actual_quaternion = actual_pose[3:].astype(np.float64)
                    recorded_quaternion = fields["root_pose"][step, 3:].astype(np.float64)
                    cosine = abs(float(actual_quaternion @ recorded_quaternion)) / (
                        np.linalg.norm(actual_quaternion) * np.linalg.norm(recorded_quaternion)
                    )
                    max_root_rotation_error = max(max_root_rotation_error, float(2 * np.arccos(np.clip(cosine, 0, 1))))
                    max_position_error = max(
                        max_position_error, float(np.max(np.abs(actual_position - fields["root_pose"][step, :3])))
                    )
                    if kind == "articulation":
                        actual_joint = wp.to_torch(asset.root_view.get_dof_positions()).cpu().numpy()[0]
                        errors = np.abs(actual_joint - fields["joint_position"][step])
                        joint_errors[asset_name] = np.maximum(
                            joint_errors.get(asset_name, np.zeros_like(errors)), errors
                        )
                        max_joint_error = max(max_joint_error, float(errors.max()))
    arm_joint_names = {
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    }
    arm_error = 0.0
    articulated_object_error = 0.0
    for asset_name, errors in joint_errors.items():
        if asset_name == "robot":
            arm_error = max(
                (
                    float(error)
                    for name, error in zip(env.scene[asset_name].joint_names, errors)
                    if name in arm_joint_names
                ),
                default=0.0,
            )
        else:
            articulated_object_error = max(articulated_object_error, float(errors.max()))
    # Bound camera-bearing arm rotation separately from compliant finger joints.
    state_valid = not bounded_replay or (
        arm_error < 1e-3
        and max_joint_error < 5e-3
        and max_position_error < 5e-4
        and articulated_object_error < 5e-4
        and max_root_rotation_error < 5e-3
    )
    image_valid = not bounded_replay or (
        all(value > 3 for value in image_contrast.values()) and all(value > 1 for value in image_motion.values())
    )
    (out_dir / f"{demo_name}_state_replay.json").write_text(
        json.dumps(
            {
                "collection_contract": transition_metadata(arena_env.embodiment.name, list(cams), FPS),
                "frames": num_steps,
                "fps": FPS,
                "observation_alignment": "pre_step" if bounded_replay else "legacy_post_step",
                "replay_mode": "physics_microstep" if bounded_replay else "physics_step",
                "physics_dt_s": env.sim.get_physics_dt(),
                "exact_states": False,
                "validation_passed": False,
                "state_validation_passed": state_valid,
                "image_validation_passed": image_valid,
                "joint_coordinate_errors": {
                    asset: dict(zip(env.scene[asset].joint_names, errors.tolist()))
                    for asset, errors in joint_errors.items()
                },
                "joint_error_limit_rad": 5e-3 if bounded_replay else None,
                "arm_joint_error_limit_rad": 1e-3 if bounded_replay else None,
                "max_arm_joint_error_rad": arm_error if bounded_replay else None,
                "articulated_object_coordinate_error_limit": 5e-4 if bounded_replay else None,
                "max_articulated_object_coordinate_error": articulated_object_error if bounded_replay else None,
                "root_position_error_limit_m": 5e-4 if bounded_replay else None,
                "root_rotation_error_limit_rad": 5e-3 if bounded_replay else None,
                "max_root_rotation_error_rad": max_root_rotation_error if bounded_replay else None,
                "state_error_source": "physics_backend" if bounded_replay else None,
                "visual_sync_review_required": True,
                "max_joint_error_rad": max_joint_error if bounded_replay else None,
                "max_root_position_error_m": max_position_error if bounded_replay else None,
                "streams": list(cams),
                "max_image_change_from_first_frame": image_motion,
                "max_image_std": image_contrast,
                "env_cfg": cfg_kwargs,
            },
            indent=2,
        )
        + "\n"
    )

    assert state_valid, (
        f"State replay drift: joints={max_joint_error:.9g} rad, root_position={max_position_error:.9g} m; "
        f"see {out_dir / (demo_name + '_state_replay.json')}"
    )
    assert image_valid, f"Frozen/empty images: motion={image_motion}, contrast={image_contrast}"

    # Only completed renders carry the final name; a killed worker leaves .part files behind,
    # which the skip check ignores, so relaunching resumes cleanly.
    for name, writer in writers.items():
        writer.close()
        validate_video(out_dir / f"{demo_name}_{name}.part.mp4", num_steps, FPS)
        (out_dir / f"{demo_name}_{name}.part.mp4").rename(out_dir / f"{demo_name}_{name}.mp4")
    manifest_path = out_dir / f"{demo_name}_state_replay.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["validation_passed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return num_steps


for path in args.hdf5:
    out_dir = pathlib.Path(f"{path}.cameras")
    out_dir.mkdir(parents=True, exist_ok=True)
    appearance_path = out_dir / "render_appearance.json"
    if args.env.startswith("ur7e_"):
        appearance = {"gripper": "black_fingertips_direct", "diffuse_color": [0.015, 0.015, 0.015], "roughness": 0.5}
        if appearance_path.exists():
            assert (
                json.loads(appearance_path.read_text()) == appearance
            ), "Use a fresh sidecar directory for new materials"
        else:
            assert not list(
                out_dir.glob("*.mp4")
            ), "Legacy camera cache: copy HDF5 to a new version before re-rendering"
            # Publish a complete marker atomically; two workers write identical metadata.
            with tempfile.NamedTemporaryFile(mode="w", dir=out_dir, suffix=".json.tmp", delete=False) as marker:
                json.dump(appearance, marker)
                temporary_path = pathlib.Path(marker.name)
            temporary_path.replace(appearance_path)
    # Read-only and unlocked so several workers may share one file.
    with h5py.File(path, "r", locking=False) as handle:
        names = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
        if args.demo_range is not None:
            names = names[args.demo_range[0] : args.demo_range[1]]
        print(f"{path}: {len(names)} demos -> {out_dir}")
        for i, name in enumerate(names):
            manifest = out_dir / f"{name}_state_replay.json"
            cache_valid = args.env != "pine_wm_first20" or (
                manifest.exists() and json.loads(manifest.read_text()).get("validation_passed", False)
            )
            if not args.force and cache_valid and all((out_dir / f"{name}_{stream}.mp4").exists() for stream in cams):
                print(f"  [{i + 1}/{len(names)}] {name}: already rendered, skipping")
                continue
            steps = rerender_demo(handle[f"data/{name}"], out_dir, name)
            print(f"  [{i + 1}/{len(names)}] {name}: {steps} steps rendered", flush=True)

print("all files done")
simulation_app.close()
