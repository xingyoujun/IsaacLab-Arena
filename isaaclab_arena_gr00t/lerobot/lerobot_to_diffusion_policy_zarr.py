# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a UR7e LeRobot dataset (with ``eef_9d``) into a diffusion_policy zarr replay buffer.

Output layout (``diffusion_policy.common.replay_buffer.ReplayBuffer``):

* ``data/camera_0``        uint8 (T, H, W, 3), one frame per control step, Jpeg2k-compressed
* ``data/robot_eef_pose``  float32 (T, 9): xyz + rotation_6d in the ROW convention pytorch3d /
  diffusion_policy's ``RotationTransformer`` use (the LeRobot ``eef_9d`` stores COLUMNS)
* ``data/gripper_pos``     float32 (T, 1): finger_joint / (pi/4), 0 open .. 1 closed
* ``data/robot_joint``     float32 (T, 6): arm joints (rad)
* ``data/action``          float32 (T, 10): absolute target eef pose (xyz + rotation_6d rows) + gripper
* ``data/timestamp``       float64 (T,)
* ``meta/episode_ends``    int64 (N,)

Run it in the diffusion_policy conda environment (needs zarr, numcodecs, imagecodecs, av)::

    python lerobot_to_diffusion_policy_zarr.py --lerobot <dataset root> --out <path>.zarr [--image-size 320 240]

The images are compressed with the Jpeg2k codec class vendored in the diffusion_policy repo
(``--dp-repo``), not imagecodecs' own: diffusion_policy registers its vendored class at load time
and refuses codec configs written by newer imagecodecs releases (``unexpected keyword 'mct'``).
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

GRIPPER_CLOSED_RAD = np.pi / 4


def rot6d_columns_to_rows(rot6d_cols: np.ndarray) -> np.ndarray:
    """Convert the first two matrix columns (c0, c1) into the first two rows (pytorch3d rotation_6d)."""
    c0, c1 = rot6d_cols[:, 0:3], rot6d_cols[:, 3:6]
    c2 = np.cross(c0, c1)
    matrices = np.stack([c0, c1, c2], axis=2)  # (T, 3, 3) with columns c0 c1 c2
    return matrices[:, :2, :].reshape(len(rot6d_cols), 6)


def read_video(path: Path, num_frames: int, size: tuple[int, int]) -> np.ndarray:
    import av
    from PIL import Image

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            image = frame.to_image()
            if image.size != size:
                image = image.resize(size, Image.BILINEAR)
            frames.append(np.asarray(image, dtype=np.uint8))
            if len(frames) == num_frames:
                break
    assert len(frames) == num_frames, f"{path}: {len(frames)} frames decoded, {num_frames} rows expected"
    return np.stack(frames)


def read_parquet(path: Path):
    """Read a LeRobot episode parquet with pyarrow directly (pandas' engine check rejects older pyarrow)."""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output .zarr directory (or .zarr.zip).")
    parser.add_argument("--video-key", default="observation.images.realsense_d435")
    parser.add_argument("--image-size", type=int, nargs=2, default=(320, 240), metavar=("W", "H"))
    parser.add_argument("--jpeg2k-level", type=int, default=50)
    parser.add_argument(
        "--dp-repo", type=Path, default=Path("/home/ubuntu/code/diffusion_policy"), help="diffusion_policy checkout."
    )
    args = parser.parse_args()

    import sys

    import zarr
    from numcodecs import Blosc

    sys.path.insert(0, str(args.dp_repo))
    from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs

    register_codecs()

    info = json.loads((args.lerobot / "meta" / "info.json").read_text())
    fps = info["fps"]
    episodes = [
        json.loads(line) for line in (args.lerobot / "meta" / "episodes.jsonl").read_text().splitlines() if line
    ]
    width, height = args.image_size

    store = (
        zarr.ZipStore(str(args.out), mode="w") if str(args.out).endswith(".zip") else zarr.DirectoryStore(str(args.out))
    )
    root = zarr.group(store=store)
    data = root.create_group("data")
    meta = root.create_group("meta")
    total = sum(ep["length"] for ep in episodes)
    images = data.create_dataset(
        "camera_0",
        shape=(total, height, width, 3),
        chunks=(1, height, width, 3),
        dtype=np.uint8,
        compressor=Jpeg2k(level=args.jpeg2k_level),
    )
    lowdim = {
        "robot_eef_pose": np.zeros((total, 9), np.float32),
        "gripper_pos": np.zeros((total, 1), np.float32),
        "robot_joint": np.zeros((total, 6), np.float32),
        "action": np.zeros((total, 10), np.float32),
        "timestamp": np.zeros((total,), np.float64),
    }
    ends, cursor = [], 0
    for ep in episodes:
        index = ep["episode_index"]
        chunk = index // info["chunks_size"]
        df = read_parquet(args.lerobot / info["data_path"].format(episode_chunk=chunk, episode_index=index))
        n = len(df)
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        eef_state = np.stack(df["observation.eef_9d"].to_numpy()).astype(np.float32)
        eef_action = np.stack(df["action.eef_9d"].to_numpy()).astype(np.float32)
        sl = slice(cursor, cursor + n)
        lowdim["robot_eef_pose"][sl] = np.concatenate(
            [eef_state[:, :3], rot6d_columns_to_rows(eef_state[:, 3:9])], axis=1
        )
        lowdim["gripper_pos"][sl] = state[:, 6:7] / GRIPPER_CLOSED_RAD
        lowdim["robot_joint"][sl] = state[:, :6]
        lowdim["action"][sl] = np.concatenate(
            [eef_action[:, :3], rot6d_columns_to_rows(eef_action[:, 3:9]), action[:, 6:7] / GRIPPER_CLOSED_RAD], axis=1
        )
        lowdim["timestamp"][sl] = cursor / fps + np.arange(n) / fps
        video = args.lerobot / info["video_path"].format(
            episode_chunk=chunk, video_key=args.video_key, episode_index=index
        )
        images[sl] = read_video(video, n, (width, height))
        cursor += n
        ends.append(cursor)
        print(f"episode {index}: {n} steps")
    for key, array in lowdim.items():
        data.create_dataset(key, data=array, chunks=(min(total, 4096),) + array.shape[1:], compressor=Blosc("zstd", 5))
    meta.create_dataset("episode_ends", data=np.asarray(ends, np.int64))
    if hasattr(store, "close"):
        store.close()
    print(f"wrote {len(ends)} episodes, {total} steps to {args.out}")


if __name__ == "__main__":
    main()
