# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only checks for G2 raw, parquet, RGB video and episode alignment; optionally use LeRobot's loader."""

import argparse
import json
import numpy as np
from pathlib import Path

from g2_dataset_io import CAMERAS, file_digest, read_core, sample_bowls


def check(root, work, official_loader=False):
    import imageio.v2 as imageio
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    entries = [json.loads(path.read_text()) for path in sorted((work / "episodes").glob("episode_*.json"))]
    total = 0
    for index, entry in enumerate(entries):
        assert entry["episode_index"] == index
        arrays, report = read_core(Path(entry["raw_dir"]) / "episodes.hdf5")
        assert file_digest(Path(entry["raw_dir"]) / "episodes.hdf5") == entry["raw_sha256"]
        count = len(arrays["action"])
        chunk = index // 1000
        table = pq.read_table(root / f"data/chunk-{chunk:03d}/episode_{index:06d}.parquet")
        assert len(table) == count
        for name, value in arrays.items():
            assert np.array_equal(np.array(table[name].to_pylist(), dtype=np.float32), value), name
        assert np.array_equal(table["frame_index"].to_numpy(), np.arange(count))
        assert np.array_equal(table["index"].to_numpy(), np.arange(total, total + count))
        assert np.allclose(table["timestamp"].to_numpy(), np.arange(count) / info["fps"], atol=1e-5)
        assert np.all(table["episode_index"].to_numpy() == index)
        for name, dimensions in CAMERAS.items():
            path = root / f"videos/chunk-{chunk:03d}/observation.images.{name}/episode_{index:06d}.mp4"
            reader = imageio.get_reader(str(path))
            assert reader.count_frames() == count
            assert reader.get_meta_data()["fps"] == info["fps"]
            for frame_index in (0, count // 2, count - 1):
                frame = reader.get_data(frame_index)
                assert frame.shape == (*dimensions, 3) and frame.std() > 5
            reader.close()
        assert report["configuration"]["bowl_xy_noise_m"] > 0
        total += count
    assert len(entries) == info["total_episodes"] and total == info["total_frames"]
    if official_loader:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset("local/g2_stack_bowls", root=root, video_backend="pyav")
        assert len(dataset) == total
        for index in (0, total // 2, total - 1):
            frame = dataset[index]
            assert frame["action"].shape == (16,)
            assert frame["observation.eef_pose"].shape == (14,)
            for camera in CAMERAS:
                assert frame[f"observation.images.{camera}"].shape[0] == 3
        print("OFFICIAL_LEROBOT_LOAD_OK", flush=True)
    print(f"DATASET_VALID episodes={len(entries)} frames={total} cameras=3", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--official_loader", action="store_true")
    args = parser.parse_args()
    # Pure sampler invariants across a much larger set than the physical pilot.
    layouts = [sample_bowls(seed, 0.02) for seed in range(1000)]
    assert len({tuple(layout) for layout in layouts}) == 1000
    assert sample_bowls(42, 0.02) == sample_bowls(42, 0.02)
    for layout in layouts:
        assert np.max(np.abs(np.array(layout) - [-0.12, -0.24, -0.10, 0, -0.12, 0.24])) <= 0.02
    check(args.root, args.work_dir, args.official_loader)


if __name__ == "__main__":
    main()
