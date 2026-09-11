# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cut every demonstration of an Arena HDF5 file at the last step the gripper is still commanded closed.

The cuMotion drivers finish a demo by releasing the object, retreating and returning home. For a
task whose success is reached while the gripper still holds the object (pulling a drawer open),
that tail carries no task information and roughly doubles the episode length. This writes a copy of
the file in which each demo ends at the last step whose gripper target equals the demo's maximum
(fully closed) target; every per-step dataset is sliced alike, ``num_samples`` and ``total`` are
recomputed and all attributes are preserved.

Usage::

    .venv/bin/python isaaclab_arena_cumotion/scripts/truncate_demos_at_gripper_release.py \\
        --input /path/open_drawer.hdf5 --output /path/open_drawer_trunc.hdf5 [--keep-after 0]
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import pathlib


def last_closed_step(gripper_target: np.ndarray, tolerance: float) -> int:
    """Index of the last step whose gripper target is within ``tolerance`` of the demo's maximum."""
    closed = gripper_target >= gripper_target.max() - tolerance
    assert closed.any(), "gripper never reaches its closed target"
    return int(np.flatnonzero(closed)[-1])


def truncate_demo(source: h5py.Group, destination: h5py.Group, num_samples: int, cut: int) -> None:
    """Copy ``source`` into ``destination`` keeping the first ``cut`` rows of every per-step dataset."""
    for key, value in source.attrs.items():
        destination.attrs[key] = value

    def visit(name: str, item) -> None:
        if isinstance(item, h5py.Group):
            group = destination.require_group(name)
            for key, value in item.attrs.items():
                group.attrs[key] = value
            return
        data = item[()]
        if item.ndim >= 1 and item.shape[0] == num_samples:
            data = data[:cut]
        dataset = destination.create_dataset(name, data=data, dtype=item.dtype)
        for key, value in item.attrs.items():
            dataset.attrs[key] = value

    source.visititems(visit)
    destination.attrs["num_samples"] = cut


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=pathlib.Path, required=True, help="Recorded HDF5 file.")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Truncated copy to write.")
    parser.add_argument(
        "--gripper-dataset",
        default="joint_pos_target",
        help="Per-step dataset holding the gripper target (default: the applied joint targets).",
    )
    parser.add_argument("--gripper-column", type=int, default=-1, help="Column of the gripper target.")
    parser.add_argument(
        "--tolerance", type=float, default=1e-3, help="Targets this close to the maximum count as closed (rad)."
    )
    parser.add_argument(
        "--keep-after", type=int, default=0, help="Extra steps to keep after the last closed step (if recorded)."
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    assert args.input.exists(), f"{args.input} does not exist"
    assert args.overwrite or not args.output.exists(), f"{args.output} exists (use --overwrite)"

    lengths_before, lengths_after = [], []
    with h5py.File(args.input, "r") as source, h5py.File(args.output, "w") as output:
        for key, value in source.attrs.items():
            output.attrs[key] = value
        data_out = output.create_group("data")
        for key, value in source["data"].attrs.items():
            data_out.attrs[key] = value
        names = sorted(source["data"], key=lambda name: int(name.split("_")[-1]))
        total = 0
        for name in names:
            demo = source["data"][name]
            num_samples = int(demo.attrs["num_samples"])
            gripper = np.asarray(demo[args.gripper_dataset])[:, args.gripper_column]
            assert gripper.shape[0] == num_samples, f"{name}: {args.gripper_dataset} has {gripper.shape[0]} rows"
            cut = min(last_closed_step(gripper, args.tolerance) + 1 + args.keep_after, num_samples)
            truncate_demo(demo, data_out.create_group(name), num_samples, cut)
            total += cut
            lengths_before.append(num_samples)
            lengths_after.append(cut)
        data_out.attrs["total"] = total

    before, after = np.array(lengths_before), np.array(lengths_after)
    print(f"{len(names)} demos: {before.sum()} -> {after.sum()} steps")
    print(f"  length before: min {before.min()} mean {before.mean():.1f} max {before.max()}")
    print(f"  length after:  min {after.min()} mean {after.mean():.1f} max {after.max()}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
