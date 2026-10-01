# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Create a traceable settled-start derivative without modifying historical raw recordings."""

import argparse
import h5py
import hashlib
import json
from pathlib import Path

from data_engine.recording.alignment import pre_step_states


def derive(source, destination, episode, start, reason):
    """Remove only a documented initialization prefix, retaining every task and terminal transition."""
    assert reason.strip(), "A measured initialization exclusion reason is required"
    assert not destination.exists(), destination
    with h5py.File(source, "r") as original:
        demo = original[f"data/{episode}"]
        count = len(demo["actions"])
        assert 0 < start < count
        pre_step_states(demo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(destination, "x") as output:
            data = output.create_group("data")
            data.attrs.update(original["data"].attrs)
            target = data.create_group(episode)
            target.attrs.update(demo.attrs)

            def copy_group(before, after, prefix=""):
                after.attrs.update(before.attrs)
                for name, item in before.items():
                    path = f"{prefix}/{name}"
                    if isinstance(item, h5py.Group):
                        copy_group(item, after.create_group(name), path)
                    elif path.startswith("/initial_state/"):
                        state_key = path.replace("/initial_state/", "states/", 1)
                        created = after.create_dataset(name, data=demo[state_key][start - 1 : start])
                        created.attrs.update(item.attrs)
                    elif item.ndim and len(item) == count:
                        created = after.create_dataset(name, data=item[start:], compression="gzip")
                        created.attrs.update(item.attrs)
                    else:
                        before.copy(item, after, name=name)

            copy_group(demo, target)
            target.attrs["num_samples"] = count - start
            data.attrs["total"] = count - start
            metadata = json.loads(data.attrs.get("env_args", "{}"))
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            metadata["derivation"] = {
                "source_file": source.name,
                "source_sha256": digest,
                "source_episode": episode,
                "source_frame_range": [start, count],
                "source_frames": count,
                "reason": reason,
                "terminal_transition_preserved": True,
            }
            data.attrs["env_args"] = json.dumps(metadata)
            pre_step_states(target)
    print(f"Created derivative {destination}; source remains unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    derive(args.source, args.destination, args.episode, args.start, args.reason)
