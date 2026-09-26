# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record the native G2 head and wrist cameras with NVIDIA hardware encoding."""

import os
from contextlib import ExitStack

CAMERA_NAMES = ("head_camera", "left_wrist_camera", "right_wrist_camera")


class ThreeViewWriter:
    """Export synchronized native RGB videos; frame transfer and file I/O remain host operations."""

    def __init__(self, directory, fps):
        import imageio.v2 as imageio

        os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
        self.stack = ExitStack()
        self.writers = {
            name: self.stack.enter_context(
                imageio.get_writer(
                    str(directory / f"{name}.mp4"),
                    fps=fps,
                    codec="h264_nvenc",
                    macro_block_size=1,
                    quality=None,
                    output_params=["-preset", "p4", "-cq", "20", "-movflags", "+faststart"],
                )
            )
            for name in CAMERA_NAMES
        }
        self.frames = 0
        self.devices = {}

    def append(self, scene):
        """Read all three GPU RGB buffers at the same simulation state."""
        for name, writer in self.writers.items():
            rgb = scene[name].data.output["rgb"].torch[0, ..., :3]
            assert rgb.is_cuda, f"Camera {name} is not on CUDA"
            self.devices[name] = str(rgb.device)
            writer.append_data(rgb.cpu().numpy())
        self.frames += 1

    def close(self):
        """Finish all video files."""
        self.stack.close()
