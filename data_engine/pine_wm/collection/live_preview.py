# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record actual GPU sensor frames during a preview, without state replay."""

import json
import numpy as np
import os

import imageio.v2 as imageio


class LivePreview:
    def __init__(self, env, output):
        self.env = env
        self.output = output / "live_preview"
        self.output.mkdir(exist_ok=True)
        self.writers = {}
        self.fps = 1.0 / float(env.step_dt)

    def capture(self):
        """Capture one post-step review frame; the caller owns the only step observer."""
        if self.writers:
            for name, writer in self.writers.items():
                tensor = self.env.scene[name].data.output["rgb"][0, ..., :3]
                self.devices[name] = str(tensor.device)
                assert tensor.is_cuda, f"Preview sensor {name} must use CUDA"
                pixels = tensor.cpu().numpy().astype(np.uint8)
                writer.append_data(pixels)
                if name not in self.first:
                    self.first[name] = pixels.astype(float)
                self.motion[name] = max(self.motion[name], float(np.abs(pixels - self.first[name]).mean()))
                self.contrast[name] = max(self.contrast[name], float(pixels.std()))
            self.frames += 1

    def begin(self, trial):
        self.trial = trial
        self.frames = 0
        self.first = {}
        self.devices = {}
        names = ["realsense_d435", "wrist_a", "wrist_b", "scene_cam"]
        self.motion = dict.fromkeys(names, 0.0)
        self.contrast = dict.fromkeys(names, 0.0)
        # Prefer the system NVENC build; imageio-ffmpeg's bundled fallback has no NVENC encoder.
        codec = "libx264"
        if os.path.exists("/usr/bin/ffmpeg"):
            os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
            codec = "h264_nvenc"
        self.writers = {
            n: imageio.get_writer(
                self.output / f"trial_{trial}_{n}.part.mp4", fps=self.fps, codec=codec, macro_block_size=8
            )
            for n in names
        }

    def finish(self, success):
        names = list(self.writers)
        self.abort()
        valid = bool(
            success
            and self.frames > 1
            and all(v > 3 for v in self.contrast.values())
            and all(v > 1 for v in self.motion.values())
        )
        files = {}
        if valid:
            for n in names:
                name = f"trial_{self.trial}_{n}.mp4"
                (self.output / f"trial_{self.trial}_{n}.part.mp4").rename(self.output / name)
                files[n] = name
        manifest = {
            "validation_passed": valid,
            "task_success": bool(success),
            "mode": "live_sensor_post_step_preview",
            "state_replay": False,
            "frames": self.frames,
            "fps": self.fps,
            "control_dt_s": float(self.env.step_dt),
            "camera_devices": self.devices,
            "files": files,
            "image_motion": self.motion,
            "image_contrast": self.contrast,
            "visual_review_required": True,
        }
        (self.output / f"trial_{self.trial}.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    def abort(self):
        for writer in self.writers.values():
            writer.close()
        self.writers = {}
