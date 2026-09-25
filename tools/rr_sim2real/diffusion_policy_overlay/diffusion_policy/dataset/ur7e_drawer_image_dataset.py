# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""UR7e open-drawer demonstrations (Isaac Lab-Arena, rr_sim2real) as a diffusion_policy image dataset.

The zarr is produced by IsaacLab-Arena's ``lerobot_to_diffusion_policy_zarr.py`` and already has the
replay-buffer layout (``data/*`` concatenated over time, ``meta/episode_ends``), so no video decoding
or conversion happens here: the store is copied into memory, still Jpeg2k-compressed, and frames
are decoded on access.

Arrays:
    camera_0        (T, 240, 320, 3) uint8   D435 colour stream (640x480 in the recording, resized)
    robot_eef_pose  (T, 9)  float32          TCP xyz + rotation_6d (pytorch3d rows), UR base frame
    gripper_pos     (T, 1)  float32          finger_joint / (pi/4): 0 open .. ~0.72 closed on the knob
    robot_joint     (T, 6)  float32          arm joints (rad), optional low_dim observation
    action          (T, 10) float32          target TCP pose (9, as above) + gripper, absolute
"""

import copy
import numpy as np
import torch

import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from threadpoolctl import threadpool_limits

register_codecs()


class Ur7eDrawerImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        rgb_keys = list()
        lowdim_keys = list()
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        # In-memory copy keeps the on-disk compressors (Jpeg2k for images), ~1.5 GB for 200 episodes.
        replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, store=zarr.MemoryStore(), keys=rgb_keys + lowdim_keys + ["action"]
        )

        for key in rgb_keys:
            c, h, w = shape_meta["obs"][key]["shape"]
            assert replay_buffer[key].shape[1:] == (
                h,
                w,
                c,
            ), f"{key}: zarr frames are {replay_buffer[key].shape[1:]}, shape_meta expects {(h, w, c)}"
        for key in lowdim_keys:
            assert replay_buffer[key].shape[1:] == tuple(shape_meta["obs"][key]["shape"]), key
        assert replay_buffer["action"].shape[1:] == tuple(shape_meta["action"]["shape"])

        key_first_k = dict()
        if n_obs_steps is not None:
            # only the first n_obs_steps observations of a sequence are used
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # min/max to [-1, 1] on every action and low_dim channel (position, rotation_6d and gripper alike)
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer["action"])
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[key])
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # T,H,W,C uint8 -> T,C,H,W float in [0, 1]
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps :]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
