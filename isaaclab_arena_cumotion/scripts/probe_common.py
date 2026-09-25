# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared set-up for the measurement probes: one environment, teleop off, success disabled.

Every probe wants the same thing -- a single registered environment, built headless with no
teleop device, and its ``success`` termination replaced by one that never fires, so a staged
solved state is not auto-reset out from under the measurement (a reset re-places the objects and
looks exactly like the object being thrown). This module is that boilerplate, plus the handful of
object read/write helpers the probes share. Import it only after the ``AppLauncher`` is up.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import fields
from typing import Any, get_args, get_origin, get_type_hints

ENV_ARG_HELP = (
    "Override a field of the environment's typed config, as FIELD=VALUE (repeatable). Booleans take"
    " true/false, 'none' clears an optional field."
)


def add_env_override_arg(parser: argparse.ArgumentParser) -> None:
    """Add the repeatable ``--env-arg FIELD=VALUE`` option every probe accepts."""
    parser.add_argument("--env-arg", action="append", default=[], metavar="FIELD=VALUE", help=ENV_ARG_HELP)


def parse_env_overrides(pairs: list[str], cfg_type: type) -> dict[str, Any]:
    """Turn ``FIELD=VALUE`` strings into typed keyword arguments for ``cfg_type``.

    Args:
        pairs: The raw ``--env-arg`` values.
        cfg_type: The environment's typed configuration dataclass.

    Returns:
        Keyword arguments ready for ``cfg_type(**overrides)``.
    """
    hints = get_type_hints(cfg_type)
    known = {config_field.name for config_field in fields(cfg_type)}
    overrides: dict[str, Any] = {}
    for pair in pairs:
        assert "=" in pair, f"--env-arg expects FIELD=VALUE, got {pair!r}"
        name, raw = pair.split("=", 1)
        assert name in known, f"{cfg_type.__name__} has no field {name!r}; known: {sorted(known)}"
        value_type = hints[name]
        if get_origin(value_type) is not None:  # str | None -> str
            value_type = next(t for t in get_args(value_type) if t is not type(None))
        if raw.lower() in ("none", "null"):
            overrides[name] = None
        elif value_type is bool:
            overrides[name] = raw.lower() in ("1", "true", "yes", "on")
        else:
            overrides[name] = value_type(raw)
    return overrides


def environment_cfg_type(env_name: str) -> type:
    """The typed configuration class of a registered environment (registering the package first)."""
    import isaaclab_arena_environments  # noqa: F401  (registers the environments)
    from isaaclab_arena.assets.registries import EnvironmentRegistry

    return EnvironmentRegistry().get_component_by_name(env_name)._legacy_argparse_cfg_type


def is_kinematic(env, name: str) -> bool:
    """Whether a scene object is a kinematic fixture (it will not fall or be pushed)."""
    from isaaclab_arena.terms.events import _velocity_is_writable

    return not _velocity_is_writable(env.scene[name])


def build_probe_env(
    env_name: str,
    overrides: dict[str, Any] | None = None,
    num_envs: int = 1,
    disable_success: bool = True,
    enable_cameras: bool = False,
    prepare: Callable[[Any], None] | None = None,
):
    """Build a registered environment for probing and reset it once.

    Args:
        env_name: Registry name of the environment.
        overrides: Typed-config overrides, e.g. from ``parse_env_overrides``.
        num_envs: Number of parallel environments.
        disable_success: Replace the ``success`` termination with one that never fires, so writing
            a solved state does not trigger the automatic reset.
        enable_cameras: Build with the embodiment's cameras.
        prepare: Optional hook run on the ``IsaacLabArenaEnvironment`` before it is compiled, e.g.
            to swap the embodiment's action config.

    Returns:
        ``(env, arena_env)``: the unwrapped Isaac Lab environment and Arena's description of it.
    """
    import torch

    from isaaclab.managers import TerminationTermCfg

    import isaaclab_arena_environments  # noqa: F401  (registers the environments)
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    factory = EnvironmentRegistry().get_component_by_name(env_name)()
    cfg_type = factory._legacy_argparse_cfg_type
    field_names = {config_field.name for config_field in fields(cfg_type)}
    kwargs = dict(overrides or {})
    if "teleop_device" in field_names:
        kwargs.setdefault("teleop_device", None)
    if enable_cameras and "enable_cameras" in field_names:
        kwargs["enable_cameras"] = True
    arena_env = factory.build(cfg_type(**kwargs))

    if disable_success:
        previous = arena_env.env_cfg_callback

        def never(env):
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        def callback(env_cfg):
            if previous is not None:
                env_cfg = previous(env_cfg)
            # SuccessRecorder asserts a term named ``success`` exists, so it is replaced, not removed.
            env_cfg.terminations.success = TerminationTermCfg(func=never, params={})
            return env_cfg

        arena_env.env_cfg_callback = callback

    if prepare is not None:
        prepare(arena_env)

    cli = ["--num_envs", str(num_envs)] + (["--enable_cameras"] if enable_cameras else [])
    arena_args = get_isaaclab_arena_cli_parser().parse_args(cli)
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args)).make_registered().unwrapped
    env.reset()
    return env, arena_env


def to_torch(array):
    """Return a torch view of an Isaac Lab data buffer (torch tensor or warp array)."""
    import torch

    import warp as wp

    if isinstance(array, torch.Tensor):
        return array
    if hasattr(array, "torch"):
        return array.torch
    return wp.to_torch(array)


def scene_object_names(env) -> list[str]:
    """Scene keys of every rigid object and non-robot articulation, in scene order."""
    names = list(env.scene.rigid_objects)
    names += [name for name in env.scene.articulations if name != "robot"]
    return names


def object_position(env, name: str, env_index: int = 0):
    """World position of an object's root, as a (3,) tensor."""
    return to_torch(env.scene[name].data.root_pos_w)[env_index].float()


def object_quat_xyzw(env, name: str, env_index: int = 0):
    """World orientation of an object's root, (x, y, z, w) as Isaac Lab 3.0 stores it."""
    return to_torch(env.scene[name].data.root_quat_w)[env_index].float()


def object_speed(env, name: str, env_index: int = 0) -> float:
    """Magnitude of an object's root linear velocity, in m/s."""
    return float(to_torch(env.scene[name].data.root_lin_vel_w)[env_index].norm())


def tilt_deg(quat_xyzw) -> float:
    """Angle between a body's local z axis and world up, in degrees."""
    import isaaclab.utils.math as math_utils

    rotation = math_utils.matrix_from_quat(quat_xyzw.reshape(1, 4))[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))


def write_object_pose(env, name: str, position_xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0), env_index: int = 0) -> None:
    """Teleport an object's root to a world pose, zeroing its velocity where PhysX allows it."""
    import torch

    from isaaclab_arena.terms.events import _velocity_is_writable

    asset = env.scene[name]
    ids = torch.tensor([env_index], device=env.device)
    pose = torch.cat(
        [
            torch.as_tensor(position_xyz, dtype=torch.float32, device=env.device).reshape(1, 3),
            torch.as_tensor(quat_xyzw, dtype=torch.float32, device=env.device).reshape(1, 4),
        ],
        dim=-1,
    )
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    if _velocity_is_writable(asset):
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)


def yaw_quat_xyzw(yaw_rad: float):
    """Quaternion (x, y, z, w) for a pure yaw."""
    import torch

    return torch.tensor([0.0, 0.0, math.sin(0.5 * yaw_rad), math.cos(0.5 * yaw_rad)])


def zero_action(env):
    """An all-zero action for ``env.step``; with the standard control stack that means 'hold'."""
    import torch

    return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


def settle(env, seconds: float) -> None:
    """Step the environment under zero actions for ``seconds`` of simulated time."""
    action = zero_action(env)
    for _ in range(max(1, round(seconds / env.step_dt))):
        env.step(action)
