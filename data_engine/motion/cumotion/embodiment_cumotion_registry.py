# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry of cuMotion descriptions keyed by embodiment name.

The description is a property of the robot hardware, so entries are registered under the
robot-family base name (e.g. ``"agibot"``) and inherited by concrete variants.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from data_engine.motion.cumotion.cumotion_embodiment_cfg import CumotionEmbodimentCfg

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

_CUMOTION_EMBODIMENT_CFGS: dict[str, CumotionEmbodimentCfg] = {}
_CUMOTION_CFG_FACTORIES: dict[str, Callable] = {}


def register_cumotion_cfg(embodiment_name: str, cfg: CumotionEmbodimentCfg, arm: str = "left") -> None:
    """Register a cuMotion config for one arm of an embodiment, erroring on a duplicate key.

    A bimanual robot needs one entry per arm: cuMotion plans a single kinematic chain, and the
    arm that is not being planned is pinned at its default configuration.
    """
    key = f"{embodiment_name}:{arm}"
    assert (
        key not in _CUMOTION_EMBODIMENT_CFGS and key not in _CUMOTION_CFG_FACTORIES
    ), f"A cuMotion config is already registered for '{key}'."
    _CUMOTION_EMBODIMENT_CFGS[key] = cfg


def get_cumotion_cfg_by_name(embodiment_name: str, arm: str = "left", env=None) -> CumotionEmbodimentCfg:
    """Return the cuMotion config registered for an exact embodiment name and arm."""
    factory = _CUMOTION_CFG_FACTORIES.get(f"{embodiment_name}:{arm}")
    if factory is not None:
        return factory(env, arm)
    cfg = _CUMOTION_EMBODIMENT_CFGS.get(f"{embodiment_name}:{arm}")
    assert cfg is not None, (
        f"No cuMotion config registered for '{embodiment_name}:{arm}'. Register one via"
        f" register_cumotion_cfg(...). Known: {sorted(_CUMOTION_EMBODIMENT_CFGS)}."
    )
    return cfg


def get_embodiment_cumotion_cfg(embodiment: EmbodimentBase, arm: str = "left", env=None) -> CumotionEmbodimentCfg:
    """Return the cuMotion config registered for an embodiment's robot family.

    Walks the class hierarchy so a config registered under a family name (e.g. ``"agibot"``) also
    covers subclassed variants that override ``name``. The most-derived match wins, so a variant
    may register its own override.

    Args:
        embodiment: Embodiment whose robot family should be looked up.
    """
    for cls in type(embodiment).__mro__:
        name = cls.__dict__.get("name")
        factory = _CUMOTION_CFG_FACTORIES.get(f"{name}:{arm}")
        if factory is not None:
            return factory(env, arm)
        cfg = _CUMOTION_EMBODIMENT_CFGS.get(f"{name}:{arm}") if name else None
        if cfg is not None:
            return cfg
    raise AssertionError(
        f"No cuMotion config registered for embodiment '{embodiment.name}' arm '{arm}', or its"
        f" robot family. Known: {sorted(_CUMOTION_EMBODIMENT_CFGS)}."
    )


def register_cumotion_factory(embodiment_name: str, factory: Callable, arm: str = "left") -> None:
    """Register a lazy hardware profile factory taking (env, arm)."""
    key = f"{embodiment_name}:{arm}"
    assert key not in _CUMOTION_EMBODIMENT_CFGS and key not in _CUMOTION_CFG_FACTORIES, f"Duplicate profile: {key}"
    _CUMOTION_CFG_FACTORIES[key] = factory


def _agibot_cfg(env, arm):
    from data_engine.motion.embodiments.agibot_legacy import AGIBOT_LEFT_ARM_CUMOTION_CFG, AGIBOT_RIGHT_ARM_CUMOTION_CFG

    return AGIBOT_LEFT_ARM_CUMOTION_CFG if arm == "left" else AGIBOT_RIGHT_ARM_CUMOTION_CFG


def _ur7e_cfg(env, arm):
    from data_engine.motion.embodiments.ur7e_robotiq import create_ur7e_robotiq_cfg

    return create_ur7e_robotiq_cfg()


def _pine_cfg(env, arm):
    from data_engine.motion.embodiments.ur7e_robotiq import create_pine_wm_cfg

    return create_pine_wm_cfg()


def _g2_cfg(env, arm):
    from data_engine.motion.embodiments.g2 import create_g2_cumotion_cfg

    return create_g2_cumotion_cfg(env, arm)


register_cumotion_factory("agibot", _agibot_cfg, arm="left")
register_cumotion_factory("agibot", _agibot_cfg, arm="right")
register_cumotion_factory("ur7e_robotiq", _ur7e_cfg)
register_cumotion_factory("pine_wm_ur7e", _pine_cfg)
register_cumotion_factory("g2", _g2_cfg, arm="left")
register_cumotion_factory("g2", _g2_cfg, arm="right")
