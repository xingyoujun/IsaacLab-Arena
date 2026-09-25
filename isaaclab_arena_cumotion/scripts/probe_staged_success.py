# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Does the success predicate accept a solved scene? Validate a task's success check without teleop.

The objects are written straight into the goal poses given on the command line and the task's
``is_success`` is read: once immediately after the write (what the predicate says about the
geometry alone) and once more after physics has run (whether the staged state survives gravity
and settling, or whether the goal pose itself is not a resting state). Doing this before the
first teleop session catches inverted joint polarities, thresholds that a resting body's velocity
noise can never satisfy, and goal poses that were placed by the wrong origin convention -- each of
which otherwise surfaces as "I did the task and it never said success" after a long session.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_staged_success.py --env agibot_sleeve_on_peg \\
        --pose peg_sleeve=0.40,0.00,0.6432 --physics_seconds 2

Poses are ``name=x,y,z`` or ``name=x,y,z,yaw_deg`` in the world frame; objects not named keep the
poses the reset gave them. The environment's success termination is disabled for the run, so the
staged state is not reset away the instant it counts as solved.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument(
    "--pose", action="append", default=[], metavar="NAME=X,Y,Z[,YAW_DEG]", help="Goal pose (repeatable)."
)
parser.add_argument("--settle_seconds", type=float, default=0.5, help="Settling time after the reset, before staging.")
parser.add_argument("--physics_seconds", type=float, default=2.0, help="Physics run after staging; 0 skips.")
from isaaclab_arena_cumotion.scripts.probe_common import add_env_override_arg  # noqa: E402

add_env_override_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
simulation_app = AppLauncher(args).app

import math  # noqa: E402

from isaaclab_arena_cumotion.scripts.probe_common import (  # noqa: E402
    build_probe_env,
    environment_cfg_type,
    object_position,
    object_quat_xyzw,
    object_speed,
    parse_env_overrides,
    settle,
    tilt_deg,
    write_object_pose,
    yaw_quat_xyzw,
)

cfg_type = environment_cfg_type(args.env)
env, arena_env = build_probe_env(args.env, parse_env_overrides(args.env_arg, cfg_type))
task = arena_env.task
assert task is not None, f"{args.env} has no task to validate"

staged = []
for raw in args.pose:
    name, values = raw.split("=", 1)
    numbers = [float(value) for value in values.split(",")]
    assert len(numbers) in (3, 4), f"--pose wants NAME=X,Y,Z[,YAW_DEG], got {raw!r}"
    yaw = math.radians(numbers[3]) if len(numbers) == 4 else 0.0
    staged.append((name, numbers[:3], yaw))
assert staged, "give at least one --pose"


def report(stage: str) -> None:
    success = bool(task.is_success(env)[0])
    print(f"\n[staged] {stage}: success = {success}")
    for name, _, _ in staged:
        position = object_position(env, name)
        print(
            f"[staged]   {name:22s} at ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.4f})"
            f"  tilt {tilt_deg(object_quat_xyzw(env, name)):5.1f} deg  speed {object_speed(env, name):.3f} m/s"
        )


settle(env, args.settle_seconds)
report("before staging")

for name, position, yaw in staged:
    write_object_pose(env, name, position, yaw_quat_xyzw(yaw))
# Read the predicate on the written state alone: a single env.step lets gravity act first.
env.scene.write_data_to_sim()
env.scene.update(env.step_dt)
report("staged, no physics")

if args.physics_seconds > 0:
    settle(env, args.physics_seconds)
    report(f"after {args.physics_seconds:g} s of physics")

env.close()
simulation_app.close()
