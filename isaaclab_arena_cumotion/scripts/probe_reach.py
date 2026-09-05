# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Which arm can reach which point, with how much lean? The IK reach scan for laying out a task.

Before an object is placed, a container positioned or a handover point chosen, ask the kinematics
whether the tool can get there: for every point, arm and lean angle the scan counts how many
headings around the vertical (times jaw spins) have an exact IK solution. The Agibot's two arms
are not mirror images, the reach band at table height is narrow (x 0.35-0.45 with a lean), pure
top-down approaches are often unreachable at the table, and the far side of the table is thin --
all of which is cheaper to learn here than halfway through a teleop session or a cuMotion run.

Points come from ``--point x,y,z`` (world frame) or from the current positions of scene objects
(``--object NAME`` plus ``--dz``); with neither, every object in the scene is scanned.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_reach.py --headless --env agibot_tidy_workbench \\
        --point 0.52,0.30,0.83 --point 0.33,-0.18,0.65 --tilts 0,30,50,75

Read the table as "poses reachable out of headings x spins": a dozen or more at a lean of 30-50
is comfortable for a grasp; a handful only at 75 means the spot is a stretch a human can make and
a planner needs a dedicated grasp family for; zero at every lean means move the object.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--arms", type=str, default="left,right", help="Comma-separated arms to scan.")
parser.add_argument("--point", action="append", default=[], metavar="X,Y,Z", help="World point to test (repeatable).")
parser.add_argument("--object", action="append", default=[], help="Scene object whose position is tested (repeatable).")
parser.add_argument("--dz", type=float, default=0.03, help="Height above an object's origin for the tool point.")
parser.add_argument("--tilts", type=str, default="0,30,50,75", help="Lean angles from straight down, degrees.")
parser.add_argument("--headings", type=int, default=24, help="Headings around the vertical per tilt.")
parser.add_argument("--spins", type=str, default="90,-90", help="Jaw spins about the approach axis, degrees.")
parser.add_argument("--horizontal", action="store_true", help="Also scan horizontal (side) approaches.")
from isaaclab_arena_cumotion.scripts.probe_common import add_env_override_arg  # noqa: E402

add_env_override_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402

from isaaclab_arena_cumotion.grasps import DOWN_FACING_ROTATION, quat_wxyz_from_matrix  # noqa: E402
from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402
from isaaclab_arena_cumotion.scripts.probe_common import (  # noqa: E402
    build_probe_env,
    environment_cfg_type,
    object_position,
    parse_env_overrides,
    scene_object_names,
    settle,
)

cfg_type = environment_cfg_type(args.env)
env, arena_env = build_probe_env(args.env, parse_env_overrides(args.env_arg, cfg_type))
settle(env, 0.5)

arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
planners = {arm: CumotionArmPlanner(env, arena_env.embodiment, arm=arm) for arm in arms}
tilts = [float(value) for value in args.tilts.split(",")]
spins = [float(value) for value in args.spins.split(",")]

points: list[tuple[str, np.ndarray]] = []
for raw in args.point:
    x, y, z = (float(value) for value in raw.split(","))
    points.append((f"({x:.2f}, {y:.2f}, {z:.3f})", np.array([x, y, z])))
for name in args.object or ([] if args.point else scene_object_names(env)):
    position = object_position(env, name).cpu().numpy() + np.array([0.0, 0.0, args.dz])
    points.append((f"{name} +{args.dz * 1000:.0f} mm", position))
assert points, "nothing to scan: give --point or --object, or run on an env with objects"


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def leaned_poses(point: np.ndarray, tilt_deg: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Top-down approach leaned by ``tilt_deg``, over all headings and jaw spins."""
    poses = []
    for heading in np.linspace(0.0, 2.0 * np.pi, args.headings, endpoint=False):
        for spin in spins:
            rotation = rot_z(heading) @ rot_y(np.radians(tilt_deg)) @ DOWN_FACING_ROTATION @ rot_z(np.radians(spin))
            poses.append((point, quat_wxyz_from_matrix(rotation)))
    return poses


def horizontal_poses(point: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Approach horizontal (tool z along a heading), jaws closing horizontally, either roll."""
    poses = []
    up = np.array([0.0, 0.0, 1.0])
    for heading in np.linspace(0.0, 2.0 * np.pi, args.headings, endpoint=False):
        approach = np.array([np.cos(heading), np.sin(heading), 0.0])
        for sign in (1.0, -1.0):
            x_axis = sign * up
            rotation = np.stack([x_axis, np.cross(approach, x_axis), approach], axis=1)
            poses.append((point, quat_wxyz_from_matrix(rotation)))
    return poses


def count(arm: str, poses) -> int:
    return sum(int(planners[arm].ik_reachable(position, quat)) for position, quat in poses)


per_tilt_total = args.headings * len(spins)
header = " ".join(f"tilt{tilt:g}" for tilt in tilts) + ("  side" if args.horizontal else "")
print(
    f"\n[reach] poses reachable out of {per_tilt_total} per lean"
    + (f" ({2 * args.headings} side)" if args.horizontal else "")
)
print(f"[reach] {'point':34s} {'arm':5s} {header}")
for label, point in points:
    for arm in arms:
        counts = [count(arm, leaned_poses(point, tilt)) for tilt in tilts]
        row = " ".join(f"{value:6d}" for value in counts)
        if args.horizontal:
            row += f" {count(arm, horizontal_poses(point)):5d}"
        print(f"[reach] {label:34s} {arm:5s} {row}", flush=True)

env.close()
simulation_app.close()
