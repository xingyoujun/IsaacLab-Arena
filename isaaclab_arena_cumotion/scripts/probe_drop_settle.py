# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Where does each object actually come to rest, and how noisy is 'at rest'? The asset intake probe.

An asset's origin is wherever its author put it -- the bottom face, the geometry centre, a shaft
frame 50 mm off the part -- and nothing in the USD tells you reliably which. The same goes for the
collider PhysX actually simulates (thin meshes get inflated) and for the velocity a resting body
reports (a stacked or cradled body never reads zero). All three decide placement heights, grasp
aiming and success thresholds, and all three are measured here by letting physics answer: every
object is read after the scene settles, then dropped from a small height and read again.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/probe_drop_settle.py --env agibot_tidy_workbench

Read ``origin above surface`` as the origin convention (0 mm = bottom-origin, half the height =
centre-origin, anything odd = re-centre the asset before placing it by origin), ``tilt`` as whether
the part stands the way it was placed, and ``speed`` as the at-rest noise floor a success
predicate's rest threshold has to clear by a safe factor.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--surface_z", type=float, default=None, help="Work-surface height; default is the Agibot table.")
parser.add_argument("--settle_seconds", type=float, default=1.5)
parser.add_argument(
    "--drop_height", type=float, default=0.05, help="Lift each object by this and let it fall; 0 skips."
)
parser.add_argument("--objects", type=str, default=None, help="Comma-separated scene keys; default all objects.")
from isaaclab_arena_cumotion.scripts.probe_common import add_env_override_arg  # noqa: E402

add_env_override_arg(parser)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
simulation_app = AppLauncher(args).app

from isaaclab_arena_cumotion.scripts.probe_common import (  # noqa: E402
    build_probe_env,
    environment_cfg_type,
    is_kinematic,
    object_position,
    object_quat_xyzw,
    object_speed,
    parse_env_overrides,
    scene_object_names,
    settle,
    tilt_deg,
    write_object_pose,
)
from isaaclab_arena_environments.agibot_tabletop_common import TABLE_TOP_Z  # noqa: E402

surface_z = TABLE_TOP_Z if args.surface_z is None else args.surface_z
cfg_type = environment_cfg_type(args.env)
env, arena_env = build_probe_env(args.env, parse_env_overrides(args.env_arg, cfg_type))
names = [name.strip() for name in args.objects.split(",")] if args.objects else scene_object_names(env)


def report(stage: str, name: str, reference=None) -> None:
    position = object_position(env, name)
    line = (
        f"[settle] {stage:10s} {name:22s} origin above surface {(float(position[2]) - surface_z) * 1000:+7.1f} mm"
        f"  tilt {tilt_deg(object_quat_xyzw(env, name)):5.1f} deg  speed {object_speed(env, name):.3f} m/s"
    )
    if reference is not None:
        line += f"  moved xy {float((position[:2] - reference[:2]).norm()) * 1000:.1f} mm"
    if is_kinematic(env, name):
        line += "  (kinematic fixture: does not fall)"
    print(line, flush=True)


print(f"[settle] surface at z = {surface_z:.4f}; objects: {names}")
settle(env, args.settle_seconds)
print(f"\n[settle] after reset + {args.settle_seconds:g} s")
for name in names:
    report("as spawned", name)

if args.drop_height > 0:
    print(
        f"\n[settle] each object lifted {args.drop_height * 1000:.0f} mm and dropped, {args.settle_seconds:g} s to"
        " settle"
    )
    for name in names:
        before = object_position(env, name).clone()
        lifted = before.clone()
        lifted[2] += args.drop_height
        write_object_pose(env, name, lifted, object_quat_xyzw(env, name))
        settle(env, args.settle_seconds)
        report("dropped", name, reference=before)

env.close()
simulation_app.close()
