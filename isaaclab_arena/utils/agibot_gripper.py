# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""How the Agibot's grippers are driven, shared by every path that closes them.

Lives under ``utils`` rather than ``embodiments/agibot`` because ``isaaclab_arena.embodiments``
imports every robot (and with them Isaac Lab and pxr) on package import; the environment
registry reads this constant before the simulation app exists, and a Kit-less pxr loaded that
early breaks the app start (Phase 1 fails in ``simulation_app.py``). Keep this module free of
Isaac Sim imports.
"""

AGIBOT_GRIPPER_RAMP_SECONDS = 0.5
"""Time a gripper open/close command is ramped over before the target is held constant.

One value for teleoperation (``RampedBinaryJointPositionAction``), recording and the cuMotion
executor, so a human close and a scripted close load the object identically. Stepping the target
in one control step drives the Agibot's fingers at ~0.9 m/s and ejected a thin-walled sleeve in
5/5 grasps; every ramp measured since holds it. Measured with ``probe_pinch`` (3 repeats each,
mid-air pinch then a shoulder swing), close peak = object speed while the fingers close:

    ramp     billet right / left     sleeve right
    1.67 s   HELD 0.53 / 0.53 m/s    HELD 0.34 m/s   (the original 200 steps at 1/120 s)
    0.50 s   HELD 0.55 / 0.53 m/s    HELD 0.41 m/s   (this value, 2026-09-06)
    0.30 s   HELD 0.54 / 0.54 m/s    HELD 0.34 m/s

The Franka in Arena closes in 0.20 s (one-step target, drive-limited); 0.5 s keeps a margin
over the ejection regime while cutting the Agibot's close from 1.5 s to under 0.5 s."""
