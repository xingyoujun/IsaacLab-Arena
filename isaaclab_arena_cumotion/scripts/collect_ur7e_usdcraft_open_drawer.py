# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect randomized USDcraft drawer openings using the shared success-only pipeline."""

from collect_ur7e_open_drawer_gpt56 import main

if __name__ == "__main__":
    main(
        environment="ur7e_usdcraft_open_drawer",
        task="usdcraft_open_drawer",
        drawer_key="drawer_rr",
        open_sign=1,
        min_open_m=0.075,
    )
