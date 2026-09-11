# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect uniformly width-scaled Articraft with the shared randomized drawer pipeline."""

from collect_ur7e_open_drawer_gpt56 import main

if __name__ == "__main__":
    main(
        environment="ur7e_open_drawer_articraft",
        task="open_drawer_articraft_v2",
        drawer_key="drawer_articraft",
        open_sign=1,
        min_open_m=0.08415,
    )
