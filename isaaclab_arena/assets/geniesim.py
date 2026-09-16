# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Location of the Genie Sim asset library (HuggingFace dataset ``agibot-world/GenieSimAssets``).

The library is distributed out of band (CC BY-NC-SA 4.0) and is not part of the Arena repository. Download the
folders you need to ``$HOME/datasets/GenieSimAssets`` on the host; the Arena container mounts that directory at
``/datasets/GenieSimAssets``. Override the location with the ``GENIESIM_ASSETS_DIR`` environment variable.
"""

import os

GENIESIM_ASSETS_DIR: str = os.environ.get("GENIESIM_ASSETS_DIR", "/datasets/GenieSimAssets")
"""Root of the GenieSimAssets checkout."""
