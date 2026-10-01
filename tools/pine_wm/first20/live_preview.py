# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entrypoint; implementation lives in data_engine.pine_wm.collection.live_preview."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

if __name__ == "__main__":
    import runpy

    runpy.run_module("data_engine.pine_wm.collection.live_preview", run_name="__main__")
else:
    from importlib import import_module

    sys.modules[__name__] = import_module("data_engine.pine_wm.collection.live_preview")
