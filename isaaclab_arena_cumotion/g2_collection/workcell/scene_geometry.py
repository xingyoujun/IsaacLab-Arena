# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep experiment-only bin scaling consistent across scene, planning and evaluation."""

import copy
import numpy as np


def scaled_bin_geometry(geometry, ratio):
    """Scale the sampled cavity under a positive axis-aligned asset scale ratio."""
    ratio = np.asarray(ratio, dtype=float)
    assert ratio.shape == (3,) and np.all(ratio > 0)
    result = copy.deepcopy(geometry)
    result["inner_xy_bounds_m"] = (np.array(geometry["inner_xy_bounds_m"]) * ratio[:2]).tolist()
    result["inner_size_xy_m"] = np.diff(np.array(result["inner_xy_bounds_m"]), axis=0)[0].tolist()
    result["floor_z_m"] *= ratio[2]
    result["rim_max_z_m"] *= ratio[2]
    for sample in result.get("wall_ray_samples", []):
        sample["height_m"] *= ratio[2]
        sample["distances_px_nx_py_ny_m"] = (np.array(sample["distances_px_nx_py_ny_m"]) * ratio[[0, 0, 1, 1]]).tolist()
    result["scale_ratio_from_geometry_input"] = ratio.tolist()
    return result
