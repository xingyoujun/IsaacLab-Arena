# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixed reviewed layouts; randomized coverage requires a separate qualification phase."""


def sample_layout(task, names, bounds, rng, workspace, stack_count=5):
    """Use the fixed central review draft; broad randomization awaits user approval."""
    from review_layout import sample_review_layout

    return sample_review_layout(task, names, bounds, stack_count)
