# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observe Arena control boundaries without replacing its action manager or recorder hooks."""


class ControlStepObserver:
    """Commit labels after env.step returns, and distinguish capture failure from incomplete physics."""

    def __init__(self, step, trace, capture=None):
        self.step = step
        self.trace = trace
        self.capture = capture
        self.busy = False

    def __call__(self, actions):
        assert not self.busy, "Reentrant control step would corrupt transition indexing"
        self.busy = True
        trace = self.trace()
        if trace is not None and trace.finished:
            trace = None
        try:
            if trace is not None:
                trace.before_step()
            try:
                result = self.step(actions)
            except BaseException as error:
                if trace is not None:
                    trace.abort_step(str(error))
                raise
            if trace is not None:
                trace.after_step()
            try:
                if self.capture is not None:
                    self.capture()
            except BaseException as error:
                if trace is not None:
                    trace.event("capture_failed", {"reason": str(error), "transition_index": trace.step - 1})
                raise
            return result
        finally:
            self.busy = False
