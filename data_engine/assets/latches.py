# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Conditional joint locks with explicit reset and measured release events."""


def next_engaged(rule, engaged, held_position, release_position):
    """Evaluate one latch without executing annotation text or modifying joint state."""
    import math

    assert rule["release_when"] == "release_joint >= release_at"
    assert math.isfinite(held_position) and math.isfinite(release_position)
    if release_position >= rule["release_at"]:
        return False
    return engaged or abs(held_position - rule["hold"]) <= rule["engage_tolerance"]


class LatchController:
    """Apply joint-limit locks; released joints retain their original drives and limits."""

    def __init__(self, asset, rules):
        self.asset = asset
        self.rules = rules
        self.original_limits = asset.data.joint_pos_limits.torch.clone()
        self.original_default_positions = asset.data.default_joint_pos.torch.clone()
        self.ids = [
            (asset.joint_names.index(r["joint"]["name"]), asset.joint_names.index(r["release_joint"]["name"]))
            for r in rules
        ]
        self.engaged = [[r["initially_engaged"] for r in rules] for _ in range(asset.num_instances)]
        self.events = []
        self.physics_step = 0
        self.apply()

    def apply(self):
        """Write the lock limits without teleporting positions or changing authored springs."""
        limits = self.original_limits.clone()
        for env_id, states in enumerate(self.engaged):
            for rule, (held_id, _), engaged in zip(self.rules, self.ids, states):
                if engaged:
                    limits[env_id, held_id, :] = rule["hold"]
        self.asset.write_joint_position_limit_to_sim_index(limits=limits, warn_limit_violation=False)
        # Isaac Lab clamps reset defaults when limits tighten. A temporary lock must
        # not turn a toaster's next reset into a pressed, already-latched state.
        self.asset.data.default_joint_pos.torch.copy_(self.original_default_positions)

    def reset(self, env_ids=None):
        """Restore initial logical state for only the environments being reset."""
        import torch

        indices = list(range(len(self.engaged))) if env_ids is None else [int(i) for i in env_ids]
        # Reset is the only place positions are written. Runtime locking and
        # release never replace the actual contact-driven state.
        self.asset.write_joint_position_to_sim_index(position=self.original_default_positions[indices], env_ids=indices)
        self.asset.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(self.original_default_positions[indices]), env_ids=indices
        )
        for index in indices:
            self.engaged[int(index)] = [rule["initially_engaged"] for rule in self.rules]
        self.apply()
        self.events.append(dict(step=self.physics_step, event="reset", env_ids=[int(i) for i in indices]))

    def update(self):
        """Sample fresh body joint data once per physics step and record every transition."""
        positions = self.asset.data.joint_pos.torch.detach().cpu().numpy()
        changed = False
        for env_id, states in enumerate(self.engaged):
            for index, (rule, (held_id, release_id)) in enumerate(zip(self.rules, self.ids)):
                held, release = float(positions[env_id, held_id]), float(positions[env_id, release_id])
                state = next_engaged(rule, states[index], held, release)
                if state != states[index]:
                    self.events.append(
                        dict(
                            step=self.physics_step,
                            env_id=env_id,
                            latch=rule["name"],
                            event="engaged" if state else "released",
                            held=held,
                            release=release,
                        )
                    )
                    states[index] = state
                    changed = True
        if changed:
            self.apply()
        self.physics_step += 1


def reset_latches(env, env_ids, asset_name, rules):
    """Install an Arena physics-step hook on first reset, then reset per-environment latch state."""
    controllers = getattr(env, "interaction_latches", None)
    if controllers is None:
        controllers = env.interaction_latches = {}
        # Run after scene.update, so joint data belongs to the just-completed physics step.
        original_update = env.scene.update

        def update(dt):
            original_update(dt)
            for controller in controllers.values():
                controller.update()

        env.scene.update = update
    if asset_name not in controllers:
        controllers[asset_name] = LatchController(env.scene[asset_name], rules)
    controllers[asset_name].reset(env_ids)
