# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Each G2-Eval environment builds from the locked pack, draws stable layouts, and judges a successful end a success.

Needs the locked pack (``python -m isaaclab_arena.g2_eval.pack download``, or ``G2_EVAL_ASSETS_DIR``).
"""

import pytest

from isaaclab_arena.g2_eval import pack
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

EPISODES = 3
SETTLE_STEPS = 30
SETTLED_M = 0.01
"""An object moves less than this while the drawn layout settles, and while the successful end settles."""
START_TILT_MAX_DEG = 10.0
END_SETTLE_STEPS = 150
"""Control steps (10 s) a successful end may take to come to rest and be judged a success."""


def _pack_present() -> bool:
    manifest = pack.pack_root() / "manifest.json"
    return manifest.is_file() and pack.digest(manifest) == pack.lock()["manifest_sha256"]


requires_pack = pytest.mark.skipif(not _pack_present(), reason="the locked G2-Eval pack is not downloaded")


def _test_g2_eval_environments(simulation_app, names=None):
    import json
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    import warp as wp

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.g2_eval.success import evaluate
    from isaaclab_arena.g2_eval.task import object_poses, object_states, relations_hold
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    arm_joints = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)], [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
    for name in names or pack.lock()["tasks"]:
        loaded = pack.load_task(name)
        task = loaded["task"]
        objects = list(task["objects"])
        randomization = task["randomization"]
        factory = EnvironmentRegistry().get_component_by_name(name)
        description = factory().build(factory._legacy_argparse_cfg_type())
        args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
        env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
        try:
            base = env.unwrapped
            robot = base.scene["robot"]

            def hold():
                """Joint-position action holding both arms where they are, grippers open."""
                q = wp.to_torch(robot.data.joint_pos)[0]
                right, left = ([q[robot.joint_names.index(j)] for j in side] for side in arm_joints)
                one = torch.ones(1, device=base.device)
                return torch.cat([torch.stack(right), one, torch.stack(left), one])[None]

            starts = []
            for _ in range(EPISODES):
                env.reset()
                start = object_poses(base, objects)[0]
                for obj, T in start.items():
                    rest = np.asarray(task["objects"][obj]["rest_pose"]["position_m"][:2])
                    reach = randomization.get("xy_m", 0.0)
                    assert np.all(np.abs(T[:2, 3] - rest) <= reach + 1e-4), (name, obj, T)
                for _ in range(SETTLE_STEPS):
                    env.step(hold())
                settled = object_poses(base, objects)[0]
                for obj in objects:
                    moved = float(np.linalg.norm(settled[obj][:3, 3] - start[obj][:3, 3]))
                    # Tipping is a turn away from how it started (it may rest tilted: an orange on its flat side).
                    tipped = float(np.degrees(np.arccos(np.clip(settled[obj][:3, 2] @ start[obj][:3, 2], -1.0, 1.0))))
                    print(f"{name} {obj}: start {start[obj][:3, 3].round(4).tolist()}, settled {moved:.4f} m")
                    assert moved < SETTLED_M, f"{name}: {obj} moved {moved:.3f} m while settling"
                    assert tipped < START_TILT_MAX_DEG, f"{name}: {obj} tipped over by {tipped:.1f} deg"
                assert not relations_hold(base, name)[0], f"{name} succeeds at its start"
                starts.append(np.r_[[start[obj][:2, 3] for obj in objects]])
            if randomization.get("mode") != "fixed":
                assert any(not np.allclose(starts[0], other) for other in starts[1:]), f"{name}: layouts all the same"

            # A successful end (the pack's provenance: the approved pilot's or a collected episode's last state, or one
            # constructed in the bins), written into the simulation and left to settle; the physics only, so that the
            # success termination does not reset the episode.
            provenance = json.loads((loaded["root"] / "provenance.json").read_text())
            origins = base.scene.env_origins
            origin = (wp.to_torch(origins) if isinstance(origins, wp.array) else origins)[0]
            index = torch.tensor([0], device=base.device)
            for obj, T in provenance["success_check"]["poses"]["end"].items():
                T = np.asarray(T, float)
                pose = torch.tensor(
                    np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()], dtype=torch.float32, device=base.device
                )
                pose[:3] += origin[:3].to(pose.dtype)
                base.scene[obj].write_root_pose_to_sim_index(root_pose=pose[None], env_ids=index)
                base.scene[obj].write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((1, 6), device=base.device), env_ids=index
                )
            # Settle until the relations hold (an object put in a bin rocks to rest), at most END_SETTLE_STEPS.
            written, held_at = None, None
            for step in range(END_SETTLE_STEPS):
                for _ in range(base.cfg.decimation):
                    base.scene.write_data_to_sim()
                    base.sim.step(render=False)
                    base.scene.update(dt=base.physics_dt)
                if written is None:
                    written = object_poses(base, objects)[0]
                if bool(relations_hold(base, name)[0]):
                    held_at = step
                    break
            end = object_poses(base, objects)[0]
            judged = evaluate(task["success"], loaded["geometry"], end, object_states(base, task)[0])
            print(f"{name} successful end held after {held_at} control steps: {judged}")
            assert held_at is not None, f"{name}: the successful end is not a success: {judged}"
            for obj in objects:
                moved = float(np.linalg.norm(end[obj][:3, 3] - written[obj][:3, 3]))
                print(f"{name} {obj}: successful end settled {moved:.4f} m")
                assert moved < SETTLED_M, f"{name}: {obj} moved {moved:.3f} m from the successful end"
        finally:
            env.close()
    return True


@requires_pack
def test_g2_eval_environments():
    assert run_function_with_persistent_simulation_app(_test_g2_eval_environments, headless=True)
