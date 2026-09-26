# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded workcell joint actions with a single initial-state restore."""

import hashlib
import json
import yaml
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main():
    """Verify physical replay and optionally record an overview video."""
    parser = get_isaaclab_arena_cli_parser()
    parser.set_defaults(device="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_completed_diagnostic", action="store_true")
    parser.add_argument("--settle_steps", type=int, default=900)
    args = parser.parse_args()
    assert args.device.startswith("cuda"), "Workcell replay requires CUDA"
    canonical_spec = Path("isaaclab_arena/embodiments/g2/assets/clean_workcell_table.yaml")
    actual_spec = yaml.safe_load(canonical_spec.read_text())
    source_spec = yaml.safe_load((args.run_dir / "sources/clean_workcell_table.yaml").read_text())
    for spec in (actual_spec, source_spec):
        spec.pop("validation_status", None)
    assert actual_spec == source_spec, "Physical scene configuration changed"
    source_success = json.loads((args.run_dir / "status.json").read_text())["task_success"]
    if not source_success:
        source_report = json.loads((args.run_dir / "reach_report.json").read_text())
        assert args.allow_completed_diagnostic and set(source_report["completed_tools"]) == {
            "metal_stock",
            "drill",
            "finished_gear",
        }, "Diagnostic replay requires all three completed physical placements"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with SimulationAppContext(args):
        import h5py
        import itertools
        import numpy as np
        import torch

        from isaaclab.managers import TerminationTermCfg

        from isaaclab_arena.assets.registries import EnvironmentRegistry
        from isaaclab_arena.embodiments.g2.g2 import G2CameraCfg, G2JointPositionActionsCfg
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena_environments.g2_clean_workcell_environment import G2CleanWorkcellTableEnvironment
        from isaaclab_arena_environments.g2_workbench_environments import G2WorkbenchEnvironmentCfg

        registry = EnvironmentRegistry()
        assert registry.is_registered("g2_clean_workcell_table", ensure_loaded=False)
        assert not registry.is_registered("g2_organize_tools", ensure_loaded=False)
        factory = G2CleanWorkcellTableEnvironment()
        description = factory.build(G2WorkbenchEnvironmentCfg(enable_cameras=args.enable_cameras))
        from isaaclab_arena_examples.g2_workcell.check_contract import check_sorting_contract

        corners = {}
        for name in factory.spec["objects"]:
            bounds = description.scene.assets[name].get_bounding_box()
            corners[name] = list(itertools.product(*zip(bounds.min_point[0].tolist(), bounds.max_point[0].tolist())))
        contract_checks = check_sorting_contract(factory.spec, corners)
        (args.output_dir / "contract_checks.json").write_text(json.dumps(contract_checks, indent=2))
        print("WORKCELL_CONTRACT_CHECKS", json.dumps(contract_checks), flush=True)
        description.embodiment.action_config = G2JointPositionActionsCfg()
        description.embodiment.camera_config = G2CameraCfg()
        original = description.env_cfg_callback
        from isaaclab_arena.embodiments.g2.recorders import G2CollectionSuccessTerm

        def configure(cfg):
            cfg = original(cfg)
            cfg.terminations.success.func = G2CollectionSuccessTerm
            for name, term in vars(cfg.terminations).items():
                if name != "success" and isinstance(term, TerminationTermCfg):
                    setattr(cfg.terminations, name, None)
            cfg.recorders = None
            cfg.sim.render_interval = cfg.decimation
            return cfg

        description.env_cfg_callback = configure
        env = ArenaEnvBuilder(description, arena_env_builder_cfg_from_argparse(args)).make_registered()
        source_hashes = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                canonical_spec,
                Path("isaaclab_arena_environments/g2_clean_workcell_environment.py"),
                Path("isaaclab_arena_environments/g2_workbench_environments.py"),
                canonical_spec.with_name("aluminum_stock.usda"),
                args.run_dir / "step_trace.npz",
            )
        }
        report = {
            "initial_state_restores": 1,
            "per_frame_object_restores": 0,
            "task_success": False,
            "validation_source_hashes": source_hashes,
        }
        writer = None
        try:
            base = env.unwrapped
            env.reset()
            with h5py.File(args.run_dir / ("episodes.hdf5" if source_success else "episodes_failed.hdf5")) as file:
                episode = file["data"][next(iter(file["data"]))]
                actions = episode["actions"][:]
                initial = {
                    category: {
                        name: {key: torch.tensor(value[0], device=base.device)[None] for key, value in fields.items()}
                        for name, fields in objects.items()
                    }
                    for category, objects in episode["initial_state"].items()
                }
            base.scene.reset_to(
                initial, env_ids=torch.tensor([0], dtype=torch.int32, device=base.device), is_relative=True
            )
            base.sim.forward()
            base.scene.update(base.step_dt)
            if args.enable_cameras:
                from isaaclab_arena_examples.g2_workcell.cameras import ThreeViewWriter

                writer = ThreeViewWriter(args.output_dir, 1 / base.step_dt)
            trace = dict(np.load(args.run_dir / "step_trace.npz"))
            object_errors, joint_errors, successes = [], [], []
            for i, action in enumerate(actions):
                env.step(torch.tensor(action, device=base.device)[None])
                successes.append(bool(base.progress_tracker.is_complete()[0]))
                joint_errors.append(
                    float(np.max(abs(base.scene["robot"].data.joint_pos.torch[0].cpu().numpy() - trace["q_after"][i])))
                )
                actual = np.stack(
                    [base.scene[str(name)].data.root_pos_w.torch[0].cpu().numpy() for name in trace["object_names"]]
                )
                object_errors.append(float(np.linalg.norm(actual - trace["object_positions"][i], axis=1).max()))
                if writer is not None:
                    writer.append(base.scene)
                if i % 200 == 0:
                    print("REPLAY_PROGRESS", i, len(actions), object_errors[-1], flush=True)
            extra = 0
            while not successes[-1] and extra < args.settle_steps:
                env.step(torch.tensor(actions[-1], device=base.device)[None])
                successes.append(bool(base.progress_tracker.is_complete()[0]))
                if writer is not None:
                    writer.append(base.scene)
                extra += 1
                if extra % 100 == 0:
                    print("SETTLE_PROGRESS", extra, successes[-1], flush=True)
            report.update(
                task_success=successes[-1],
                additional_settle_steps=extra,
                source_task_success=source_success,
                steps=len(actions),
                maximum_joint_error_rad=max(joint_errors),
                maximum_object_position_error_m=max(object_errors),
                final_object_position_error_m=object_errors[-1],
            )
            assert report["task_success"], "Physical joint replay did not finish the task"
        finally:
            if writer is not None:
                writer.close()
            (args.output_dir / "replay_report.json").write_text(json.dumps(report, indent=2))
            print("REPLAY_RESULT", json.dumps(report), flush=True)
            env.close()


if __name__ == "__main__":
    main()
