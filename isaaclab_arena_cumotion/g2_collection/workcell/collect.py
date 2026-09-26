# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate physical workcell cleanup and retain successful and failed raw episodes."""

import copy
import hashlib
import json
import shutil
import traceback
from pathlib import Path

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def export_geometry(factory, directory):
    """Export scaled collision surfaces and matching cavity metrics for the new scene."""
    import itertools
    import numpy as np
    import trimesh

    from pxr import Usd

    from isaaclab_arena_cumotion.g2_collection.workcell.mesh_geometry import mesh_parts, selected_mesh
    from isaaclab_arena_cumotion.g2_collection.workcell.scene_geometry import scaled_bin_geometry
    from isaaclab_arena_environments.g2_workbench_environments import G2WorkbenchEnvironmentCfg

    description = factory.build(G2WorkbenchEnvironmentCfg())
    from isaaclab_arena_cumotion.g2_collection.workcell.check_contract import check_sorting_contract

    corners = {}
    for name in factory.spec["objects"]:
        bounds = description.scene.assets[name].get_bounding_box()
        corners[name] = list(itertools.product(*zip(bounds.min_point[0].tolist(), bounds.max_point[0].tolist())))
    checks = check_sorting_contract(factory.spec, corners)
    (directory.parent / "contract_checks.json").write_text(json.dumps(checks, indent=2))
    print("WORKCELL_CONTRACT_CHECKS", json.dumps(checks), flush=True)
    directory.mkdir()
    report = {"objects": {}, "bins": {}}
    for _, name, *_ in factory.object_layout:
        asset = description.scene.assets[name]
        if name == "metal_stock":
            mesh = trimesh.creation.box(extents=[0.07, 0.045, 0.035])
            vertices, faces = mesh.vertices * np.asarray(asset.scale), mesh.faces
        else:
            stage = Usd.Stage.Open(asset.usd_path)
            vertices, faces, collision = selected_mesh(mesh_parts(stage, stage.GetDefaultPrim()))
            assert collision, f"Missing authored collision mesh: {name}"
            vertices *= np.asarray(asset.scale)
        np.savez_compressed(directory / f"{name}_mesh.npz", vertices=vertices, faces=faces)
        report["objects"][name] = {
            "asset": asset.usd_path,
            "scale": asset.scale,
            "bounds_m": [vertices.min(0).tolist(), vertices.max(0).tolist()],
        }
    reference = factory.spec["bin_geometry_reference"]
    geometry = {k: v for k, v in reference.items() if k != "scale"}
    geometry["inner_size_xy_m"] = np.diff(np.array(geometry["inner_xy_bounds_m"]), axis=0)[0].tolist()
    for name in factory.spec["bins"]:
        report["bins"][name] = scaled_bin_geometry(geometry, np.array(factory.spec["bin_scale"]) / reference["scale"])
    (directory / "geometry_report.json").write_text(json.dumps(report, indent=2))


def configure_recording(description, args):
    """Keep explicit task evaluation and save compact pre-action core observations."""
    from isaaclab.managers import DatasetExportMode, TerminationTermCfg

    from isaaclab_arena.embodiments.g2.recorders import G2CollectionSuccessTerm, core_recorder_cfg
    from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg

    original = description.env_cfg_callback

    def configure(cfg):
        cfg = original(cfg)
        cfg.sim.render_interval = cfg.decimation
        cfg.terminations.success.func = G2CollectionSuccessTerm
        for name, term in vars(cfg.terminations).items():
            if name != "success" and isinstance(term, TerminationTermCfg):
                setattr(cfg.terminations, name, None)
        cfg.recorders = ArenaEnvRecorderManagerCfg(
            dataset_export_dir_path=str(args.output_dir),
            dataset_filename="episodes",
            dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES,
            export_in_record_pre_reset=False,
        )
        if not args.enable_cameras:
            cfg.recorders.record_pre_step_flat_camera_observations = None
        cfg.recorders.g2_core = core_recorder_cfg()
        return cfg

    description.env_cfg_callback = configure


def execute(session, config):
    """Perform one reset and retain the complete observed trajectory even on failure."""
    import numpy as np
    import torch

    from isaaclab_arena_cumotion.g2_collection.workcell.grasp_motion import run_grasp_lift
    from isaaclab_arena_cumotion.g2_collection.workcell.placement import bin_sample, evaluate_placement, run_pick_place

    s = session
    from isaaclab_arena_cumotion.g2_collection.workcell.native_cumotion import install

    install(s)
    s.report["planning_backend"] = s.args.planner_backend
    from isaaclab_arena.recording.alignment import transition_metadata

    contract = transition_metadata("g2", ["head_camera_rgb", "left_wrist_camera_rgb", "right_wrist_camera_rgb"])
    s.base.cfg.get_ep_meta = lambda: {
        "collection_contract": contract,
        "env_name": "g2_clean_workcell_table",
        "action_config": "isaaclab_arena.embodiments.g2.g2:G2JointPositionActionsCfg",
        "joint_names": s.robot.joint_names,
        "task": s.args.scene_factory.spec,
        "mode": s.args.stage,
        "pose_convention": "right then left xyz+xyzw; metres; pre-action observation",
    }
    s.report.update(
        task_success=False,
        completed_tools={},
        collection_stage=s.args.stage,
        task_variant="clean_workcell_table",
        joint_names=s.robot.joint_names,
        configuration={
            "seed": s.args.seed,
            "table_height_m": 0.75,
            "layout": "fixed",
            "image_capture": "live_pre_action" if s.args.enable_cameras else "none",
        },
    )
    order = config["sequence"] if s.args.stage == "sequence" else [s.args.tool]
    s.report["sequence_order"] = order
    placed = {}
    bin_origins = {}

    def guard():
        for name, origin in bin_origins.items():
            displacement = float(np.linalg.norm(s.object_pose(name)[0] - origin))
            assert displacement < 0.005, f"Collection bin moved: {name}: {displacement} m"
        for name, reference in placed.items():
            destination = config["mapping"][name]
            result = evaluate_placement([reference, bin_sample(s, name, destination)], s.geometry["bins"][destination])
            assert result["passed"], f"Previously placed {name} disturbed: {result}"

    s.sequence_guard[0] = guard
    s.reset()
    initial_displacement = {}
    origins = s.base.scene.env_origins
    origins = getattr(origins, "torch", origins)
    for name in s.meshes:
        default_state = s.base.scene[name].data.default_root_state
        expected = getattr(default_state, "torch", default_state)[0, :2]
        actual = s.base.scene[name].data.root_pos_w.torch[0, :2] - origins[0, :2]
        initial_displacement[name] = float(torch.linalg.vector_norm(actual - expected))
    s.report["initial_layout_displacement_xy_m"] = initial_displacement
    assert max(initial_displacement.values()) < 0.005, f"Initial layout shifted during settling: {initial_displacement}"
    bin_origins.update({name: s.object_pose(name)[0].copy() for name in s.args.scene_factory.spec["bins"]})
    try:
        for name in order:
            s.args.tool = name
            s.args.destination = config["mapping"][name]
            s.geometry["bin"] = copy.deepcopy(s.geometry["bins"][s.args.destination])
            operation = run_grasp_lift if s.args.stage == "grasp_lift" else run_pick_place
            operation(s, config, initialize=False)
            s.report["completed_tools"][name] = copy.deepcopy({
                key: s.report[key]
                for key in (
                    "grasp",
                    "placement",
                    "grasp_lift_success",
                    "place_success",
                    "carry_max_drift_m",
                    "carry_max_angle_deg",
                )
                if key in s.report
            })
            if s.args.stage != "grasp_lift":
                placed[name] = bin_sample(s, name, s.args.destination)
            s.save()
            print("WORKCELL_OBJECT_COMPLETE", name, flush=True)
        if s.args.stage == "sequence":
            s.phase[0] = "verify_all_categories"
            success = False
            for index in range(config.get("final_settle_steps", 900)):
                s.step()
                success = bool(s.base.progress_tracker.is_complete()[0])
                if success and index >= 29:
                    break
            s.report["final_settle_steps"] = index + 1
            assert success, "Three-category completion predicate failed"
            assert s.report["reset_count"] == 1
            s.report["task_success"] = True
        s.report["stage_success"] = True
    finally:
        s.save()
        s.base.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        s.base.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[s.report["task_success"]]], device=s.base.device)
        )
        s.base.recorder_manager.export_episodes([0])
        print("WORKCELL_RAW_EXPORTED", s.report["task_success"], flush=True)


def main(argv=None):
    """Start a single immutable diagnostic attempt using the shared motion executor."""
    import yaml

    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stage", choices=["grasp_lift", "pick_place", "sequence"], default="sequence")
    parser.add_argument("--tool", choices=["metal_stock", "drill", "finished_gear"], default="metal_stock")
    parser.add_argument("--robot_yaml", type=Path, default=None)
    parser.add_argument("--robot_urdf", type=Path, default=None)
    parser.set_defaults(device="cuda:0")
    parser.add_argument("--planner_backend", choices=("cumotion",), default="cumotion")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    args = parser.parse_args(argv)
    from isaaclab_arena.assets.g2_asset_paths import planning_path

    args.robot_yaml = planning_path("g2_robot_yaml", args.robot_yaml)
    args.robot_urdf = planning_path("g2_tcp_urdf", args.robot_urdf)
    args.config = args.config or Path(__file__).with_name("motion_config_cumotion.yaml")
    assert args.device.startswith("cuda"), "Workcell collection requires CUDA"
    assert args.num_envs == 1
    assert not args.record_video or args.enable_cameras
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text())
    args.mode = "sequence" if args.stage == "sequence" else "pick_place"
    args.position_tolerance, args.rotation_tolerance, args.object_tolerance = 0.025, 5.0, 0.005
    args.record_steps = True
    args.payload_capacities = {"left": 64, "right": 256}
    args.destination = config["mapping"][args.tool]
    args.geometry_dir = args.output_dir / "geometry"
    args.configure_description = lambda description: configure_recording(description, args)
    shutil.copyfile(args.config, args.output_dir / "motion_config.yaml")
    sources = args.output_dir / "sources"
    sources.mkdir()
    from isaaclab_arena.assets.g2_asset_paths import transfer_urdf

    transfer_models = {path for tool in config["tools"].values() if (path := transfer_urdf(tool)) is not None}
    manifest = {}
    for path in [
        *Path(__file__).parent.glob("*.py"),
        args.config,
        Path("isaaclab_arena_environments/g2_clean_workcell_environment.py"),
        Path("isaaclab_arena_environments/g2_workbench_environments.py"),
        Path("isaaclab_arena/embodiments/g2/assets/clean_workcell_table.yaml"),
        args.robot_yaml,
        args.robot_urdf,
        *transfer_models,
    ]:
        manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path not in (args.robot_yaml, args.robot_urdf) and path.suffix != ".urdf":
            shutil.copyfile(path, sources / path.name)
    (args.output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    with SimulationAppContext(args):
        status = {"completed": False, "task_success": False, "stage_success": False}
        try:
            from isaaclab_arena_cumotion.g2_collection.workcell.motion import run_session
            from isaaclab_arena_environments.g2_clean_workcell_environment import G2CleanWorkcellTableEnvironment

            args.scene_factory = G2CleanWorkcellTableEnvironment()
            export_geometry(args.scene_factory, args.geometry_dir)
            if args.profile:
                import cProfile

                profiler = cProfile.Profile()
                try:
                    profiler.runcall(run_session, args, lambda session: execute(session, config))
                finally:
                    profiler.dump_stats(str(args.output_dir / "collector_profile.pstats"))
            else:
                run_session(args, lambda session: execute(session, config))
            report = json.loads((args.output_dir / "reach_report.json").read_text())
            status.update(
                completed=True, task_success=report["task_success"], stage_success=report.get("stage_success", False)
            )
        except Exception:
            status["error"] = traceback.format_exc()
            (args.output_dir / "error.txt").write_text(status["error"])
            raise
        finally:
            (args.output_dir / "status.json").write_text(json.dumps(status, indent=2))
            reach_path = args.output_dir / "reach_report.json"
            if reach_path.exists():
                report = json.loads(reach_path.read_text())
                report["attempts"] = [{"attempt": 0, "success": status["task_success"]}]
                (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
            print("WORKCELL_STATUS", json.dumps(status), flush=True)


if __name__ == "__main__":
    import os

    assert os.environ.get("ARENA_G2_FRAMEWORK") == "1", "Use tools/data_collection/g2.py"
    main()
