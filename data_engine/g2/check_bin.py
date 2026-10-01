# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cold-cook a G2 bin and verify its cavity, floor contacts and CPU fallback log."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def probe(args):
    """Run a small physical drop test using the production bin spawn configuration."""
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    sim_args = get_isaaclab_arena_cli_parser().parse_args(["--device", "cuda:0"])
    sim_args.headless = True
    with SimulationAppContext(sim_args):
        import numpy as np

        import carb
        import isaaclab.sim as sim
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from omni.physx.scripts.ifaces import get_physx_cooking_private_interface
        from pxr import PhysicsSchemaTools, PhysxSchema, UsdUtils

        from isaaclab_arena_environments.g2_clean_workcell_environment import G2CleanWorkcellTableEnvironment

        carb.logging.acquire_logging().set_log_enabled(True)
        carb.settings.get_settings().set("/log/level", "warning")
        carb.settings.get_settings().set("/log/outputStreamLevel", "warning")
        carb.log_warn("G2_BIN_DIAGNOSTIC_LOGGING_ENABLED")
        # Process-local settings: bypass asynchronous cache reuse without deleting shared caches.
        carb.settings.get_settings().set("/physics/cooking/ujitsoCollisionCooking", False)
        get_physx_cooking_private_interface().release_runtime_mesh_cache()
        import omni.kit.app

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "omni.physx.asset_validator", True
        )
        from omni.physxassetvalidator import get_physx_asset_validator_interface

        context = sim.SimulationContext(sim.SimulationCfg(dt=1 / 120, device="cuda:0", enable_scene_query_support=True))
        ground = sim.CuboidCfg(size=(2, 2, 0.02), collision_props=sim.CollisionPropertiesCfg())
        ground.func("/World/Ground", ground, translation=(0, 0, -0.01))
        factory = G2CleanWorkcellTableEnvironment()
        obj = factory.make_object(
            factory.spec["bin_asset"], {"instance_name": "blue_bin", "scale": tuple(factory.spec["bin_scale"])}
        )
        cfg = obj.object_cfg.copy()
        cfg.prim_path = "/World/bin"
        cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0.01))
        if args.variant == "source":
            cfg.spawn.prim_physics = {}
        container = RigidObject(cfg)
        blocks = []
        for index, (x, y) in enumerate([(0, 0), (-0.08, -0.04), (0.06, 0.04)]):
            blocks.append(
                RigidObject(
                    RigidObjectCfg(
                        prim_path=f"/World/block{index}",
                        spawn=sim.CuboidCfg(
                            size=(0.025, 0.025, 0.025),
                            rigid_props=sim.RigidBodyPropertiesCfg(),
                            mass_props=sim.MassPropertiesCfg(mass=0.05),
                            collision_props=sim.CollisionPropertiesCfg(),
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.18)),
                    )
                )
            )
        context.reset()
        # Force the validator's cooking path: an ordinary reset may reuse a cached hull and emit no warning.
        collider = context.stage.GetPrimAtPath("/World/bin/Bin_B04_01")
        stage_id = UsdUtils.StageCache.Get().GetId(context.stage).ToLongInt()
        get_physx_cooking_private_interface().release_runtime_mesh_cache()
        compatibility_hint = get_physx_asset_validator_interface().convex_gpu_compatibility_is_valid(
            stage_id, PhysicsSchemaTools.sdfPathToInt(collider.GetPath())
        )
        # This API has returned True alongside CPU fallback warnings; the parent must also inspect the log.
        for _ in range(300):
            context.step(render=False)
            container.update(context.get_physics_dt())
            for block in blocks:
                block.update(context.get_physics_dt())
        scenes = []
        for prim in context.stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                api = PhysxSchema.PhysxSceneAPI(prim)
                scenes.append(
                    dict(
                        gpu_dynamics=api.GetEnableGPUDynamicsAttr().Get(), broadphase=api.GetBroadphaseTypeAttr().Get()
                    )
                )
        assert scenes and all(s["gpu_dynamics"] and s["broadphase"] == "GPU" for s in scenes)
        points = [b.data.root_pos_w.torch[0].cpu().numpy() for b in blocks]
        velocities = [float(b.data.root_lin_vel_w.torch[0].norm()) for b in blocks]
        bin_origin = container.data.root_pos_w.torch[0].cpu().numpy()
        reference = factory.spec["bin_geometry_reference"]
        floor = reference["floor_z_m"] * factory.spec["bin_scale"][2] / reference["scale"][2]
        floor_errors = [abs(float(p[2] - bin_origin[2] - 0.0125 - floor)) for p in points]
        assert max(floor_errors) < 0.003, f"Object did not settle on cavity floor: {floor_errors}"
        assert max(velocities) < 0.01, f"Unsettled probes: {velocities}"
        wall_checks = []
        # Exercise real contacts rather than a scene-query acceleration structure.
        for index, block in enumerate(blocks[1:]):
            state = block.data.root_state_w.torch.clone()
            state[:, :3] = state.new_tensor([0.7, index * 0.1, 0.03])
            state[:, 7:] = 0
            block.write_root_state_to_sim(state)
        bounds = np.asarray(reference["inner_xy_bounds_m"]) * (
            np.asarray(factory.spec["bin_scale"][:2]) / np.asarray(reference["scale"][:2])
        )
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            state = blocks[0].data.root_state_w.torch.clone()
            state[:, :3] = state.new_tensor(bin_origin + [0, 0, 0.035])
            state[:, 3:7] = state.new_tensor([0, 0, 0, 1])
            state[:, 7:] = 0
            state[:, 7 + axis] = sign * 2.0
            blocks[0].write_root_state_to_sim(state)
            furthest = 0.0
            for _ in range(180):
                context.step(render=False)
                container.update(context.get_physics_dt())
                blocks[0].update(context.get_physics_dt())
                relative = (blocks[0].data.root_pos_w.torch[0] - container.data.root_pos_w.torch[0]).cpu().numpy()
                furthest = max(furthest, float(sign * relative[axis]))
                assert bounds[0, axis] - 0.015 < relative[axis] < bounds[1, axis] + 0.015, "Probe escaped bin wall"
                assert relative[2] > 0.005, "Probe passed through floor"
            assert furthest > 0.05, "Probe did not reach wall region"
            wall_checks.append(dict(axis=axis, sign=sign, furthest_m=furthest, final_relative_m=relative.tolist()))
        result = dict(
            variant=args.variant,
            validator_compatibility_hint=compatibility_hint,
            device=context.device,
            physics_scenes=scenes,
            physics_steps=1020,
            scale=factory.spec["bin_scale"],
            floor_errors_m=floor_errors,
            settled_speeds_m_s=velocities,
            block_positions_m=np.asarray(points).tolist(),
            wall_contact_checks=wall_checks,
            bin_displacement_xy_m=float(container.data.root_pos_w.torch[0, :2].norm()),
            contact_checks_passed=True,
        )
        from isaaclab_arena.assets.usdcraft_scene import bundle_root

        source_paths = [
            Path(__file__),
            ROOT / "isaaclab_arena/assets/convex_decomposition.py",
            ROOT / "isaaclab_arena_environments/g2_clean_workcell_environment.py",
            ROOT / "data_engine/g2/clean_workcell_table.yaml",
        ]
        result["source_sha256"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        }
        result["asset_manifest_sha256"] = hashlib.sha256((bundle_root() / "manifest.json").read_bytes()).hexdigest()
        (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    """Isolate the simulator lifetime and reject CPU fallback in the qualified variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("source", "qualified"), default="qualified")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--info", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_process:
        probe(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    log = args.output / "runtime.log"
    with log.open("w") as stream:
        status = subprocess.run(
            [
                sys.executable,
                __file__,
                "--variant",
                args.variant,
                "--output",
                str(args.output),
                "--probe-process",
                "--info",
            ],
            cwd=ROOT,
            env=dict(os.environ, ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE="1"),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    assert status.returncode == 0, f"Probe failed; inspect {log}"
    report = args.output / "report.json"
    result = json.loads(report.read_text())
    assert "G2_BIN_DIAGNOSTIC_LOGGING_ENABLED" in log.read_text(), "PhysX warning capture is not verified"
    result["cpu_collision_fallback_warnings"] = [
        line for line in log.read_text().splitlines() if "fall back to CPU" in line
    ]
    result["passed"] = result["contact_checks_passed"] and not result["cpu_collision_fallback_warnings"]
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.variant == "qualified":
        assert result["passed"], "CPU collision fallback remains"


if __name__ == "__main__":
    main()
