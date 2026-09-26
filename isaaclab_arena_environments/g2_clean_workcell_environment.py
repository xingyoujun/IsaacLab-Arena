# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Three-category G2 workcell cleanup with fixed semantic bin colors."""

import copy
import hashlib
import itertools
import json
import yaml
from pathlib import Path

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.assets.usdcraft_scene import resolve_asset
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena_environments.g2_workbench_environments import G2WorkbenchEnvironmentCfg, _G2WorkbenchEnvironment

CONFIG_PATH = Path(__file__).parents[1] / "isaaclab_arena/embodiments/g2/assets/clean_workcell_table.yaml"


def load_spec():
    """Load and validate the fixed one-object-per-category sorting contract."""
    from isaaclab_arena.assets.usdcraft_scene import bundle_root

    root = bundle_root()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if "g2_clean_workcell" in manifest["entries"]:
            descriptor = json.loads(Path(resolve_asset("g2_clean_workcell")).read_text())
            assert (
                hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == descriptor["configuration_sha256"]
            ), "G2 scene configuration differs from the selected bundle snapshot; regenerate the local bundle"
    spec = yaml.safe_load(CONFIG_PATH.read_text())
    objects, bins = spec["objects"], spec["bins"]
    assert len(objects) == len(bins) == 3
    assert set(spec["sequence"]) == set(objects) and len(spec["sequence"]) == 3
    assert {item["destination"] for item in objects.values()} == set(bins)
    assert {item["category"] for item in objects.values()} == {
        "raw_material",
        "tool",
        "finished_part",
    }
    for item in objects.values():
        assert item["category"] == bins[item["destination"]]["category"]
    return spec


def sorted_objects(env, spec, corners):
    """Require complete bin containment, settled objects and open grippers for one second."""
    import torch

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    passed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    reference = spec["bin_geometry_reference"]
    ratio = torch.tensor(spec["bin_scale"], device=env.device) / torch.tensor(reference["scale"], device=env.device)
    bounds = torch.tensor(reference["inner_xy_bounds_m"], device=env.device) * ratio[:2]
    floor = reference["floor_z_m"] * ratio[2]
    rim = reference["rim_max_z_m"] * ratio[2]
    limits = spec["success"]
    for name, item in spec["objects"].items():
        data = env.scene[name].data
        target = env.scene[item["destination"]].data
        local = torch.tensor(corners[name], device=env.device).expand(env.num_envs, -1, -1)
        world = quat_apply(data.root_quat_w.torch[:, None, :].expand(-1, 8, -1), local)
        world = world + data.root_pos_w.torch[:, None, :]
        relative = quat_apply_inverse(
            target.root_quat_w.torch[:, None, :].expand(-1, 8, -1),
            world - target.root_pos_w.torch[:, None, :],
        )
        passed &= (relative[:, :, :2] >= bounds[0] + limits["planar_margin_m"]).all(dim=(1, 2))
        passed &= (relative[:, :, :2] <= bounds[1] - limits["planar_margin_m"]).all(dim=(1, 2))
        bottom = relative[:, :, 2].amin(dim=1)
        passed &= (bottom - floor).abs() <= limits["floor_tolerance_m"]
        passed &= relative[:, :, 2].amax(dim=1) <= rim + 0.003
        passed &= data.root_lin_vel_w.torch.norm(dim=1) < limits["max_linear_speed_m_s"]
        passed &= data.root_ang_vel_w.torch.norm(dim=1) < limits["max_angular_speed_rad_s"]
        passed &= target.root_lin_vel_w.torch.norm(dim=1) < limits["max_linear_speed_m_s"]
    robot = env.scene["robot"]
    for side, outer, inner in (("l", "idx41", "idx31"), ("r", "idx81", "idx71")):
        q = robot.data.joint_pos.torch
        passed &= q[:, robot.joint_names.index(f"{outer}_gripper_{side}_outer_joint1")] > 0.65
        passed &= q[:, robot.joint_names.index(f"{inner}_gripper_{side}_inner_joint1")] < -0.65
    if not hasattr(env, "_workcell_settled_steps"):
        env._workcell_settled_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    count = env._workcell_settled_steps
    count[env.episode_length_buf <= 1] = 0
    count[:] = torch.where(passed, count + 1, 0)
    return count * env.step_dt >= limits["stable_seconds"]


@register_environment
class G2CleanWorkcellTableEnvironment(_G2WorkbenchEnvironment, ArenaEnvironmentFactory[G2WorkbenchEnvironmentCfg]):
    """Sort aluminum stock, a cordless drill and a finished gear into three colored bins."""

    name = "g2_clean_workcell_table"
    _legacy_argparse_cfg_type = G2WorkbenchEnvironmentCfg

    def __init__(self, spec=None):
        super().__init__()
        self.spec = copy.deepcopy(spec) if spec is not None else load_spec()
        self.object_layout = tuple(
            (item["asset"], name, *item["xy"], item["yaw_deg"]) for name, item in self.spec["objects"].items()
        ) + tuple(
            (self.spec["bin_asset"], name, *item["xy"], self.spec["bin_yaw_deg"])
            for name, item in self.spec["bins"].items()
        )
        self.scale_overrides = {name: tuple(self.spec["bin_scale"]) for name in self.spec["bins"]}
        self.scale_overrides.update(
            {name: tuple(item["scale"]) for name, item in self.spec["objects"].items() if "scale" in item}
        )

    def make_object(self, registry_name, params):
        """Use a physical aluminum billet and color the existing open-bin asset."""
        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType

        name = params["instance_name"]
        if registry_name == "aluminum_stock":
            return Object(
                name=name,
                usd_path=str(resolve_asset("g2_aluminum_stock")),
                scale=params.get("scale", (1.0, 1.0, 1.0)),
                object_type=ObjectType.RIGID,
                spawn_cfg_addon={
                    "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)
                },
            )
        template = super().make_object(registry_name, params)
        if name not in self.spec["bins"]:
            return template
        spawn = copy.deepcopy(template.spawn_cfg_addon)
        spawn["visual_material"] = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(self.spec["bins"][name]["rgb"]),
            roughness=0.55,
            metallic=0.0,
        )
        return Object(
            name=name,
            usd_path=template.usd_path,
            scale=template.scale,
            object_type=template.object_type,
            spawn_cfg_addon=spawn,
            asset_cfg_addon=copy.deepcopy(template.asset_cfg_addon),
        )

    def build(self, cfg):
        """Build the physical scene and its category-aware success condition."""
        environment = super().build(cfg)
        corners = {}
        for name in self.spec["objects"]:
            bounds = environment.scene.assets[name].get_bounding_box()
            corners[name] = list(itertools.product(*zip(bounds.min_point[0].tolist(), bounds.max_point[0].tolist())))
        environment.task = CleanWorkcellTask(self.spec, corners)
        return environment


class CleanWorkcellTask(NoTask):
    """Expose the fixed sorting instruction and physical completion predicate."""

    def __init__(self, spec, corners):
        super().__init__()
        from functools import partial

        from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
        from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

        self.task_description = spec["instruction"]
        self.terminations = TaskTerminationCfg(
            timeout_s=600.0,
            desired_subtask_success_state=[True],
            success=[
                ProgressObjective(
                    name="clean_workcell",
                    parent_subtask_idx=0,
                    predicate_sequence=[partial(sorted_objects, spec=spec, corners=corners)],
                )
            ],
        )

    def get_termination_cfg(self):
        """Return category-aware success and timeout terms."""
        return self.terminations
