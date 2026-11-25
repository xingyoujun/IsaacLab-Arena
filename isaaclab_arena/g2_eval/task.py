# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The task of a G2-Eval environment: each episode's layout, success and time limit from the pack's ``task.json``.

A ``g2.eval.task.v1`` task draws each episode's layout (``draw_layout``) and judges its success relations
(``success.evaluate``). A ``g2.eval.task.v2`` task starts each episode from one of its listed layouts
(``layout.layout_index``) and is judged every control step by its success check (``success_check.Judge``, one per
environment, made anew at each reset): the episode ends as a success when the judge decides ``success`` and as a
failure when it decides ``failed`` (a query counted past its aim). The judge's start (each object's pose and each
arm's end body) is the state after the episode's first control step: G2-Eval has no settle steps after a reset,
unlike the data engine's evaluation (15).
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.g2_eval import layout, pack
from isaaclab_arena.g2_eval.success import evaluate
from isaaclab_arena.g2_eval.success_check import Judge
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import make_configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from isaaclab_arena.assets.object import Object

REST_LIFT_M = 0.001
"""Objects start this far above their resting pose when ``task.json`` gives no ``spawn_lift_m`` (as the data engine)."""
DROPPED_BELOW_M = 0.3
"""An object this far below the table top has fallen off it: the episode ends as a failure."""
GRIPPER_JOINTS = (("l", "idx41", "idx31"), ("r", "idx81", "idx71"))
"""Per gripper: side, outer and inner finger joint prefixes (outer opens positive, inner negative)."""
ARMS = {"left": "l", "right": "r"}
"""The success check's arm names and their gripper sides."""


def is_v2(task: dict) -> bool:
    """Whether a ``task.json`` is a ``g2.eval.task.v2`` task (judged by its ``success_check``)."""
    return task["schema"] == pack.TASK_SCHEMA_CHECK


def contact_sensor_name(obj: str, target: str) -> str:
    """The scene name of the sensor measuring ``obj``'s contact force against ``target``."""
    return f"g2_eval_contact_{obj}_on_{target}"


def contact_pairs(task: dict) -> list[tuple[str, str]]:
    """The (object, target) pairs whose contact force the task's success reads."""
    return sorted({(e["object"], e["target"]) for e in task.get("success") or [] if e["relation"] == "placed"})


class G2EvalTask(TaskBase):
    """One G2-Eval task (``pack.load_task``): reset places the episode's layout, ``success`` judges the task.

    Args:
        name: The task's name in the locked pack.
        objects: The scene's task objects by name (their contact sensors are built from them).
    """

    def __init__(self, name: str, objects: dict[str, Object]):
        loaded = pack.load_task(name)
        task = loaded["task"]
        super().__init__(episode_length_s=task["episode_length_s"], task_description=task["instruction_en"])
        self.name = name
        self.objects = list(task["objects"])
        self.dropped_z = float(task["scene"]["table_top_z_m"]) - DROPPED_BELOW_M
        sensors = []
        for obj, target in contact_pairs(task):
            cfg = objects[obj].get_contact_sensor_cfg(contact_against_object=objects[target])
            sensors.append((contact_sensor_name(obj, target), type(cfg), cfg))
        self.scene_config = make_configclass("G2EvalSceneCfg", sensors)() if sensors else None

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        if is_v2(pack.load_task(self.name)["task"]):
            return G2EvalTerminationsCfg(
                success=TerminationTermCfg(func=check_succeeded, params={"name": self.name}),
                check_failed=TerminationTermCfg(func=check_failed, params={"name": self.name}),
                object_dropped=TerminationTermCfg(
                    func=object_dropped, params={"objects": self.objects, "minimum_z": self.dropped_z}
                ),
            )
        return G2EvalTerminationsCfg(
            success=TerminationTermCfg(func=task_success, params={"name": self.name}),
            object_dropped=TerminationTermCfg(
                func=object_dropped, params={"objects": self.objects, "minimum_z": self.dropped_z}
            ),
        )

    def get_events_cfg(self):
        return G2EvalEventsCfg(reset_layout=EventTermCfg(func=reset_layout, mode="reset", params={"name": self.name}))

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]


def rest_pose(spec: dict, lift_m: float = REST_LIFT_M) -> tuple[np.ndarray, np.ndarray]:
    """An object's start: its resting position lifted by ``lift_m``, and its xyzw quaternion."""
    position = np.asarray(spec["rest_pose"]["position_m"], float) + [0.0, 0.0, lift_m]
    return position, np.asarray(spec["rest_pose"]["quat_xyzw"], float)


def layout_pose(task: dict, index: int, obj: str, lift_m: float) -> tuple[np.ndarray, np.ndarray]:
    """An object's start in the listed layout ``index``: its position lifted by ``lift_m``, and its xyzw quaternion."""
    placed = task["randomization"]["layouts"][index]["objects"][obj]
    position = np.asarray(placed["position_m"], float) + [0.0, 0.0, lift_m]
    return position, np.asarray(placed["quat_xyzw"], float)


def draw_layout(task: dict, rng: np.random.Generator) -> dict:
    """One episode's layout draw (per object ``dxy_m`` and ``dyaw_deg``) by the task's randomization ``mode``:
    ``perturb`` (the data engine's sampler), ``uniform_xy`` (a legacy collection's offsets) or ``fixed``."""
    randomization = task["randomization"]
    mode = randomization.get("mode", "perturb")
    if mode == "fixed":
        return {obj: {"dxy_m": [0.0, 0.0], "dyaw_deg": 0.0} for obj in task["objects"]}
    if mode == "uniform_xy":
        positions = {obj: spec["rest_pose"]["position_m"][:2] for obj, spec in task["objects"].items()}
        keys = ("min_separation_m", "table_half_extent_m", "footprint_radius_m")
        options = {key: randomization[key] for key in keys if key in randomization}
        return layout.sample_uniform(positions, rng, randomization["xy_m"], randomization["objects"], **options)
    assert mode == "perturb", f"unknown randomization mode {mode!r}"
    settings = {key: randomization[key] for key in layout.SAMPLE_KEYS if key in randomization}
    return layout.sample(randomization["objects"], rng, **settings)


def _origins(env: ManagerBasedEnv) -> torch.Tensor:
    import warp as wp

    origins = env.scene.env_origins
    return wp.to_torch(origins) if isinstance(origins, wp.array) else origins


def reset_layout(env: ManagerBasedEnv, env_ids: torch.Tensor, name: str) -> None:
    """Place each object of the resetting environments (fixed ones too) at the episode's start.

    A v1 task: its rest pose moved by the episode's layout draw; the draw of episode ``k`` of environment ``i`` uses
    the generator seeded ``[env seed, i, k]``. A v2 task: its pose in the listed layout ``layout.layout_index`` (and a new
    judge for the episode). Either way a run with the same seed and number of environments evaluates the same layouts.
    """
    from scipy.spatial.transform import Rotation

    if env_ids is None:
        return
    task = pack.load_task(name)["task"]
    lift = float(task.get("spawn_lift_m", REST_LIFT_M))
    seed = int(env.cfg.seed or 0)
    episodes = env.__dict__.setdefault("_g2_eval_episodes", {})
    origins = _origins(env)
    v2 = is_v2(task)
    layouts = env.__dict__.setdefault("_g2_eval_layouts", {})
    for env_id in env_ids.tolist():
        episode = episodes.get(env_id, 0)
        episodes[env_id] = episode + 1
        if v2:
            layout_id = layout.layout_index(env_id, episode, env.num_envs, len(task["randomization"]["layouts"]))
            layouts[env_id] = layout_id
            starts = {obj: layout_pose(task, layout_id, obj, lift) for obj in task["objects"]}
            judges(env, name)[env_id] = {"judge": Judge(task["success_check"]), "initial": None, "ee_start": None}
        else:
            drawn = draw_layout(task, np.random.default_rng([seed, env_id, episode]))
            starts = {}
            for obj, spec in task["objects"].items():
                position, quat_xyzw = rest_pose(spec, lift)
                turn = Rotation.from_euler("z", drawn[obj]["dyaw_deg"], degrees=True)
                position[:2] += drawn[obj]["dxy_m"]
                starts[obj] = position, (turn * Rotation.from_quat(quat_xyzw)).as_quat()
        index = torch.tensor([env_id], device=env.device)
        for obj, (position, quat_xyzw) in starts.items():
            pose = torch.tensor(np.r_[position, quat_xyzw], dtype=torch.float32, device=env.device)
            pose[:3] += origins[env_id, :3].to(pose.dtype)
            asset = env.scene[obj]
            asset.write_root_pose_to_sim_index(root_pose=pose[None], env_ids=index)
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=env.device), env_ids=index
            )


def object_poses(env: ManagerBasedEnv, objects: list[str]) -> list[dict[str, np.ndarray]]:
    """Per environment, each object's pose T_W_O (4x4) relative to its environment origin."""
    from scipy.spatial.transform import Rotation

    import warp as wp

    origins = _origins(env).detach().cpu().numpy().astype(np.float64)
    poses = [{} for _ in range(env.num_envs)]
    for obj in objects:
        values = wp.to_torch(env.scene[obj].data.root_pose_w).detach().cpu().numpy().astype(np.float64)
        rotations = Rotation.from_quat(values[:, 3:7]).as_matrix()
        for env_id in range(env.num_envs):
            T = np.eye(4)
            T[:3, :3] = rotations[env_id]
            T[:3, 3] = values[env_id, :3] - origins[env_id, :3]
            poses[env_id][obj] = T
    return poses


def ee_poses(env: ManagerBasedEnv, bodies: dict[str, str]) -> list[dict[str, np.ndarray]]:
    """Per environment, each arm's end body pose (4x4) relative to its environment origin (``bodies``: arm -> body)."""
    from scipy.spatial.transform import Rotation

    import warp as wp

    robot = env.scene["robot"]
    origins = _origins(env).detach().cpu().numpy().astype(np.float64)
    values = wp.to_torch(robot.data.body_link_pose_w).detach().cpu().numpy().astype(np.float64)
    poses = [{} for _ in range(env.num_envs)]
    for arm, body in bodies.items():
        index = robot.body_names.index(body)
        rotations = Rotation.from_quat(values[:, index, 3:7]).as_matrix()  # xyzw
        for env_id in range(env.num_envs):
            T = np.eye(4)
            T[:3, :3] = rotations[env_id]
            T[:3, 3] = values[env_id, index, :3] - origins[env_id, :3]
            poses[env_id][arm] = T
    return poses


def gripper_openings(env: ManagerBasedEnv, arms) -> list[dict[str, float]]:
    """Per environment, each arm's gripper opening: its finger joints' mean angle over the open angle
    (``G2_GRIPPER_OPEN``; the inner joint negated), clipped to 1 open, 0 closed (``is_all_gripper_open``)."""
    import warp as wp

    from isaaclab_arena.embodiments.g2.g2 import G2_GRIPPER_OPEN

    robot = env.scene["robot"]
    q = wp.to_torch(robot.data.joint_pos).detach().cpu().numpy().astype(np.float64)
    fingers = {side: (outer, inner) for side, outer, inner in GRIPPER_JOINTS}
    openings = [{} for _ in range(env.num_envs)]
    for arm in arms:
        side = ARMS[arm]
        outer, inner = fingers[side]
        a = q[:, robot.joint_names.index(f"{outer}_gripper_{side}_outer_joint1")]
        b = -q[:, robot.joint_names.index(f"{inner}_gripper_{side}_inner_joint1")]
        values = np.clip((a + b) / 2 / G2_GRIPPER_OPEN, 0.0, 1.0)
        for env_id in range(env.num_envs):
            openings[env_id][arm] = float(values[env_id])
    return openings


def judges(env: ManagerBasedEnv, name: str) -> dict:
    """The environments' judges of a v2 task: env id -> ``judge`` and its start (``initial``, ``ee_start``)."""
    found = env.__dict__.setdefault("_g2_eval_judges", {})
    if len(found) < env.num_envs:
        spec = pack.load_task(name)["task"]["success_check"]
        for env_id in range(env.num_envs):
            found.setdefault(env_id, {"judge": Judge(spec), "initial": None, "ee_start": None})
    return found


def judge_outcomes(env: ManagerBasedEnv, name: str) -> list[str | None]:
    """Per environment, its judge's outcome after this control step (``success``, ``failed`` or None).

    Each judge is updated once per control step (the success and failure terms share it): with the objects' poses,
    the arms' end bodies and gripper openings; the first update of an episode also records its start.
    """
    step = int(env.common_step_counter)
    cached = env.__dict__.get("_g2_eval_outcomes")
    if cached is not None and cached[0] == step:
        return cached[1]
    spec = pack.load_task(name)["task"]["success_check"]
    labels = [label for label, entry in spec["objects"].items() if not entry.get("pose")]
    poses = object_poses(env, labels)
    ends = ee_poses(env, spec["ee_bodies"])
    openings = gripper_openings(env, spec["ee_bodies"])
    outcomes = []
    for env_id, state in sorted(judges(env, name).items()):
        if state["initial"] is None:
            state["initial"], state["ee_start"] = poses[env_id], ends[env_id]
        judge = state["judge"]
        outcomes.append(
            judge.update(poses[env_id], ends[env_id], state["ee_start"], state["initial"], openings[env_id])
        )
    env.__dict__["_g2_eval_outcomes"] = (step, outcomes)
    return outcomes


def check_succeeded(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Whether a v2 task's judge decided ``success``, per environment."""
    return torch.tensor([o == "success" for o in judge_outcomes(env, name)], dtype=torch.bool, device=env.device)


def check_failed(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Whether a v2 task's judge decided ``failed`` (a query counted past its aim), per environment."""
    return torch.tensor([o == "failed" for o in judge_outcomes(env, name)], dtype=torch.bool, device=env.device)


def object_states(env: ManagerBasedEnv, task: dict) -> list[dict]:
    """Per environment, what the legacy success relations read besides poses: each object's linear and angular
    velocity and the contact force of each ``placed`` pair (``success.evaluate``'s ``state``)."""
    import warp as wp

    states = [{"lin_vel": {}, "ang_vel": {}, "contact_n": {}} for _ in range(env.num_envs)]
    for obj in task["objects"]:
        data = env.scene[obj].data
        lin = wp.to_torch(data.root_lin_vel_w).detach().cpu().numpy()
        ang = wp.to_torch(data.root_ang_vel_w).detach().cpu().numpy()
        for env_id, state in enumerate(states):
            state["lin_vel"][obj], state["ang_vel"][obj] = lin[env_id].tolist(), ang[env_id].tolist()
    for obj, target in contact_pairs(task):
        # As PickAndPlaceTask's object_on_destination: one sensed body filtered against one target body.
        force = wp.to_torch(env.scene[contact_sensor_name(obj, target)].data.force_matrix_w)
        assert force.shape[1] == 1 and force.shape[2] == 1, f"contact sensor of {obj} on {target}: {force.shape}"
        norms = torch.linalg.vector_norm(force, dim=-1).reshape(-1).detach().cpu().numpy()
        for env_id, state in enumerate(states):
            state["contact_n"][f"{obj}>{target}"] = float(norms[env_id])
    return states


def grippers_open(env: ManagerBasedEnv, minimum_rad: float) -> torch.Tensor:
    """Whether both grippers are open past ``minimum_rad``, per environment (the clean-workcell collection's check)."""
    import warp as wp

    robot = env.scene["robot"]
    q = wp.to_torch(robot.data.joint_pos)
    result = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for side, outer, inner in GRIPPER_JOINTS:
        result &= q[:, robot.joint_names.index(f"{outer}_gripper_{side}_outer_joint1")] > minimum_rad
        result &= q[:, robot.joint_names.index(f"{inner}_gripper_{side}_inner_joint1")] < -minimum_rad
    return result


def relations_hold(env: ManagerBasedEnv, name: str) -> torch.Tensor:
    """Whether every success relation of the task holds now (and the grippers are open if it asks), per environment."""
    loaded = pack.load_task(name)
    task, geometry = loaded["task"], loaded["geometry"]
    poses = object_poses(env, list(task["objects"]))
    states = object_states(env, task)
    holds = [
        all(r["holds"] for r in evaluate(task["success"], geometry, pose, state)) for pose, state in zip(poses, states)
    ]
    result = torch.tensor(holds, dtype=torch.bool, device=env.device)
    if task.get("grippers_open_min_rad") is not None:
        result &= grippers_open(env, task["grippers_open_min_rad"])
    return result


def task_success(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Whether the task succeeded, per environment: its relations hold, for ``success_hold_s`` when it sets one."""
    holds = relations_hold(env, name)
    hold_s = pack.load_task(name)["task"].get("success_hold_s")
    if not hold_s:
        return holds
    # Consecutive control steps the relations held, restarted with each episode (the clean-workcell collection).
    count = env.__dict__.setdefault(
        "_g2_eval_held_steps", torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    )
    count[env.episode_length_buf <= 1] = 0
    count[:] = torch.where(holds, count + 1, 0)
    return count * env.step_dt >= hold_s


def object_dropped(env: ManagerBasedRLEnv, objects: list[str], minimum_z: float) -> torch.Tensor:
    """Whether any of ``objects`` fell below ``minimum_z`` (task frame), per environment."""
    dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for env_id, poses in enumerate(object_poses(env, objects)):
        dropped[env_id] = any(T[2, 3] < minimum_z for T in poses.values())
    return dropped


@configclass
class G2EvalTerminationsCfg:
    """Termination terms of a G2-Eval task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING

    check_failed: TerminationTermCfg | None = None
    """A v2 task: its success check decided the episode failed."""


@configclass
class G2EvalEventsCfg:
    """Event terms of a G2-Eval task."""

    reset_layout: EventTermCfg = MISSING
