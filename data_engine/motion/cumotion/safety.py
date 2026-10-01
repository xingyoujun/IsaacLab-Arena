# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent sphere checks for planned commands and measured robot bodies."""

import numpy as np
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation


class MotionSafetyError(RuntimeError):
    """Retain structured evidence when a motion must not execute."""

    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(str(evidence))


def joint_samples(waypoints, max_step_rad=0.02):
    """Include endpoints and bound the largest joint increment between checked samples."""
    points = np.asarray(waypoints, dtype=np.float64)
    assert points.ndim == 2 and len(points) and np.isfinite(points).all()
    assert max_step_rad > 0
    yield points[0]
    for start, end in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / max_step_rad)))
        for index in range(1, count + 1):
            yield start + (end - start) * (index / count)


def line_error(position, start, end):
    """Return distance to the finite Cartesian approach segment, in metres."""
    vector = end - start
    scale = float(np.dot(vector, vector))
    alpha = np.clip(np.dot(position - start, vector) / scale, 0, 1) if scale > 1e-12 else 0
    return float(np.linalg.norm(position - (start + alpha * vector)))


class MotionSafety:
    """Reject overlapping robot spheres independently of the planner's success flag."""

    def __init__(self, planner):
        from data_engine.motion.cumotion.robot_description import import_cumotion

        self.planner = planner
        self.inspector = import_cumotion().create_robot_world_inspector(planner.robot_description)
        self.actual = None
        self.path_constraint = None
        self.report = dict(planned_samples=0, measured_samples=0, max_joint_step_rad=0.02, failures=[])

    def reject(self, **evidence):
        self.report["failures"].append(evidence)
        raise MotionSafetyError(evidence)

    def check_configuration(self, q, stage="planned"):
        q = np.asarray(q, dtype=np.float64)
        limits = self.planner.joint_limits
        if not np.isfinite(q).all() or np.any(q < limits[:, 0] - 1e-4) or np.any(q > limits[:, 1] + 1e-4):
            self.reject(stage=stage, reason="joint_limits", joints=q.tolist())
        pairs = list(self.inspector.frames_in_self_collision(q.reshape(-1, 1)))
        if pairs:
            self.reject(stage=stage, reason="self_collision", pairs=pairs, joints=q.tolist())
        self.report["planned_samples"] += 1
        if self.path_constraint is not None:
            constraint = self.path_constraint
            pose = self.planner.kinematics.pose(q, self.planner.cfg.tool_frame)
            base_rotation = Rotation.from_quat(self.planner.base_quat_wxyz[[1, 2, 3, 0]])
            position = self.planner.base_pos + base_rotation.apply(np.array(pose.translation, copy=True).reshape(3))
            rotation = base_rotation * Rotation.from_matrix(np.array(pose.rotation.matrix(), copy=True))
            distance = line_error(position, constraint["start"], constraint["end"])
            angle = float((constraint["rotation"].inv() * rotation).magnitude())
            if distance > constraint["position_tolerance_m"] or angle > constraint["angle_tolerance_rad"]:
                self.reject(stage=stage, reason="cartesian_corridor", deviation_m=distance, angle_rad=angle)

    def check_path(self, points, stage="planned_path"):
        for q in joint_samples(points, self.report["max_joint_step_rad"]):
            self.check_configuration(q, stage)

    def check_actual(self):
        if self.actual is None:
            self.actual = MeasuredSphereGuard(self.planner)
        pairs, clearance = self.actual.check()
        self.report["measured_samples"] += 1
        self.report["minimum_measured_clearance_m"] = min(
            self.report.get("minimum_measured_clearance_m", float("inf")), clearance
        )
        if pairs:
            self.reject(stage="measured", reason="self_collision", pairs=pairs, clearance_m=clearance)
        if self.path_constraint is not None:
            # Measured active joints also remain inside the same Cartesian corridor.
            self.check_configuration(self.planner.joint_positions(), "measured_cartesian")


class MeasuredSphereGuard:
    """Transform collision spheres using measured USD body poses and check pairs on the env device."""

    def __init__(self, planner):
        import torch

        self.robot = planner.robot
        self.device = planner.env.device
        cfg = planner.cfg
        description = yaml.safe_load(Path(cfg.lula_robot_description).read_text())
        body_names = list(self.robot.data.body_names)
        parents = {j.find("child").attrib["link"]: j for j in ET.parse(cfg.robot_urdf).getroot().findall("joint")}
        aliases = {cfg.tool_frame: cfg.sim_tool_body} if cfg.sim_tool_body else {}
        self.names, body_ids, centers, radii = [], [], [], []
        for entry in description["collision_spheres"]:
            for link, spheres in entry.items():
                current, translation, rotation = link, np.zeros(3), Rotation.identity()
                while aliases.get(current, current) not in body_names:
                    assert current in parents, f"No measured collision body for {link}"
                    joint = parents[current]
                    assert joint.attrib["type"] == "fixed", f"Unmapped moving collision frame: {link}/{current}"
                    origin = joint.find("origin")
                    offset = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                    parent_rotation = Rotation.from_euler("xyz", rpy)
                    translation = offset + parent_rotation.apply(translation)
                    rotation = parent_rotation * rotation
                    current = joint.find("parent").attrib["link"]
                body = body_names.index(aliases.get(current, current))
                for sphere in spheres:
                    if sphere["radius"] <= 0:
                        continue
                    self.names.append(link)
                    body_ids.append(body)
                    centers.append(translation + rotation.apply(sphere["center"]))
                    radii.append(sphere["radius"])
        first, second = [], []
        for i, name in enumerate(self.names):
            for j in range(i + 1, len(self.names)):
                other = self.names[j]
                if (
                    name == other
                    or other in cfg.self_collision_ignore.get(name, ())
                    or name in cfg.self_collision_ignore.get(other, ())
                ):
                    continue
                first.append(i)
                second.append(j)
        self.body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        self.centers = torch.tensor(np.array(centers), device=self.device, dtype=torch.float32)
        self.radii = torch.tensor(radii, device=self.device, dtype=torch.float32)
        self.first = torch.tensor(first, device=self.device, dtype=torch.long)
        self.second = torch.tensor(second, device=self.device, dtype=torch.long)

    def check(self):
        import torch

        position, quaternion = self.robot.data.body_pos_w, self.robot.data.body_quat_w
        position = position.torch if hasattr(position, "torch") else position
        quaternion = quaternion.torch if hasattr(quaternion, "torch") else quaternion
        q = quaternion[0, self.body_ids]  # Isaac Lab 3.0: xyzw.
        vector = 2 * torch.cross(q[:, :3], self.centers, dim=1)
        centers = position[0, self.body_ids] + self.centers + q[:, 3:] * vector + torch.cross(q[:, :3], vector, dim=1)
        distances = torch.linalg.vector_norm(centers[self.first] - centers[self.second], dim=1)
        clearances = distances - self.radii[self.first] - self.radii[self.second]
        if not len(clearances):
            return [], 1e6
        collisions = torch.nonzero(clearances < 0).flatten()[:8].cpu().tolist()
        pairs = [(self.names[int(self.first[i])], self.names[int(self.second[i])]) for i in collisions]
        return pairs, float(clearances.min())
