# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Robot-independent skill/stage intervals indexed by completed control transitions."""

import copy
import json
import math

SCHEMA_VERSION = "arena.skill_trace.v1"


class SkillTrace:
    """Record one actor/environment stream; parallel actors use separate synchronized traces."""

    def __init__(self, task_id, step_dt, actor_id="robot", env_id=0, source="script"):
        assert math.isfinite(step_dt) and step_dt > 0
        self.task_id = task_id
        self.step_dt = step_dt
        self.actor_id = actor_id
        self.env_id = env_id
        self.source = source
        self.skills = []
        self.stages = []
        self.events = []
        self.step_skill_ids = []
        self.step_stage_ids = []
        self.skill = None
        self.current_stage = None
        self.pending = None
        self.finished = False

    @property
    def step(self):
        return len(self.step_stage_ids)

    def start_skill(self, name, entities=None, parameters=None):
        """Begin a uniquely numbered skill instance; names may repeat across objects."""
        assert not self.finished and self.skill is None and self.pending is None
        self.skill = len(self.skills)
        self.skills.append({
            "id": self.skill,
            "name": name,
            "actor_id": self.actor_id,
            "entities": copy.deepcopy(entities or {}),
            "parameters": copy.deepcopy(parameters or {}),
            "start_step": self.step,
            "end_step": None,
            "execution_status": "running",
            "success": None,
        })
        return self.skill

    def stage(self, name):
        """Begin a stage at the next action; close the previous executed stage as completed."""
        assert not self.finished and self.skill is not None and self.pending is None
        self._close_stage("completed")
        self.current_stage = len(self.stages)
        self.stages.append({
            "id": self.current_stage,
            "skill_instance_id": self.skill,
            "name": name,
            "start_step": self.step,
            "end_step": None,
            "execution_status": "running",
        })

    def _close_stage(self, status):
        if self.current_stage is not None:
            self.stages[self.current_stage].update(end_step=self.step, execution_status=status)
            self.current_stage = None

    def event(self, name, details=None):
        """Attach an instantaneous event to a control boundary; it consumes no frame."""
        self.events.append({
            "name": name,
            "step": self.step,
            "skill_instance_id": self.skill,
            "stage_instance_id": self.current_stage,
            "details": copy.deepcopy(details or {}),
        })

    def reject_stage(self, reason):
        """Preserve a rejected candidate, including any motion already executed."""
        assert self.pending is None
        self.event("candidate_rejected", {"reason": str(reason)})
        self._close_stage("failed")

    def before_step(self):
        """Snapshot labels before env.step; commit only after that transition returns."""
        assert not self.finished and self.pending is None
        assert self.skill is not None and self.current_stage is not None, "Every recorded action needs a stage"
        self.pending = (self.skill, self.current_stage)

    def after_step(self):
        """Commit exactly one completed control transition."""
        assert self.pending is not None
        skill, stage = self.pending
        self.step_skill_ids.append(skill)
        self.step_stage_ids.append(stage)
        self.pending = None

    def abort_step(self, reason):
        """Mark an interrupted transition without claiming a complete observation/video pair."""
        assert self.pending is not None
        self.event("incomplete_transition", {"reason": str(reason)})
        self.pending = None

    def end_skill(self, success=None, evidence=None, status="completed"):
        """Close execution; semantic success stays unknown unless supplied with measured evidence."""
        assert self.skill is not None and self.pending is None
        assert status in {"completed", "failed", "interrupted"}
        assert success is None or isinstance(success, bool)
        if success is not None:
            assert evidence is not None, "A semantic outcome needs evidence"
        self._close_stage(status)
        self.skills[self.skill].update(
            end_step=self.step, execution_status=status, success=success, evidence=copy.deepcopy(evidence)
        )
        self.skill = None

    def finish(self, success, error=None):
        """Seal the episode; completed children do not inherit episode success automatically."""
        assert not self.finished and self.pending is None
        if self.skill is not None:
            self.end_skill(status="failed" if error or not success else "completed")
        self.finished = True
        document = {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "actor_id": self.actor_id,
            "env_id": self.env_id,
            "source": self.source,
            "step_dt_s": self.step_dt,
            "num_steps": self.step,
            "interval_convention": "[start_step,end_step)",
            "alignment": {
                "labels": "action[t]: observation[t] -> post_state[t] (next observation)",
                "live_preview_frame": "frame[t] is post_state[t], at simulation time (t+1)*step_dt_s",
                "video_seek": "frame_index * step_dt_s; playback begins at first post-step frame",
            },
            "episode_success": bool(success),
            "episode_error": error,
            "skills": self.skills,
            "stages": self.stages,
            "events": self.events,
            "step_skill_ids": self.step_skill_ids,
            "step_stage_ids": self.step_stage_ids,
        }
        validate_trace(document)
        return copy.deepcopy(document)


def validate_trace(document, expected_steps=None):
    """Validate contiguous action coverage, hierarchy and half-open bounds."""
    assert document["schema_version"] == SCHEMA_VERSION
    count = document["num_steps"]
    assert type(count) is int and count >= 0
    assert math.isfinite(document["step_dt_s"]) and document["step_dt_s"] > 0
    if expected_steps is not None:
        assert count == expected_steps, f"Annotation/action mismatch: {count} != {expected_steps}"
    skills, stages = document["skills"], document["stages"]
    assert len(document["step_skill_ids"]) == len(document["step_stage_ids"]) == count
    for index, skill in enumerate(skills):
        assert skill["id"] == index and 0 <= skill["start_step"] <= skill["end_step"] <= count
    for index, stage in enumerate(stages):
        assert stage["id"] == index
        assert type(stage["skill_instance_id"]) is int and 0 <= stage["skill_instance_id"] < len(skills)
        parent = skills[stage["skill_instance_id"]]
        assert parent["start_step"] <= stage["start_step"] <= stage["end_step"] <= parent["end_step"]
    for step, (skill_id, stage_id) in enumerate(zip(document["step_skill_ids"], document["step_stage_ids"])):
        assert type(skill_id) is int and 0 <= skill_id < len(skills)
        assert type(stage_id) is int and 0 <= stage_id < len(stages)
        stage = stages[stage_id]
        assert stage["skill_instance_id"] == skill_id
        assert stage["start_step"] <= step < stage["end_step"]
    for event in document["events"]:
        assert 0 <= event["step"] <= count
        skill_id, stage_id = event["skill_instance_id"], event["stage_instance_id"]
        assert skill_id is None or (type(skill_id) is int and 0 <= skill_id < len(skills))
        assert stage_id is None or (type(stage_id) is int and 0 <= stage_id < len(stages))
        if stage_id is not None:
            assert stages[stage_id]["skill_instance_id"] == skill_id


def write_hdf5_trace(demo, document, expected_steps):
    """Add portable trace metadata and per-action IDs to an HDF5 episode group."""
    import h5py
    import numpy as np

    validate_trace(document, expected_steps)
    assert "annotations" not in demo, "Do not overwrite existing annotations"
    group = demo.create_group("annotations")
    metadata = {k: v for k, v in document.items() if k not in {"step_skill_ids", "step_stage_ids"}}
    group.create_dataset("metadata_json", data=json.dumps(metadata, ensure_ascii=False), dtype=h5py.string_dtype())
    for name in ("step_skill_ids", "step_stage_ids"):
        group.create_dataset(name, data=np.asarray(document[name], dtype=np.int32))
    group.attrs["schema_version"] = SCHEMA_VERSION
