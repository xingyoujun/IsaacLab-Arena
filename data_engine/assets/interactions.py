# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read and audit body-local USDCraft interaction candidates without starting simulation."""

import hashlib
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_interactions(usd_path):
    """Load a hash-bound sidecar, or return None for an unannotated legacy asset."""
    usd_path = Path(usd_path)
    path = usd_path.with_name("interaction_annotations.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    assert data["schema"] == "usdcraft.training_interactions" and data["schema_version"] == 1
    assert data["source"]["usd_sha256"] == hashlib.sha256(usd_path.read_bytes()).hexdigest(), "Stale annotation"
    assert data["conventions"]["quat_order"] == "wxyz"
    assert data["conventions"]["meters_per_unit"] == 1 and data["conventions"]["up_axis"] == "Z"
    assert len({item["name"] for item in data["interactions"]}) == len(data["interactions"])
    for rule in data.get("latches", []):
        assert rule["release_when"] == "release_joint >= release_at", "Unsupported latch predicate"
        assert isinstance(rule["initially_engaged"], bool)
        assert rule["engage_tolerance"] > 0
        assert np.isfinite([rule["hold"], rule["release_at"], rule["engage_tolerance"]]).all()
        assert rule["joint"]["name"] != rule["release_joint"]["name"], "A latch needs two distinct joints"
    return data


def interaction(data, name):
    """Select one named candidate; never substitute a different interaction silently."""
    return next(item for item in data["interactions"] if item["name"] == name)


def world_pose(candidate, T_W_L):
    """Compose a measured body pose (xyz, xyzw) with the annotated interaction frame."""
    pose = candidate["pose"]
    R_W_L = Rotation.from_quat(T_W_L[3:])
    q = pose["quat_wxyz"]
    R_L_G = Rotation.from_quat([*q[1:], q[0]])
    return np.r_[np.asarray(T_W_L[:3]) + R_W_L.apply(pose["position_m"]), (R_W_L * R_L_G).as_quat()]


def audit_asset(usd_path, jaw_width_m=0.085):  # noqa: C901
    """Check annotations against composed USD; report geometric candidates separately from task readiness."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    data = load_interactions(usd_path)
    assert data is not None, f"Missing interaction annotation: {usd_path}"
    stage = Usd.Stage.Open(str(usd_path))
    assert stage
    root = stage.GetDefaultPrim()
    embedded = json.loads(root.GetCustomDataByKey("usdcraft:interactions") or "[]")
    embedded = {item["name"]: item for item in embedded}
    links = {item["body"]: item for item in data["asset"]["links"]}
    result = dict(asset=str(usd_path), errors=[], warnings=[], interactions=[], runtime_validated=False)
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != "Z":
        result["errors"].append("USD must use metres and Z up")
    for item in data["interactions"]:
        errors, warnings = [], list(item.get("issues", []))
        pose = item["pose"]
        link = links[item["body"]]
        body = stage.GetPrimAtPath(link["prim_path"])
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            errors.append("Body does not resolve to a rigid body")
        raw = embedded.get(item["name"])
        if raw is None:
            errors.append("Missing embedded USD interaction")
        else:
            for field, value in (
                ("point", pose["position_m"]),
                ("approach", pose["approach_axis"]),
                ("closing", pose["closing_axis"]),
                ("width", pose.get("width_m")),
            ):
                actual = raw.get(field)
                same = (
                    actual == value if value is None else actual is not None and np.allclose(actual, value, atol=2e-6)
                )
                if not same:
                    errors.append(f"Embedded USD/sidecar mismatch: {field}")
            if raw["part_prim"] != link["prim_path"] or raw["kind"] != item["kind"]:
                errors.append("Embedded body/kind mismatch")
        q = np.asarray(pose["quat_wxyz"])
        if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 2e-5:
            errors.append("Invalid quaternion")
        else:
            matrix = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
            if not np.allclose(matrix[:, 2], pose["approach_axis"], atol=2e-5):
                errors.append("Pose +Z does not match approach")
            if pose["closing_axis"] is not None and not np.allclose(matrix[:, 1], pose["closing_axis"], atol=2e-5):
                errors.append("Pose +Y does not match closing axis")
        width = pose.get("width_m")
        if width is not None:
            if not np.isfinite(width) or width <= 0:
                errors.append("Invalid pinch width")
            elif width > jaw_width_m:
                warnings.append("Pinch wider than Robotiq 2F-85 opening")
            contacts = np.asarray(item.get("contacts_m", []))
            if contacts.shape == (2, 3):
                if not np.allclose(contacts.mean(axis=0), pose["position_m"], atol=2e-6):
                    errors.append("Contact midpoint differs from grasp origin")
                if abs(np.linalg.norm(contacts[1] - contacts[0]) - width) > 2e-6:
                    errors.append("Contact separation differs from width")
        point = np.asarray(pose["position_m"])
        lo, hi = np.asarray(link["aabb_in_body_m"])
        if not np.isfinite(point).all() or np.any(point < lo - 0.003) or np.any(point > hi + 0.003):
            warnings.append("Interaction point outside body AABB by more than 3 mm")
        joint = item.get("joint")
        if joint:
            prim = stage.GetPrimAtPath(joint["prim"])
            schema = UsdPhysics.RevoluteJoint if joint["type"] == "revolute" else UsdPhysics.PrismaticJoint
            if not prim.IsA(schema):
                errors.append("Joint type/path mismatch")
            else:
                usd_joint = schema(prim)
                scale = np.pi / 180 if joint["type"] == "revolute" else 1
                limits = np.array([usd_joint.GetLowerLimitAttr().Get(), usd_joint.GetUpperLimitAttr().Get()]) * scale
                if not np.allclose(limits, joint["limits"], atol=2e-6):
                    errors.append("USD/annotation joint limits differ")
                for value in (joint["start"], joint["target"]):
                    if not np.isfinite(value) or not limits[0] - 2e-6 <= value <= limits[1] + 2e-6:
                        errors.append("Joint target/start outside limits")
                body1 = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
                if body1 != [body.GetPath()]:
                    errors.append("Interaction not bound to joint child body")
                axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[usd_joint.GetAxisAttr().Get()]
                local_rotation = UsdPhysics.Joint(prim).GetLocalRot1Attr().Get()
                direction = Gf.Rotation(Gf.Quatd(local_rotation)).TransformDir(Gf.Vec3d(*axis))
                if not np.allclose(direction, joint["axis_in_body"], atol=2e-5):
                    errors.append("Joint axis differs from USD child-local axis")
            if raw and (raw["joint_prim"] != joint["prim"] or raw["joint"] != joint["name"]):
                errors.append("Embedded joint mismatch")
            success = item.get("success", {})
            for latch in data.get("latches", []):
                if latch["release_joint"]["name"] == joint["name"]:
                    if success.get("target", 0) - success.get("tolerance", 0) < latch["release_at"]:
                        warnings.append("Interaction success tolerance permits success before latch release")
        result["interactions"].append(
            dict(
                name=item["name"],
                kind=item["kind"],
                body=item["body"],
                errors=errors,
                warnings=warnings,
                jaw_compatible=width is None or width <= jaw_width_m,
            )
        )
        result["errors"].extend(f"{item['name']}: {error}" for error in errors)
        result["warnings"].extend(f"{item['name']}: {warning}" for warning in warnings)
    latches = json.loads(root.GetCustomDataByKey("usdcraft:latches") or "[]")
    for latch in data.get("latches", []):
        raw = next((item for item in latches if item["name"] == latch["name"]), None)
        if raw is None or any(raw.get(key) != latch[key] for key in ("hold", "release_at")):
            result["errors"].append(f"Latch mismatch: {latch['name']}")
        result["warnings"].append(
            f"{latch['name']}: USD metadata needs a runtime latch controller; sidecar supplies reset/tolerance"
            " semantics"
        )
    result["status"] = "fail" if result["errors"] else "pass_with_warnings" if result["warnings"] else "pass"
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from data_engine.assets.manage import verify

    manifest = verify(args.assets)
    reports = {}
    for name, entry in manifest["entries"].items():
        path = args.assets / entry["path"]
        if path.with_name("interaction_annotations.json").is_file():
            reports[name] = audit_asset(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    print({key: (value["status"], len(value["interactions"])) for key, value in reports.items()})
    assert not any(report["errors"] for report in reports.values()), f"See {args.output}"


if __name__ == "__main__":
    main()
