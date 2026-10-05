# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the test_assets_v0 manifest: Lightwheel, Infinigen and USDCraft assets per category and task.

Static only (no simulator). For every asset it records the source, the task, the target joint chosen
for that task and *how* it was chosen (declared by the asset, a name rule, or not resolvable), what
the asset itself says about open direction and handle, and a static estimate of the manual steps a
data engine would still need. Runtime results (Isaac load, renders) are added by render_assets.py.

    .venv/bin/python tools/asset_baselines/build_test_assets.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

LIGHTWHEEL = Path("/home/ubuntu/playground/Lightwheel_OpenSource/Manipulation")
INFINIGEN = Path("/home/ubuntu/playground/baselines/infinigen_exports/usd")
USDCRAFT = Path("/home/ubuntu/code/usdcraft_v2")
USDCRAFT_BATCHES = [
    Path("/home/ubuntu/playground/baselines/test_assets_v0/usdcraft_gen_v0.log"),
    Path("/home/ubuntu/playground/baselines/test_assets_v0/usdcraft_gen_v0_retry1.log"),
]
USDCRAFT_CSV = Path("/home/ubuntu/playground/baselines/test_assets_v0/usdcraft_gen_v0.csv")
OUT = Path(__file__).resolve().parents[2] / "outputs/asset_baselines/test_assets_v0"

# Task per category; the success rule is the frozen matrix-plan threshold, shown for review only.
TASKS = {
    "microwave": dict(task="open_door", label="微波炉：开门", joint="revolute", success="门转到 min(行程 50%, 45°)"),
    "toaster_oven": dict(
        task="open_door", label="台面烤箱：开下翻门", joint="revolute", success="门转到 min(行程 50%, 45°)"
    ),
    "toaster": dict(task="press_lever", label="烤面包机：按下拉杆", joint="prismatic", success="拉杆按到行程 80%"),
    "drawer_cabinet": dict(
        task="open_drawer", label="抽屉柜：拉开抽屉", joint="prismatic", success="抽屉拉出 min(行程 50%, 0.10 m)"
    ),
    "cabinet_door": dict(task="open_door", label="小柜门：开门", joint="revolute", success="门转到 min(行程 50%, 45°)"),
}
# Rough longest-side priors (m) for a tabletop instance; out-of-range sizes are flagged, never changed.
SIZE_PRIOR = {
    "microwave": (0.40, 0.70),
    "toaster_oven": (0.33, 0.60),
    "toaster": (0.20, 0.42),
    "drawer_cabinet": (0.20, 0.70),
    "cabinet_door": (0.20, 0.70),
}
# Name rules a data engine could apply without a declaration (target joint only).
NAME_RULES = {
    "microwave": r"door|microjoint",
    "toaster_oven": r"door",
    "toaster": r"lever|slider|carriage",
    "drawer_cabinet": r"drawer",
    "cabinet_door": r"door|hinge",
}
LIGHTWHEEL_PICK = {"microwave": "Microwave", "toaster_oven": "ToasterOven", "toaster": "Toaster"}
INFINIGEN_PICK = {
    "microwave": ["microwave"],
    "toaster_oven": ["oven"],
    "toaster": ["toaster"],
    "drawer_cabinet": ["drawer"],
    "cabinet_door": ["cabinet"],
}
LICENSES = {"lightwheel": "CC BY-NC 4.0", "infinigen": "BSD-3-Clause (generator)", "usdcraft": "internal"}


def closure(path: Path) -> tuple[str, int, list[str]]:
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    files = sorted({layer.realPath for layer in layers if layer.realPath} | set(assets))
    digest = hashlib.sha256()
    for name in files:
        digest.update(Path(name).name.encode())
        digest.update(Path(name).read_bytes())
    return digest.hexdigest(), len(files), sorted(set(unresolved))


def inspect(path: Path) -> dict:
    stage = Usd.Stage.Open(str(path))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(stage.GetPseudoRoot())
    rng = box.ComputeAlignedRange()
    size = [round(v * mpu, 4) for v in (rng.GetMax() - rng.GetMin())] if not rng.IsEmpty() else None
    bodies = mass = density = colliders = material_density = 0
    invalid_topology = {"visual": 0, "collision": 0}
    approx: dict[str, int] = {}
    joints = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies += 1
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            if (api.GetMassAttr().Get() or 0) > 0:
                mass += 1
            elif (api.GetDensityAttr().Get() or 0) > 0:
                density += 1
        if prim.HasAPI(UsdPhysics.MaterialAPI) and (UsdPhysics.MaterialAPI(prim).GetDensityAttr().Get() or 0) > 0:
            material_density += 1
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            indices = mesh.GetFaceVertexIndicesAttr().Get() or []
            if sum(counts) != len(indices):
                invalid_topology["collision" if prim.HasAPI(UsdPhysics.CollisionAPI) else "visual"] += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders += 1
            kind = (
                str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                else ("mesh_default" if prim.IsA(UsdGeom.Mesh) else "primitive")
            )
            approx[kind] = approx.get(kind, 0) + 1
        if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
            joint = dict(
                name=prim.GetName(),
                path=str(prim.GetPath()),
                type=prim.GetTypeName().replace("Physics", "").replace("Joint", "").lower(),
            )
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if not attr.HasAuthoredValue():
                    continue
                if name in ("physics:lowerLimit", "physics:upperLimit", "physics:axis"):
                    value = attr.Get()
                    joint[name.split(":")[1]] = round(value, 4) if isinstance(value, float) else str(value)
                elif name.endswith("physics:stiffness") or name.endswith("physics:damping"):
                    joint["drive_" + name.rsplit(":", 1)[1]] = round(float(attr.Get()), 4)
                elif name in ("physxJoint:jointFriction",) or name.startswith("physxJointAxis:"):
                    joint[name] = round(float(attr.Get()), 4)
            joints.append(joint)
    root = stage.GetDefaultPrim()
    custom = dict(root.GetCustomData()) if root else {}
    sha, files, unresolved = closure(path)
    return dict(
        meters_per_unit=mpu,
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        size_m=size,
        rigid_bodies=bodies,
        mass_explicit=mass,
        density_only=density,
        material_density=material_density,
        invalid_mesh_topology=invalid_topology,
        colliders=colliders,
        collider_approximation=approx,
        joints=joints,
        declared_interactions=(
            json.loads(custom["usdcraft:interactions"]) if "usdcraft:interactions" in custom else None
        ),
        closure_sha256=sha,
        closure_files=files,
        unresolved=unresolved,
    )


def bind(category: str, static: dict, annotations: dict | None) -> dict:
    """Choose the task's target joint and say how; never guess a direction the asset does not state."""
    spec = TASKS[category]
    if annotations:
        kinds = {"revolute": ("rotate", "pull", "turn"), "prismatic": ("pull", "press", "push", "toggle")}[
            spec["joint"]
        ]
        # Prefer the declared opening/pressing operation over its inverse (e.g. close_door).
        ordered = sorted(
            annotations.get("interactions", []),
            key=lambda i: not re.search(r"open|press|push_down|lower", i.get("name", ""), re.I),
        )
        for item in ordered:
            joint = item.get("joint") or {}
            if joint.get("type") == spec["joint"] and item.get("kind") in kinds and joint.get("target") is not None:
                if spec["task"] == "press_lever" and item["kind"] not in ("press", "push", "toggle"):
                    continue
                return dict(
                    joint=joint.get("name"),
                    derivation="declared",
                    interaction=item.get("name"),
                    kind=item.get("kind"),
                    open_direction="declared",
                    start=joint.get("start"),
                    target=joint.get("target"),
                    limits=joint.get("limits"),
                    unit=joint.get("unit"),
                    handle="declared",
                    handle_point=(item.get("pose") or {}).get("position_m"),
                    width_m=(item.get("pose") or {}).get("width_m"),
                )
    candidates = [
        j for j in static["joints"] if j["type"] == spec["joint"] and re.search(NAME_RULES[category], j["name"], re.I)
    ]
    if candidates:
        joint = candidates[0]
        return dict(
            joint=joint["name"],
            derivation="name_rule",
            ambiguous=len(candidates) > 1,
            open_direction="unknown",
            lower=joint.get("lowerLimit"),
            upper=joint.get("upperLimit"),
            handle="none",
        )
    return dict(joint=None, derivation="unresolved", open_direction="unknown", handle="none")


def manual_steps(category: str, static: dict, binding: dict) -> list[dict]:
    """Static estimate of what a data engine still has to supply before task-ready."""
    steps = []
    low, high = SIZE_PRIOR[category]
    if static["size_m"] and not low <= max(static["size_m"]) <= high:
        steps.append(dict(step="scale", reason=f"最长边 {max(static['size_m']):.3f} m 不在品类先验 {low}–{high} m"))
    if binding["derivation"] == "unresolved":
        steps.append(dict(step="bind_joint", reason="没有声明，名称规则也找不到目标关节"))
    elif binding.get("ambiguous"):
        steps.append(dict(step="bind_joint", reason="名称规则命中多个关节，需要人工选择"))
    if binding["open_direction"] == "unknown":
        steps.append(dict(step="open_direction", reason="资产没有声明哪一端是开"))
    if binding["handle"] == "none":
        steps.append(dict(step="handle", reason="资产没有声明把手/接触点（只能用检测器或人工标注）"))
    bad = static["invalid_mesh_topology"]
    if bad["visual"] or bad["collision"]:
        steps.append(
            dict(
                step="fix_topology",
                reason=f"网格拓扑无效（faceVertexCounts 与索引数不符）：可视 {bad['visual']}、碰撞 {bad['collision']}",
            )
        )
    if static["colliders"] == 0:
        steps.append(dict(step="colliders", reason="没有任何碰撞体"))
    if static["mass_explicit"] == 0 and static["density_only"] == 0 and static["material_density"] == 0:
        steps.append(dict(step="mass", reason="没有质量或密度（PhysX 将使用默认值）"))
    return steps


def ingest_fixed_topology(path: Path) -> Path | None:
    """Write a sibling copy whose triangle meshes get one face count per triangle; record, never hide."""
    stage = Usd.Stage.Open(str(path))
    fixed = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if sum(counts) != len(indices) and set(counts) == {3} and len(indices) % 3 == 0:
            mesh.GetFaceVertexCountsAttr().Set([3] * (len(indices) // 3))
            fixed += 1
    if not fixed:
        return None
    target = path.with_name(path.stem + ".ingest_fixed_topology.usda")
    stage.GetRootLayer().Export(str(target))
    return target


def usdcraft_rows() -> list[dict]:
    categories = {row["row_id"]: row for row in csv.DictReader(USDCRAFT_CSV.open())}
    found = {}
    for log in (log for log in USDCRAFT_BATCHES if log.is_file()):
        for line in log.read_text().splitlines():
            match = re.search(r'row finished.*row_id="([^"]+)" \| status=generation_ready \| record_id="([^"]+)"', line)
            if match:
                found[match[1]] = match[2]
    rows = []
    for row_id, record in sorted(found.items()):
        category = row_id.rsplit("_", 1)[0]
        rows.append(dict(row_id=row_id, record=record, category=category, prompt=categories[row_id]["prompt"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()
    entries = []

    for category, prefix in LIGHTWHEEL_PICK.items():
        names = sorted(d.name for d in LIGHTWHEEL.iterdir() if re.fullmatch(prefix + r"\d+", d.name))[
            : args.per_category
        ]
        for name in names:
            path = next((LIGHTWHEEL / name).glob("*.usd"))
            entries.append(
                dict(
                    asset_id=f"lightwheel/{name}",
                    source="lightwheel",
                    category=category,
                    usd=str(path),
                    origin=dict(pick="前 N 个（按编号排序）"),
                )
            )

    for category, kinds in INFINIGEN_PICK.items():
        for kind in kinds:
            for seed_dir in (
                sorted((INFINIGEN / kind).glob("*"))[: args.per_category] if (INFINIGEN / kind).is_dir() else []
            ):
                usd = seed_dir / f"{kind}.usda"
                if usd.is_file():
                    meta = (
                        json.loads((seed_dir / "metadata.json").read_text())
                        if (seed_dir / "metadata.json").is_file()
                        else {}
                    )
                    entries.append(
                        dict(
                            asset_id=f"infinigen/{kind}_{seed_dir.name}",
                            source="infinigen",
                            category=category,
                            usd=str(usd),
                            origin=dict(
                                generator=kind,
                                seed=int(seed_dir.name),
                                proxy=kind == "oven" and category == "toaster_oven",
                                joint_labels=[
                                    v.get("joint label")
                                    for v in meta.values()
                                    if isinstance(v, dict) and "joint label" in v
                                ],
                            ),
                        )
                    )

    for row in usdcraft_rows():
        usd = USDCRAFT / "data/cache/record_materialization" / row["record"] / "isaac/model.usdc"
        if usd.is_file():
            entries.append(
                dict(
                    asset_id=f"usdcraft/{row['row_id']}",
                    source="usdcraft",
                    category=row["category"],
                    usd=str(usd),
                    origin=dict(record_id=row["record"], prompt=row["prompt"], input="text"),
                )
            )

    for entry in entries:
        static = inspect(Path(entry["usd"]))
        annotations = None
        if entry["source"] == "usdcraft":
            sidecar = (
                USDCRAFT / "data/cache/record_training" / entry["origin"]["record_id"] / "interaction_annotations.json"
            )
            annotations = json.loads(sidecar.read_text()) if sidecar.is_file() else None
        entry["task"] = TASKS[entry["category"]]
        entry["license"] = LICENSES[entry["source"]]
        entry["static"] = {k: v for k, v in static.items() if k != "declared_interactions"}
        entry["declared_interaction_count"] = len(annotations.get("interactions", [])) if annotations else 0
        entry["binding"] = bind(entry["category"], static, annotations)
        entry["manual_steps_estimate"] = manual_steps(entry["category"], static, entry["binding"])
        if any(step["step"] == "fix_topology" for step in entry["manual_steps_estimate"]):
            fixed = ingest_fixed_topology(Path(entry["usd"]))
            if fixed:
                entry["ingest"] = dict(
                    usd=str(fixed),
                    rewrites=[
                        "triangle meshes: faceVertexCounts reset to one count per triangle (source exporter bug)"
                    ],
                    closure_sha256=closure(fixed)[0],
                )

    # Keep runtime results only for byte-identical assets; any change requires a new Isaac run.
    previous = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").is_file() else {}
    kept = {
        (e["asset_id"], e["static"]["closure_sha256"]): e["runtime"]
        for e in previous.get("entries", [])
        if e.get("runtime")
    }
    for entry in entries:
        runtime = kept.get((entry["asset_id"], entry["static"]["closure_sha256"]))
        if runtime:
            entry["runtime"] = runtime
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema="asset_baselines.test_assets.v0",
        note="静态检查；人工步骤为静态估计，运行时结果另行记录",
        tasks=TASKS,
        entries=entries,
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
    by_source: dict[str, list[int]] = {}
    for entry in entries:
        by_source.setdefault(entry["source"], []).append(len(entry["manual_steps_estimate"]))
    for source, counts in by_source.items():
        print(f"{source}: {len(counts)} assets, manual steps (static estimate) mean {sum(counts) / len(counts):.2f}")
    print(OUT / "manifest.json")


if __name__ == "__main__":
    main()
