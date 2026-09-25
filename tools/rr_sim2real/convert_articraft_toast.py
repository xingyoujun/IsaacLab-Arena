# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export the Articraft toaster, uniformly matching USDcraft's measured X width."""

import argparse
import pathlib
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True, type=pathlib.Path)
parser.add_argument("--output-dir", required=True, type=pathlib.Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Usd  # noqa: E402

scale = 0.143194 / 0.18
args.output_dir.mkdir(parents=True, exist_ok=True)
destination = args.output_dir / "articraft_toast.usd"
assert not destination.exists(), f"Refusing to overwrite {destination}"
tree = ET.parse(args.source)
for element in tree.iter():
    if element.tag == "origin" and "xyz" in element.attrib:
        element.set("xyz", " ".join(str(float(x) * scale) for x in element.get("xyz").split()))
    if element.tag == "mesh":
        element.set("filename", str((args.source.parent / element.get("filename")).resolve()))
        element.set("scale", " ".join(str(float(x) * scale) for x in element.get("scale", "1 1 1").split()))
    if element.tag == "box":
        element.set("size", " ".join(str(float(x) * scale) for x in element.get("size").split()))
    if element.tag in ("cylinder", "sphere"):
        for attr in ("radius", "length"):
            if attr in element.attrib:
                element.set(attr, str(float(element.get(attr)) * scale))
    if element.tag == "mass":
        element.set("value", str(float(element.get("value")) * scale**3))
    if element.tag == "inertia":
        for attr, value in element.attrib.items():
            element.set(attr, str(float(value) * scale**5))
for joint in tree.findall("joint"):
    if joint.get("type") == "prismatic":
        limit = joint.find("limit")
        for attr in ("lower", "upper", "velocity"):
            limit.set(attr, str(float(limit.get(attr)) * scale))
scaled_urdf = args.output_dir / "articraft_toast_scaled_import.urdf"
tree.write(scaled_urdf)
converter = UrdfConverter(
    UrdfConverterCfg(
        asset_path=str(scaled_urdf),
        usd_dir=str(args.output_dir / "usd_import"),
        usd_file_name="articraft_toast.usda",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=True,
        merge_fixed_joints=False,
        collision_type="Convex Decomposition",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=None)
        ),
    )
)
stage = Usd.Stage.Open(converter.usd_path)
stage.Flatten().Export(str(destination))
print(f"Uniform scale {scale:.12f}; exported {destination}")
app.close()
