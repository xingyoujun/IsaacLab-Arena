# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that exported references survive relocation and planning joints remain unchanged."""

import shutil
import xml.etree.ElementTree as ET

from data_engine.assets.prepare_g2 import collect_usd, planning_urdf


def test_reference_survives_relocation(tmp_path):
    from pxr import Usd, UsdGeom, UsdUtils

    source = tmp_path / "source"
    source.mkdir()
    child = Usd.Stage.CreateNew(str(source / "child.usda"))
    cube = UsdGeom.Cube.Define(child, "/Cube")
    cube.GetSizeAttr().Set(0.12)
    child.SetDefaultPrim(cube.GetPrim())
    child.GetRootLayer().Save()
    parent = Usd.Stage.CreateNew(str(source / "parent.usda"))
    root = parent.DefinePrim("/Object", "Xform")
    root.GetReferences().AddReference(str(source / "child.usda"))
    parent.SetDefaultPrim(root)
    parent.GetRootLayer().Save()
    collect_usd(source / "parent.usda", tmp_path / "bundle/model.usda")
    shutil.copytree(tmp_path / "bundle", tmp_path / "moved")
    shutil.rmtree(source)
    _, _, missing = UsdUtils.ComputeAllDependencies(str(tmp_path / "moved/model.usda"))
    assert not missing
    stage = Usd.Stage.Open(str(tmp_path / "moved/model.usda"))
    assert stage.GetDefaultPrim().GetAttribute("size").Get() == 0.12


def test_planning_geometry_uses_spheres(tmp_path):
    source = tmp_path / "source.urdf"
    source.write_text(
        '<robot name="test"><link name="base"><visual><geometry><mesh'
        ' filename="package://unavailable"/></geometry></visual><inertial><mass value="2"/></inertial></link><joint'
        ' name="q" type="revolute"><parent link="base"/><child link="tip"/><limit lower="-1"'
        ' upper="1"/></joint></robot>'
    )
    target = tmp_path / "out.urdf"
    planning_urdf(source, target)
    old, new = ET.parse(source).getroot(), ET.parse(target).getroot()
    assert ET.tostring(old.find("joint")) == ET.tostring(new.find("joint"))
    assert new.find("link/inertial/mass").attrib["value"] == "2"
    assert new.find(".//mesh") is None
