# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Conservative geometric checks independent of the motion-planning backend."""

import numpy as np


def collider_transform(cache, body, collider):
    """Map collider coordinates into the rigid pose frame, retaining inherited instance scale."""
    T_B_W = cache.GetLocalToWorldTransform(body).RemoveScaleShear().GetInverse()
    return np.array(cache.GetLocalToWorldTransform(collider) * T_B_W).T


def penetration(a, b, primitive=None):
    """Return minimum OBB separating-axis overlap; negative means separated."""
    _, ca, ra, ha = a
    _, cb, rb, hb = b
    cross = np.cross(ra.T[:, None, :], rb.T[None, :, :]).reshape(-1, 3)
    norms = np.linalg.norm(cross, axis=1)
    keep = norms > 1e-8
    axes = np.concatenate((ra.T, rb.T, cross[keep] / norms[keep, None]))
    radius_a = np.abs(axes @ ra) @ ha
    kind, axis = primitive or ("box", None)
    if kind == "mesh":
        projections = (axis * ha) @ ra.T @ axes.T
        a_min = axes @ ca + projections.min(axis=0)
        a_max = axes @ ca + projections.max(axis=0)
        b_center = axes @ cb
        b_radius = np.abs(axes @ rb) @ hb
        return float(np.minimum(a_max - (b_center - b_radius), b_center + b_radius - a_min).min())
    if kind == "sphere":
        delta = np.abs(rb.T @ (ca - cb)) - hb
        distance = np.linalg.norm(np.maximum(delta, 0)) + min(float(max(delta)), 0)
        return float(max(ha) - distance)
    if kind == "cylinder":
        axial = np.clip(np.abs(axes @ ra[:, axis]), 0, 1)
        radius = float(max(np.delete(ha, axis)))
        radius_a = axial * ha[axis] + radius * np.sqrt(1 - axial * axial)
    radius_b = np.abs(axes @ rb) @ hb
    separation = np.abs(axes @ (ca - cb))
    return float(np.min(radius_a + radius_b - separation))


def validate_held_object(boxes, held, contact_supports=(), primitive_geometry=None):
    """Reject a carried collider that crosses the table or penetrates another asset."""
    moving = [b for b in boxes if b[0] == held]
    static = [b for b in boxes if b[0] != held and b[0] not in contact_supports]
    for index, item in enumerate(moving):
        _, c, r, h = item
        extent = np.abs(r) @ h
        bottom_support = extent[2]
        kind, axis = (primitive_geometry or {}).get(held, [("box", None)] * len(moving))[index]
        if kind == "mesh":
            bottom_support = -float(np.min((axis * h) @ r[2, :]))
        elif kind == "sphere":
            bottom_support = float(max(h))
        elif kind == "cylinder":
            axial = abs(r[2, axis])
            radius = float(max(np.delete(h, axis)))
            bottom_support = axial * h[axis] + radius * np.sqrt(max(0, 1 - axial * axial))
        if c[2] - bottom_support < 0.7385:
            raise RuntimeError("held_object_crosses_table")
        for other in static:
            _, oc, ort, oh = other
            if np.any(np.abs(c - oc) > extent + np.abs(ort) @ oh):
                continue
            if penetration(item, other, (kind, axis)) > 0.0015:
                raise RuntimeError(f"held_object_collision:{held}:{other[0]}")


def validate_table_footprints(boxes):
    """Keep every object collider inside the one-metre table with a 10 mm edge margin."""
    for name, c, r, h in boxes:
        extent = np.abs(r) @ h
        if np.any(np.abs(c[:2]) + extent[:2] > 0.49):
            raise RuntimeError(f"layout_outside_table:{name}")


def sphere_box_distances(points, radii, centers, rotations, half_extents):
    """Return the signed sphere-to-OBB surface distances for every pair."""
    local = np.einsum("sni,nij->snj", points[:, None, :] - centers[None, :, :], rotations)
    delta = np.abs(local) - half_extents[None, :, :]
    return np.linalg.norm(np.maximum(delta, 0), axis=2) + np.minimum(delta.max(axis=2), 0) - radii[:, None]
