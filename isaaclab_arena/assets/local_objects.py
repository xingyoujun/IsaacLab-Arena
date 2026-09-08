# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets loaded from local (non-Nucleus) USD files on the developer's machine.

These are experimental assets that are not part of the shared object library. Their ``usd_path``
points at an absolute host path, so they are only usable on machines where that path exists.
"""

import os

from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.turnable import Turnable
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject, SmallGear
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

LOCAL_ASSET_DIR = os.environ.get("ARENA_LOCAL_ASSET_DIR", "/home/ubuntu/playground/objects/arena_local")
"""Host directory holding the local USD assets. Override with ``ARENA_LOCAL_ASSET_DIR``."""


@register_asset
class RoboDojoSimpleRoom(LibraryBackground):
    """The plain room RoboDojo stages every simulated task in.

    ``env_cfg/scene/default.yml`` selects ``Room: Simple_Room_nolight`` at scale 0.5 and pairs it
    with an HDRI dome (``brown_photostudio_02_4k.hdr``, intensity 1000) -- the "nolight" variant
    carries no emitters of its own, so the dome does all the lighting. Arena environments that
    want RoboDojo's look should spawn this together with the ``brown_photostudio_robolab`` HDR
    and drop the ground plane, since the room brings its own floor.
    """

    name = "robodojo_simple_room"
    tags = ["background", "robodojo"]
    # Verbatim copy of RoboDojo's Assets/Room/Simple_Room_nolight, textures included.
    usd_path = f"{LOCAL_ASSET_DIR}/simple_room_nolight/simple_room_nolight.usd"
    object_min_z = -0.05

    SCALE = (0.5, 0.5, 0.5)
    """The scale ``default.yml`` applies. The asset is authored several times life size."""


@register_asset
class RoboDojoTable(LibraryBackground):
    """The tabletop RoboDojo works on: a mahogany slab, 1.1 m deep by 1.4 m wide by 50 mm thick.

    RoboDojo builds this at run time rather than shipping it as an asset
    (``env/scene_manager/objects/table.py``): a Cube mesh scaled to ``[1.4, 1.1, 0.05]`` with the
    ``material_0122`` MDL (Mahogany_Planks) bound to it. The USD here is that slab authored once,
    with its long axis along y so an Agibot facing +x needs no rotation, and at its real size so
    the prim origin is exactly the slab centre.
    """

    name = "robodojo_table"
    tags = ["background", "robodojo"]
    usd_path = f"{LOCAL_ASSET_DIR}/robodojo_table/robodojo_table.usda"
    object_min_z = -0.05

    HALF_THICKNESS_M = 0.025
    """The work surface is ``origin_z + HALF_THICKNESS_M``, exactly -- there is no lip or inset."""

    DEPTH_M = 1.1
    WIDTH_M = 1.4

    # RoboDojo's table loader builds its PhysicsMaterial with static and dynamic friction both
    # 0.8, against Isaac Lab's default of 0.5.
    spawn_cfg_addon = {
        "physics_material": RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8, restitution=0.0),
    }


@register_asset
class Bowl(LibraryObject):
    """A stackable bowl from RoboDojo's stack_bowls task."""

    name = "bowl"
    tags = ["object"]
    # RoboDojo's Assets/Object/RoboDojo/Rigid/bowl/00001, used exactly as it ships: Z-up, metres,
    # defaultPrim /root with RigidBodyAPI, collision approximated by ``convexDecomposition``.
    #
    # Use the asset unmodified. RoboDojo does the same -- its loader
    # (env/scene_manager/objects/rigid.py) sets only mass and friction and passes
    # ``collision=False`` to SingleGeometryPrim, so the collider is whatever the usdz authored.
    usd_path = f"{LOCAL_ASSET_DIR}/bowl.usdz"
    object_type = ObjectType.RIGID

    HALF_HEIGHT_M = 0.0301
    """Origin is at the geometry centre, so a bowl rests on a surface at ``surface_z + this``.

    The bowl is 110 mm across and 60 mm tall, sitting 30.1 mm below its origin."""

    # Friction values are RoboDojo's, taken from its loader rather than from the asset's
    # metadata.json: env/scene_manager/objects/rigid.py builds the PhysicsMaterial from
    # ``static_friction`` (default 0.6) and ``dynamic_friction`` (default 1.5) in the instance
    # config, and stack_bowls.yml sets neither, so the bowls run at 0.6/1.5. The metadata's
    # ``friction: 0.55`` key is never read. Dynamic above static is unusual but it is what the
    # benchmark ships, and a stack of identical bowls needs the extra grip to stay up.
    #
    # Mass is deliberately NOT set: the bowl runs at whatever its USD density gives (~0.075 kg).
    #
    # Declaring it here would be dead code that reads as if it worked. Isaac Lab's
    # ``MassPropertiesCfg`` only *modifies* an existing attribute, and this source has no
    # ``UsdPhysics.MassAPI``, so a ``mass_props`` entry is silently dropped (logged as "Could not
    # perform 'modify_mass_properties'"). Forcing metadata's 0.32 kg would need a ``.usda``
    # override layer applying ``PhysicsMassAPI`` -- and it buys nothing: both masses were measured
    # to behave identically, and RoboDojo's own articulation loader never sets mass either.
    spawn_cfg_addon = {
        "physics_material": RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=1.5, restitution=0.0),
    }


@register_asset
class Bread(LibraryObject):
    """A slice of bread from RoboDojo's make_toast task."""

    name = "bread"
    tags = ["object", "robodojo"]
    # RoboDojo's Assets/Object/RoboDojo/Rigid/bread/00000, wrapped in a ``.usda`` layer that
    # applies ``PhysicsMassAPI`` so the mass below can take effect -- the shipped usdz has none.
    usd_path = f"{LOCAL_ASSET_DIR}/bread_massed.usda"
    object_type = ObjectType.RIGID

    THICKNESS_M = 0.0118
    """The slice's thin dimension, along its local Z. Its face is 116.9 x 116.1 mm."""

    FACE_WIDTH_M = 0.1169
    FACE_HEIGHT_M = 0.1161

    # RoboDojo's rigid loader (env/scene_manager/objects/rigid.py) builds the PhysicsMaterial from
    # the instance config's ``static_friction`` (default 0.6) and ``dynamic_friction`` (default
    # 1.5); make_toast.yml sets neither, so the bread runs at the defaults. The metadata's
    # ``friction: 0.45`` key is never read -- the same pattern as the bowl above.
    #
    # Mass, unlike the bowl's, *is* set: at the USD's default density a 117 x 116 x 12 mm slab
    # weighs ~0.16 kg, over twice RoboDojo's 0.07 kg, and a slice this thin is pinched rather than
    # cradled, so the extra weight works directly against the grasp.
    spawn_cfg_addon = {
        "physics_material": RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=1.5, restitution=0.0),
    }


@register_asset
class BreadShelf(LibraryObject):
    """The four-slot toast rack the bread starts in, from RoboDojo's make_toast task."""

    name = "bread_shelf"
    tags = ["object", "robodojo"]
    # RoboDojo's Assets/Object/RoboDojo/Geometry/bread_shelf/00000, used as it ships.
    usd_path = f"{LOCAL_ASSET_DIR}/bread_shelf.usdz"
    object_type = ObjectType.RIGID

    HALF_EXTENTS_M = (0.07915, 0.041675, 0.0381)
    """Half the rack's 158.3 x 83.5 x 76.2 mm bounding box; its origin is at the box centre.

    RoboDojo's ``is_A_in_B`` tests containment against exactly this box, projected to world XY."""

    HALF_HEIGHT_M = HALF_EXTENTS_M[2]

    SLOT_X_M = (0.035, 0.010, -0.015, -0.039)
    """Where the four slots sit along the rack's local X, from its ``passive.support`` metadata."""

    SLOT_Z_M = 0.035
    """Height of every slot's support point above the rack origin, from the same metadata."""

    # The rack is a fixture, not something to be picked up. RoboDojo loads it through its
    # GeometryObject path (env/scene_manager/objects/geometry.py), which explicitly sets
    # ``rigidBodyEnabled = False`` and ``physics:kinematicEnabled`` on the referenced prim even
    # though the usdz authors a RigidBodyAPI. ``kinematic_enabled`` is Isaac Lab's equivalent.
    #
    # Friction is left at RoboDojo's GeometryObject defaults, which are 0.0/0.0 -- deliberately
    # frictionless, so a slice slides into a slot rather than catching on its lip.
    spawn_cfg_addon = {
        "rigid_props": RigidBodyPropertiesCfg(kinematic_enabled=True),
        "physics_material": RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0, restitution=0.5),
    }


@register_asset
class Toaster(LibraryObject):
    """A two-slot toaster with a sliding lever, from RoboDojo's make_toast task."""

    name = "toaster"
    tags = ["object", "robodojo"]
    # RoboDojo's Assets/Object/RoboDojo/Articulation/toaster/00000, used as it ships.
    usd_path = f"{LOCAL_ASSET_DIR}/toaster.usdz"
    object_type = ObjectType.ARTICULATION

    HALF_HEIGHT_M = 0.0817
    """Origin is at the geometry centre of a 226.7 x 155.4 x 161.9 mm body."""

    LEVER_JOINT_NAME = "joint_1"
    """The lever. Its metadata tags this joint as the ``toast_botton`` affordance; the USD makes it
    prismatic with range [-0.0138, 0.0642] m, and measurement confirms that running it towards the
    upper limit carries the lever body straight down the toaster's own -Z. It rests at 0, so it
    starts 17.7% of the way along its travel, and it is a pure damper (drive stiffness 0, damping
    100) -- once pushed down it stays there rather than springing back.

    Deliberately not exposed through the ``Pressable`` affordance: ``is_pressed`` flips the
    fraction to ``1 - fraction`` for any joint whose lower limit is negative, which reverses the
    meaning of ``LEVER_PRESSED_FRACTION`` below."""

    LEVER_PRESSED_FRACTION = 0.85
    """RoboDojo's ``is_joint_position_above_ratio(toaster, percentage=0.85, tag="toast_botton")``,
    measured from the lower limit -- so the lever counts as down past 52.5 mm of its 78 mm travel."""

    SLOT_RECT_LOCAL_M = {
        "toast_slot1": ((-0.078, 0.072), (-0.041, -0.011)),
        "toast_slot2": ((-0.078, 0.072), (0.014, 0.039)),
    }
    """``(x_min, x_max), (y_min, y_max)`` of each bread slot in the toaster's own frame.

    RoboDojo stores each slot as four corner points under ``passive.functional`` and tests
    containment against the polygon they span in world XY. The corners are axis-aligned in the
    toaster frame and the toaster is only ever yawed, so an axis-aligned test in the local frame
    is the same predicate, without reconstructing the polygon every step."""

    SLOT_Z_LOCAL_M = -0.051
    """Height of both slots' corner points in the toaster frame -- the bottom of the slot."""


@register_asset
class PegPlatform(LibraryObject):
    """The platform with its fixed peg, from USDCraft's ``peg_and_sleeve_assembly`` (agibot_assets_v0)."""

    name = "peg_platform"
    tags = ["object", "usdcraft"]
    # The generated model.usdc holds the platform and the sleeve as sibling rigid bodies under one
    # /Asset prim, which Isaac Lab's RigidObject cannot take. peg_platform.usda references the whole
    # asset (so the /Asset/Looks and /Asset/PhysicsMaterials bindings stay intact) and deactivates
    # the sleeve. Colliders, mass (1.80 kg via MassAPI) and the rubber-foot physics material
    # (friction 0.9/0.7) are used exactly as USDCraft authored them.
    usd_path = f"{LOCAL_ASSET_DIR}/peg_sleeve/peg_platform.usda"
    object_type = ObjectType.RIGID

    HALF_EXTENTS_XY_M = (0.12, 0.09)
    """Half the 240 x 180 mm deck footprint, long side along local x."""

    DECK_TOP_Z_M = 0.020
    """The deck's top face above the origin. The origin is at the underside of the feet, so the
    platform stands on a surface at exactly ``surface_z``."""

    PEG_RADIUS_M = 0.012
    PEG_TOP_Z_M = 0.068
    """The peg is 24 mm across and rises from the deck top to 68 mm above the origin."""

    SLEEVE_SEATED_OFFSET_M = (0.0, 0.0, DECK_TOP_Z_M)
    """Where the sleeve body sits relative to this body in the authored assembly: centred on the
    peg, bottom face on the deck."""

    # The platform is the fixture the sleeve is assembled onto. Held kinematic so a jammed insert
    # cannot shove it across the table; the environment can switch it back to dynamic.
    spawn_cfg_addon = {
        "rigid_props": RigidBodyPropertiesCfg(kinematic_enabled=True),
    }


@register_asset
class PegSleeve(LibraryObject):
    """The removable sleeve from USDCraft's ``peg_and_sleeve_assembly`` (agibot_assets_v0)."""

    name = "peg_sleeve"
    tags = ["object", "usdcraft"]
    # sleeve.usda is the twin of peg_platform.usda: same reference, with the platform deactivated
    # instead. Mass (61 g via MassAPI) and the sleeve physics material (friction 0.9/0.7) are
    # USDCraft's. The shell collider is overridden in that layer to an SDF (resolution 256):
    # USDCraft's convexDecomposition of the hollow shell bulged into the bore to r 9.06 mm, so the
    # 12 mm peg could not enter. The two grip bands keep their convex hulls (that is what the pads
    # pinch). Verified 2026-09-06: the SDF layer reaches PhysX -- dropped concentric or 6 mm off
    # axis the sleeve seats on the peg to 0.0 mm with 0.5 mm radial play.
    usd_path = f"{LOCAL_ASSET_DIR}/peg_sleeve/sleeve.usda"
    object_type = ObjectType.RIGID

    HEIGHT_M = 0.065
    """The sleeve's origin is at its bottom face, so it stands on a surface at ``surface_z``."""

    OUTER_RADIUS_M = 0.0225
    """Radius over the two rubber grip bands; the orange shell between them is 20 mm in radius."""


AGIBOT_ASSETS_V0_DIR = os.environ.get("ARENA_AGIBOT_ASSETS_V0_DIR", "/home/ubuntu/playground/objects/agibot_assets_v0")
"""Host directory of the USDCraft ``agibot_assets_v0`` export (one ``rec_*/isaac/model.usdc`` per asset)."""


def _agibot_asset_usd(record: str) -> str:
    """Path of a USDCraft record's PhysX entry inside ``AGIBOT_ASSETS_V0_DIR``."""
    return f"{AGIBOT_ASSETS_V0_DIR}/{record}/isaac/model.usdc"


# The workbench containers are fixtures: kinematic, so a dropped part cannot shove them.
_FIXTURE_SPAWN_ADDON = {"rigid_props": RigidBodyPropertiesCfg(kinematic_enabled=True)}


@register_asset
class GrayTray(LibraryObject):
    """Low-profile grey sorting tray from USDCraft's agibot_assets_v0 (320 x 220 x 28 mm)."""

    name = "gray_tray"
    tags = ["object", "usdcraft", "container"]
    usd_path = _agibot_asset_usd(
        "rec_create-one-standalone-low-profile-rectangular-so_20260902_093920_eebb12bf_0470470e"
    )
    object_type = ObjectType.RIGID

    HALF_EXTENTS_XY_M = (0.160, 0.110)
    """Half the outer footprint; the origin is at the centre of the underside."""

    RIM_HEIGHT_M = 0.028
    """Top of the walls above the origin."""

    WALL_MARGIN_M = 0.025
    """Shrink of the footprint for the "inside" test, covering the walls and the rim."""

    spawn_cfg_addon = _FIXTURE_SPAWN_ADDON


@register_asset
class GreenTray(LibraryObject):
    """Shallow green parts tray from USDCraft's agibot_assets_v0 (328 x 228 x 45 mm).

    Its colliders are authored as primitive floor and wall pieces, so the interior is exact."""

    name = "green_tray"
    tags = ["object", "usdcraft", "container"]
    usd_path = _agibot_asset_usd(
        "rec_create-one-standalone-shallow-rectangular-parts_20260902_094317_6f04a73d_ef0edf84"
    )
    object_type = ObjectType.RIGID

    HALF_EXTENTS_XY_M = (0.164, 0.114)
    RIM_HEIGHT_M = 0.045
    WALL_MARGIN_M = 0.025
    spawn_cfg_addon = _FIXTURE_SPAWN_ADDON


@register_asset
class BlueBin(LibraryObject):
    """Deep blue storage bin from USDCraft's agibot_assets_v0 (408 x 298 x 200 mm, tapered walls)."""

    name = "blue_bin"
    tags = ["object", "usdcraft", "container"]
    usd_path = _agibot_asset_usd(
        "rec_create-one-standalone-deep-rectangular-storage-b_20260902_094833_a40972a0_f2587215"
    )
    object_type = ObjectType.RIGID

    HALF_EXTENTS_XY_M = (0.204, 0.149)
    RIM_HEIGHT_M = 0.200
    WALL_MARGIN_M = 0.040
    """Wider than the trays': the walls taper inwards towards the floor."""

    spawn_cfg_addon = _FIXTURE_SPAWN_ADDON


@register_asset
class Wrench(LibraryObject):
    """Combination wrench from USDCraft's agibot_assets_v0 (169 x 33 x 6 mm, 124 g)."""

    name = "wrench"
    tags = ["object", "usdcraft", "tool"]
    usd_path = _agibot_asset_usd(
        "rec_create-one-standalone-compact-combination-wrench_20260902_094027_0e2a7fd4_9db90af6"
    )
    object_type = ObjectType.RIGID

    HALF_EXTENTS_M = (0.0845, 0.0165, 0.003)
    """Origin at the centre of the underside; it lies flat, 6 mm thick."""


@register_asset
class MetalBillet(LibraryObject):
    """Rectangular metal billet from USDCraft's agibot_assets_v0 (80 x 40 x 20 mm, 173 g)."""

    name = "metal_billet"
    tags = ["object", "usdcraft", "raw_material"]
    usd_path = _agibot_asset_usd(
        "rec_create-one-standalone-small-rectangular-metal-bi_20260902_093519_2ae80357_28c77f8c"
    )
    object_type = ObjectType.RIGID

    HALF_EXTENTS_M = (0.040, 0.020, 0.010)
    """Origin at the centre of the underside."""


@register_asset
class MetalBlock(LibraryObject):
    """Raw-stock cube for the workbench: a 48 mm block (Isaac's dex cube at Arena's 0.8 scale) with the
    USDCraft billet's matte machined-aluminium look, authored locally (2026-09-06)."""

    name = "metal_block"
    tags = ["object"]
    usd_path = f"{LOCAL_ASSET_DIR}/metal_block/metal_block.usda"
    object_type = ObjectType.RIGID
    HALF_EXTENTS_M = (0.024, 0.024, 0.024)
    """Origin at the centre of the underside; 48 mm cube, exact Cube collider, 0.30 kg."""


@register_asset
class BearingAssembly(LibraryObject):
    """Finished bearing assembly from USDCraft's agibot_assets_v0 (72 mm across, 35 mm tall, 520 g).

    Two links -- the housing (root) and a freely spinning inner race -- so it loads as a
    floating-base articulation rather than a rigid body."""

    name = "bearing_assembly"
    # Not tagged "object": its two links are sibling rigid bodies at one depth, which Arena's
    # detect_object_type rejects ("multiple roots"), so the registry-wide detection test would
    # fail on it. It stays selectable by name (tidy_workbench's ``finished_part``), but the Agibot
    # cannot hold it in any tested form -- see docs/agibot/asset_guide.md.
    tags = ["usdcraft", "finished_part"]
    usd_path = _agibot_asset_usd(
        "rec_create-one-standalone-finished-metal-bearing-ass_20260902_094319_8e29fcfd_0ae7bb99"
    )
    object_type = ObjectType.ARTICULATION

    RADIUS_M = 0.036
    HEIGHT_M = 0.035
    """Origin at the centre of the flange's underside."""


@register_asset
class SmallGearCentred(LibraryObject):
    """Arena's Factory small gear with its geometry re-centred on the body origin.

    The library ``small_gear`` is authored in the gear base's shaft frame -- 50.75 mm off its own
    origin -- which breaks origin-based placement and membership tests. The local layer shifts the
    meshes so the axis passes through the origin and the underside is at z = 0, and drops the
    articulation-root API so the single-link gear loads as a plain rigid body (a zero-joint
    articulation cannot be built on the CPU device). SDF collider, 50 g and Arena's spawn add-ons
    are the library's. Unscaled it is 22 mm across and 25 mm tall; use it at the library's scale
    of 2.0 (44 x 50 mm).
    """

    name = "small_gear_centred"
    tags = ["object", "factory", "finished_part"]
    usd_path = f"{LOCAL_ASSET_DIR}/factory/small_gear_centred.usda"
    object_type = ObjectType.RIGID
    spawn_cfg_addon = SmallGear.spawn_cfg_addon

    RADIUS_M = 0.0109
    HEIGHT_M = 0.025
    """Unscaled; the origin is at the centre of the underside."""


# --- USDCraft export of 2026-09-07: the drawer cabinet and the detent knob ---------------------

AGIBOT_ASSETS_0907_DIR = os.environ.get("ARENA_AGIBOT_ASSETS_0907_DIR", "/home/ubuntu/playground/objects/assets_0907")
"""Host directory of the USDCraft export of 2026-09-07 (one ``rec_*/isaac/model.usdc`` per asset)."""


def _agibot_asset_0907_usd(record: str) -> str:
    """Path of a USDCraft record's PhysX entry inside ``AGIBOT_ASSETS_0907_DIR``."""
    return f"{AGIBOT_ASSETS_0907_DIR}/{record}/isaac/model.usdc"


# Articulated fixtures are pinned by a fixed joint from their root link to the world (Isaac Lab's
# spawner-level ``fix_root_link``): a kinematic flag would freeze every link, the drawer and the
# knob included.
_FIXED_BASE_SPAWN_ADDON = {"fix_root_link": True}


@register_asset
class DrawerCabinet(LibraryObject, Openable):
    """Compact walnut tabletop cabinet with one prismatic drawer, from USDCraft (2026-09-07).

    Two links: the carcass (root, 300 wide x 230 deep x 185 mm tall, 2.0 kg, pinned to the world)
    and the drawer (274 x 200 x 153 mm box with a bar handle, 1.14 kg). The asset's drawer front is
    on its -y side; the origin is at the centre of the carcass's underside. The ``Openable``
    affordance reads the drawer's travel with the polarity measured in simulation (see
    ``DRAWER_CLOSED_JOINT_POS_M`` / ``DRAWER_OPEN_JOINT_POS_M``), not Isaac Lab's normalised
    joint position, whose sign flips for a joint with a negative lower limit.
    """

    name = "drawer_cabinet"
    # Not tagged "object": its links are sibling rigid bodies at one depth, which Arena's
    # detect_object_type rejects ("multiple roots"), like the bearing assembly.
    tags = ["usdcraft", "openable", "fixture"]
    usd_path = _agibot_asset_0907_usd(
        "rec_create-a-simulator-ready-compact-wooden-tabletop_20260907_070639_95b890ae_7db5d872"
    )
    object_type = ObjectType.ARTICULATION
    spawn_cfg_addon = _FIXED_BASE_SPAWN_ADDON

    openable_joint_name = "cabinet_to_drawer"
    openable_threshold = 0.5

    HALF_WIDTH_M = 0.150
    HALF_DEPTH_M = 0.115
    HEIGHT_M = 0.185
    """Carcass extents; the origin is at the centre of the underside, the drawer slides along -y."""

    DRAWER_CLOSED_JOINT_POS_M = -0.096
    DRAWER_OPEN_JOINT_POS_M = 0.064
    """Joint positions of ``cabinet_to_drawer`` (limits [-0.096, 0.064]) at which the drawer front is
    flush with the carcass (closed) and pulled out by its full 160 mm of travel (open). The asset
    rests at joint 0 with the drawer 96 mm out. Positive joint travel moves the drawer toward -y,
    i.e. out of the cabinet -- measured on the spawned articulation, 2026-09-07."""

    DRAWER_FRONT_Y_AT_JOINT_ZERO_M = -0.218
    """The drawer's front face (asset frame) at joint position 0; it moves by minus the joint position."""

    HANDLE_CENTRE_AT_JOINT_ZERO_M = (0.0, -0.244, 0.099)
    """Centre of the 112 mm bar handle at joint 0: an 11 x 11 mm bar standing 20.5 mm proud of the
    drawer front on two posts 80 mm apart, so a pad fits behind it for a top-down pinch."""

    DRAWER_INTERIOR_LOCAL_M = ((-0.125, 0.125), (-0.206, -0.027), (0.0255, 0.14))
    """``(x, y, z)`` ranges of the drawer's inner cavity in the asset frame at joint 0 (250 x 179 mm
    floor at z 25.5 mm, walls 114 mm tall); shift y by minus the joint position."""

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            openable_joint_name=self.openable_joint_name,
            openable_threshold=self.openable_threshold,
        )

    def get_openness(self, env, asset_cfg=None):
        """Openness 0 (drawer flush) .. 1 (drawer fully out), from the measured joint polarity."""
        from isaaclab.managers import SceneEntityCfg

        from isaaclab_arena.utils.joint_utils import get_unnormalized_joint_position, normalize_value

        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        joint_pos = get_unnormalized_joint_position(env, asset_cfg)
        return normalize_value(joint_pos, self.DRAWER_CLOSED_JOINT_POS_M, self.DRAWER_OPEN_JOINT_POS_M)

    def rotate_revolute_joint(self, env, env_ids, asset_cfg=None, percentage: float = 0.0):
        """Slide the drawer to the given openness (0 closed .. 1 fully out); the name is Openable's."""
        from isaaclab.managers import SceneEntityCfg

        from isaaclab_arena.utils.joint_utils import set_unnormalized_joint_position, unnormalize_value

        assert 0.0 <= percentage <= 1.0, "Percentage must be between 0.0 and 1.0"
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        joint_pos = unnormalize_value(percentage, self.DRAWER_CLOSED_JOINT_POS_M, self.DRAWER_OPEN_JOINT_POS_M)
        set_unnormalized_joint_position(env, asset_cfg, joint_pos, env_ids)


@register_asset
class RotaryKnob(LibraryObject, Turnable):
    """Industrial 0-100 rotary control knob on a fixed dial panel, from USDCraft (2026-09-07).

    Three links: the panel (root, a 130 x 130 x 7 mm plate in the asset's x-z plane, pinned to the
    world), the ridged knob (75 mm across, 30 mm proud of the panel's -y face, 0.11 kg, turning
    about the asset's y axis through 0..270 degrees) and a spring-loaded detent plunger that clicks
    into the knob's 51-valley cam. The knob joint itself is nearly free (damping 1.7e-4, no
    Coulomb friction): whatever holds an angle is the detent. The origin is the knob's axis point
    on the panel's mid-plane.
    """

    name = "rotary_knob"
    # Not tagged "object" for the same reason as the drawer cabinet (sibling links).
    tags = ["usdcraft", "turnable", "fixture"]
    usd_path = _agibot_asset_0907_usd(
        "rec_create-a-simulator-ready-industrial-rotary-contr_20260907_070836_bb674e43_2843401c"
    )
    object_type = ObjectType.ARTICULATION
    spawn_cfg_addon = _FIXED_BASE_SPAWN_ADDON

    turnable_joint_name = "panel_to_knob"
    min_level_angle_deg = 27.0
    max_level_angle_deg = 270.0
    num_levels = 10
    """Ten 27-degree levels over the 270-degree dial (a 0-100 scale: level k is the reading
    10 k .. 10 (k + 1)); the first 27 degrees are the dead zone (level -1), where the knob is put at
    reset. Level k spans [27 + 27 k, 27 + 27 (k + 1)) degrees -- a +/- 13.5 degree window."""

    PANEL_HALF_SIZE_M = 0.065
    PANEL_HALF_THICKNESS_M = 0.0035
    KNOB_RADIUS_M = 0.0375
    KNOB_FACE_Y_M = -0.033
    """The knob's front face lies 33 mm along the asset's -y axis from the origin."""

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        # Scale must come in here: the spawn config is generated during construction, so setting
        # ``.scale`` afterwards changes nothing in the simulation.
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            scale=scale,
            turnable_joint_name=self.turnable_joint_name,
            min_level_angle_deg=self.min_level_angle_deg,
            max_level_angle_deg=self.max_level_angle_deg,
            num_levels=self.num_levels,
        )
