# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Put the sleeve onto the peg in ``agibot_sleeve_on_peg`` with cuMotion-planned motions, and record
HDF5 demos.

The sleeve is a body of revolution standing on the table. The grasp copies the one a human
teleoperator used successfully (recorded demo, 2026-09-03): the tool approaches almost
horizontally -- 60-80 deg from straight down -- with the jaws closing horizontally across the
upper grip band. Held that way the sleeve hangs from the pads and is free to swing about the jaw
axis, so when it is lowered onto the peg the peg itself squares it up before the fingers open.
The heading around the sleeve and the lean are handed to the planner as free parameters. The release is the authored seated pose on the platform's peg,
again with the spin about the vertical free; on arrival the tool-to-sleeve offset is re-measured
and the final descent re-aimed, since a 0.5 mm radial clearance leaves no room for in-hand slip.

With ``--record-dir`` the run is captured as a ``record_demos.py``-compatible HDF5 dataset, the
same machinery as stack_bowls_cumotion.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/sleeve_on_peg_cumotion.py --video /tmp/sleeve.mp4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="agibot_sleeve_on_peg")
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--record-every", type=int, default=3, help="Keep one video frame in N physics steps.")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--grasp-height",
    type=float,
    default=0.047,
    help=(
        "Where the pads pinch, above the sleeve's bottom face. 0.047 is the upper grip band, where the human demo"
        " pinched."
    ),
)
parser.add_argument(
    "--tilt",
    type=float,
    nargs="+",
    default=(75.0, 65.0, 80.0, 55.0),
    help="Tool leans from straight-down to offer; the human demo held 75 deg (jaws nearly horizontal). Gentlest first.",
)
parser.add_argument(
    "--jaw-spin",
    type=float,
    nargs="+",
    default=(0.0, 180.0),
    help=(
        "Rolls about the approach axis to offer. With the lean applied about the tool's y axis, 0 and 180 keep the jaws"
        " horizontal; they are the same pinch with the wrist flipped."
    ),
)
parser.add_argument(
    "--table-raise", type=float, default=0.0, help="Raise the work surface by this much (env table_raise_m)."
)
parser.add_argument(
    "--tool-pad-offset",
    type=float,
    default=0.0167,
    help="How far the tool frame sits past the pads along the approach; the grasp point is raised by it.",
)
parser.add_argument(
    "--release-standoff",
    type=float,
    default=0.060,
    help=(
        "Height above the seat to carry to and re-measure at before the final descent; 0.060 puts the sleeve's bottom"
        " 12 mm over the peg tip."
    ),
)
parser.add_argument(
    "--final-descent-speed", type=float, default=0.08, help="Playback speed of the final descent onto the peg."
)
parser.add_argument(
    "--max-hang-tilt",
    type=float,
    default=20.0,
    help=(
        "Largest lean of the lifted sleeve, in degrees, that is still carried to the peg; beyond it the grasp is"
        " retried."
    ),
)
parser.add_argument("--sleeve-side", type=str, default="both", choices=("left", "right", "both"))
parser.add_argument("--max-grasp-attempts", type=int, default=3)
parser.add_argument(
    "--release-open",
    type=float,
    default=1.0,
    help="Fraction of the fully-open gripper position to open to at the release.",
)
parser.add_argument(
    "--sleeve-y-min",
    type=float,
    default=0.135,
    help="Inner edge of the sleeve's |y| landing band (env sleeve_abs_y_min_m).",
)
parser.add_argument("--record-dir", type=str, default=None, help="Directory for the HDF5 dataset; enables recording.")
parser.add_argument("--dataset-name", type=str, default="sleeve_on_peg", help="HDF5 file name, without extension.")
parser.add_argument("--num-demos", type=int, default=1, help="How many demonstrations to run (and record).")
parser.add_argument("--seed", type=int, default=None, help="Seed for the per-reset sleeve placement.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.managers.recorder_manager import DatasetExportMode  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from isaaclab_arena.assets.local_objects import PegPlatform, PegSleeve  # noqa: E402
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.embodiments.agibot.demo_recorders import agibot_demo_recorder_cfg  # noqa: E402
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena_cumotion.executor import ArmExecutor, EnvActionExecutor, JointActionInterface  # noqa: E402
from isaaclab_arena_cumotion.grasps import (  # noqa: E402
    DOWN_FACING_ROTATION,
    GraspProposal,
    _rot_y,
    _rot_z,
    matrix_from_quat_wxyz,
    quat_wxyz_from_matrix,
)
from isaaclab_arena_cumotion.pick_place import PickAndPlace  # noqa: E402
from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402

NUM_HEADINGS = 24
"""Horizontal approach directions to try around the sleeve."""

TABLE_TOP_Z = 0.6232 + args.table_raise
TABLE_OBSTACLE = "/obstacles/table"
DECK_OBSTACLE = "/obstacles/deck"
PEG_OBSTACLE = "/obstacles/peg"
SLEEVE_KEY = "peg_sleeve"
PLATFORM_KEY = "peg_platform"

# ------------------------------------------------------------------------------------- env ---
recording = args.record_dir is not None
arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
env_cfg = factory._legacy_argparse_cfg_type(
    teleop_device=None,
    enable_cameras=False,
    sleeve_side=args.sleeve_side,
    table_raise_m=args.table_raise,
    sleeve_abs_y_min_m=args.sleeve_y_min,
)
arena_env = factory.build(env_cfg)
PLATFORM_SCALE = env_cfg.platform_scale

if recording:
    from isaaclab_arena.embodiments.agibot.agibot import AgibotDualArmJointActionsCfg  # noqa: E402

    # Joint-space actions, so cuMotion's planned joint paths pass through the action manager
    # verbatim instead of being re-solved (and fought) by RMPFlow. See handover_toast_cumotion.
    arena_env.embodiment.action_config = AgibotDualArmJointActionsCfg()

    _prev_cb = arena_env.env_cfg_callback

    def _recording_env_cfg(patched):
        """Attach the demo recorder; success is judged and exported by this script instead."""
        patched = _prev_cb(patched) if _prev_cb is not None else patched
        patched.recorders = agibot_demo_recorder_cfg(with_cameras=False)  # states-only; images re-rendered offline
        patched.recorders.dataset_export_dir_path = args.record_dir
        patched.recorders.dataset_filename = args.dataset_name
        patched.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        patched.terminations.success = None
        patched.episode_length_s = 600.0
        return patched

    arena_env.env_cfg_callback = _recording_env_cfg

builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
DECIMATION = max(1, round(env.step_dt / env.sim.get_physics_dt()))
if recording:
    args.record_every = max(1, round(args.record_every / DECIMATION))

_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
torch.manual_seed(_seed)
np.random.seed(_seed % 2**32)
print(f"scene placement seed: {_seed}")

camera = None
writer = None
if args.video is not None:
    import imageio.v2 as iio  # noqa: E402

    camera = Camera(
        CameraCfg(
            prim_path="/World/demo_cam",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 30.0)),
        )
    )
    video_path = pathlib.Path(args.video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(video_path, fps=args.fps, codec="libx264", macro_block_size=8)

env.sim.reset()
env.reset()
if camera is not None:
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[1.20, 0.85, 1.15]], device=env.device),
        targets=torch.tensor([[0.40, 0.00, 0.68]], device=env.device),
    )

step_counter = [0]
frame_counter = [0]


def grab_frame() -> None:
    """Keep one camera frame in ``--record-every``; the sensor must be told to update."""
    step_counter[0] += 1
    if writer is None or step_counter[0] % args.record_every != 0:
        return
    camera.update(env.sim.get_physics_dt())
    writer.append_data(camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
    frame_counter[0] += 1


def object_pose(key: str) -> tuple[np.ndarray, np.ndarray]:
    """An object's world position and rotation matrix."""
    asset = env.scene[key]
    position = wp.to_torch(asset.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = wp.to_torch(asset.data.root_quat_w)[0].detach().cpu().float()
    rotation = math_utils.matrix_from_quat(quat_xyzw.unsqueeze(0))[0].numpy().astype(np.float64)
    return position, rotation


def sleeve_position() -> np.ndarray:
    return object_pose(SLEEVE_KEY)[0]


def sleeve_tilt_deg() -> float:
    """How far the sleeve's axis leans from vertical."""
    _, rotation = object_pose(SLEEVE_KEY)
    return float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))


def seat_centre() -> np.ndarray:
    """Where the sleeve's origin sits when seated: the authored offset, in the platform's frame."""
    position, rotation = object_pose(PLATFORM_KEY)
    return position + rotation @ (np.asarray(PegPlatform.SLEEVE_SEATED_OFFSET_M) * PLATFORM_SCALE)


# -------------------------------------------------------------------------------- planning ---
embodiment = arena_env.embodiment
arms = {}
platform_position, platform_rotation = object_pose(PLATFORM_KEY)
deck_half = np.array(PegPlatform.HALF_EXTENTS_XY_M) * PLATFORM_SCALE
deck_top = PegPlatform.DECK_TOP_Z_M * PLATFORM_SCALE
peg_top = PegPlatform.PEG_TOP_Z_M * PLATFORM_SCALE
peg_radius = PegPlatform.PEG_RADIUS_M * PLATFORM_SCALE
for arm in ("left", "right"):
    planner = CumotionArmPlanner(env, embodiment, arm=arm)
    error_m = planner.kinematics_error_m()
    print(f"{arm} arm kinematics cross-check: {error_m * 1000:.2f} mm")
    assert error_m < 1e-3, f"cuMotion's {arm}-arm kinematics disagree with the simulated robot"
    # The work surface, modelled 30 mm below the real top so a grasp near the table stays
    # plannable while gross sweeps through it are blocked (as in stack_bowls_cumotion).
    planner.add_box_obstacle(
        TABLE_OBSTACLE, np.array([0.185, 0.0, TABLE_TOP_Z - 0.09]), (2.2, 1.4, 0.12), safety_tolerance_m=0.0
    )
    # The platform is kinematic, so its deck and peg are static boxes at their true poses.
    planner.add_box_obstacle(
        DECK_OBSTACLE,
        platform_position + platform_rotation @ np.array([0.0, 0.0, 0.5 * deck_top]),
        (2 * deck_half[0], 2 * deck_half[1], deck_top),
        quat_wxyz=quat_wxyz_from_matrix(platform_rotation),
    )
    planner.add_box_obstacle(
        PEG_OBSTACLE,
        platform_position + platform_rotation @ np.array([0.0, 0.0, 0.5 * (deck_top + peg_top)]),
        (2 * peg_radius, 2 * peg_radius, peg_top - deck_top),
        quat_wxyz=quat_wxyz_from_matrix(platform_rotation),
    )
    # The sleeve's origin is its bottom face, so a box centred on it hangs half into the table;
    # the half above is the part that matters for the arm.
    planner.add_scene_object_obstacle(
        SLEEVE_KEY, (2 * PegSleeve.OUTER_RADIUS_M, 2 * PegSleeve.OUTER_RADIUS_M, 2 * PegSleeve.HEIGHT_M)
    )
    arms[arm] = planner

if recording:
    interface = JointActionInterface(env)
    executors = {
        arm: EnvActionExecutor(
            env, arms[arm], interface, f"{arm}_arm_action", f"{arm}_gripper_action", on_step=grab_frame
        )
        for arm in arms
    }
else:
    interface = None
    executors = {arm: ArmExecutor(env, planner, on_step=grab_frame) for arm, planner in arms.items()}
pick_places = {arm: PickAndPlace(arms[arm], executors[arm], contact_obstacles=(TABLE_OBSTACLE,)) for arm in arms}


def send_home(arm: str, home_q: dict[str, np.ndarray]) -> None:
    """Return an arm to the configuration it started in, and reopen its gripper."""
    planner = arms[arm]
    executors[arm].open_gripper()
    plan = planner.plan_config(planner.joint_positions(), home_q[arm])
    if plan is None or not plan.is_executable():
        print(f"  the {arm} arm could not plan its way home")
        return
    executors[arm].follow(plan.path, speed=0.2)


def pitched_grasps() -> list[GraspProposal]:
    """Near-horizontal pinches across the standing sleeve at ``--grasp-height``, the human's grasp.

    The orientation is the top-down frame leaned by ``--tilt`` about the tool's jaw axis, so the
    jaws stay horizontal and close across the cylinder; every heading around the sleeve is
    offered. The tool frame is aimed ``--tool-pad-offset`` past the axis along the approach,
    because the pads sit that far short of the frame (measured 16.5 mm on the recorded grasp).
    """
    position, _ = object_pose(SLEEVE_KEY)
    axis_point = position + np.array([0.0, 0.0, args.grasp_height])
    proposals = []
    for tilt in args.tilt:
        for i in range(NUM_HEADINGS):
            yaw = 2.0 * np.pi * i / NUM_HEADINGS
            for spin in args.jaw_spin:
                orientation = _rot_z(yaw) @ _rot_y(np.radians(tilt)) @ DOWN_FACING_ROTATION @ _rot_z(np.radians(spin))
                approach = orientation[:, 2]
                proposals.append(
                    GraspProposal(
                        f"tilt {tilt:g} heading {np.degrees(yaw):.0f} deg spin {spin:g}",
                        axis_point + approach * args.tool_pad_offset,
                        quat_wxyz_from_matrix(orientation),
                    )
                )
    return proposals


def spin_release_poses(centre: np.ndarray, offset: np.ndarray, rotation: np.ndarray, step_deg: int = 15):
    """Release poses around the vertical for an object held ``offset`` from the tool; the sleeve
    is a body of revolution, so every spin is the same seat."""
    poses = []
    for spin_deg in range(0, 360, step_deg):
        spin = np.radians(spin_deg)
        spin_matrix = np.array([[np.cos(spin), -np.sin(spin), 0.0], [np.sin(spin), np.cos(spin), 0.0], [0.0, 0.0, 1.0]])
        poses.append(
            (f"spin {spin_deg} deg", centre + spin_matrix @ offset, quat_wxyz_from_matrix(spin_matrix @ rotation))
        )
    return poses


def _near_arm(position: np.ndarray) -> str:
    """The arm on the same side of the robot as a position."""
    return "left" if position[1] > arms["left"].base_pos[1] else "right"


def run_demo() -> list[str]:
    """Pick the sleeve and seat it on the peg; returns report lines."""
    home_q = {arm: planner.joint_positions() for arm, planner in arms.items()}
    start = sleeve_position()
    seat = seat_centre()
    print(
        f"sleeve at ({start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}); seat at ({seat[0]:.3f}, {seat[1]:.3f},"
        f" {seat[2]:.3f})"
    )

    candidates = pitched_grasps()
    reach = {arm: [p for p in candidates if arms[arm].ik_reachable(p.position, p.quat_wxyz)] for arm in arms}

    def _placeable(arm: str, proposal: GraspProposal) -> bool:
        offset = proposal.position - start
        rotation = matrix_from_quat_wxyz(proposal.quat_wxyz)
        return any(
            arms[arm].ik_reachable(position, quat)
            for _, position, quat in spin_release_poses(seat + np.array([0.0, 0.0, 0.12]), offset, rotation)
        )

    placeable = {arm: [p for p in cands if _placeable(arm, p)] for arm, cands in reach.items()}
    print(
        "  reachable grasp candidates: "
        + ", ".join(f"{a} {len(n)} ({len(placeable[a])} with a reachable release stand-off)" for a, n in reach.items())
    )
    near_arm = _near_arm(start)
    far_arm = "right" if near_arm == "left" else "left"
    arm = near_arm if placeable[near_arm] or not placeable[far_arm] else far_arm
    proposals = placeable[arm] or reach[arm] or reach[far_arm]
    if not proposals:
        return ["sleeve unreachable by either arm"]
    if not placeable[arm]:
        arm = far_arm if (not reach[arm] and reach[far_arm]) else arm
        print("  no candidate has a reachable release; going ahead on grasp reachability alone")
    print(f"  using the {arm} arm ({'its own side' if arm == near_arm else 'far side'}), {len(proposals)} candidates")
    planner, pick_place = arms[arm], pick_places[arm]

    for other in arms.values():
        other.set_obstacle_enabled(SLEEVE_KEY, False)

    report: list[str] = []
    pick = place = None
    tried: set[str] = set()
    for attempt in range(args.max_grasp_attempts):
        untried = [p for p in proposals if p.label not in tried]
        if not untried:
            break
        if attempt:
            print(f"  retrying with a different grasp ({len(untried)} candidates left)")
        pick = pick_place.pick(
            untried,
            mute_during_descent=(),
            verify_grasp=lambda: sleeve_position()[2] - start[2] > 0.02,
            lift_speed=0.15,
        )
        for line in pick.trace:
            print(f"  {line}")
        if not pick.success:
            break
        tried.add(pick.label)

        grasp_offset = planner.tool_position() - sleeve_position()
        grasp_rotation = matrix_from_quat_wxyz(pick.grasp_quat_wxyz)
        hang_tilt = sleeve_tilt_deg()
        print(
            f"  sleeve rose {(sleeve_position()[2] - start[2]) * 1000:.1f} mm, tilts {hang_tilt:.1f} deg,"
            f" held {np.linalg.norm(grasp_offset[:2]) * 1000:.1f} mm off its axis"
        )
        if hang_tilt > args.max_hang_tilt:
            # Held across its diameter the sleeve is free to swing about the jaw axis: sometimes it
            # hangs upright (its centre of mass is below the upper band), sometimes friction pins
            # it flat against the pads at the tool's own lean. A sleeve leaning that far cannot
            # find the peg, so set it back down where it came from and try another grasp.
            print(f"  hanging {hang_tilt:.0f} deg from vertical (limit {args.max_hang_tilt:g}); putting it back")
            executors[arm].follow(pick.descend_path, speed=0.2)
            executors[arm].open_gripper(hold_arm_at=planner.joint_positions())
            executors[arm].follow(pick.descend_path, reverse=True)
            start = sleeve_position()
            continue
        release_targets = spin_release_poses(seat, grasp_offset, grasp_rotation)

        def _seat_report() -> str:
            here = sleeve_position()
            return (
                f"sleeve {np.linalg.norm(here[:2] - seat[:2]) * 1000:.1f} mm off the peg axis,"
                f" {(here[2] - seat[2]) * 1000:+.1f} mm above the seat, tilt {sleeve_tilt_deg():.1f} deg"
            )

        # Stand off just above the peg tip, not high above it: the sleeve swings about the jaw
        # axis while it descends, and every degree of swing moves its bottom face ~0.8 mm
        # sideways against a 0.5 mm radial clearance. Re-measuring 12 mm over the tip and
        # creeping the last stretch is what a human operator did (demo 5 dropped from 120 mm,
        # arrived 10 mm off the axis at 20 deg of swing, and perched on the tip).
        place = pick_place.place(
            release_targets,
            approach_height_m=args.release_standoff,
            descend_speed=args.final_descent_speed,
            mute_during_descent=(PEG_OBSTACLE, DECK_OBSTACLE),
            release_gripper_pos=args.release_open * planner.cfg.gripper_open_pos,
            retarget=lambda: seat + (planner.tool_position() - sleeve_position()),
            on_release=_seat_report,
        )
        for line in place.trace:
            print(f"  {line}")
        if place.success:
            break
        # Put the sleeve back where it came from and try another grasp.
        executors[arm].follow(pick.descend_path, speed=0.2)
        executors[arm].open_gripper(hold_arm_at=planner.joint_positions())
        executors[arm].follow(pick.descend_path, reverse=True)

    for other in arms.values():
        other.set_obstacle_enabled(SLEEVE_KEY, True)
        other.update_obstacle_pose(SLEEVE_KEY)
    send_home(arm, home_q)
    if pick is None or not pick.success:
        return [f"pick failed ({pick.failure if pick else 'no candidate left'})"]
    if not place.success:
        return [f"place failed ({place.failure}), {len(tried)} grasps tried"]
    executors[arm].step(steps=max(1, 120 // (DECIMATION if recording else 1)))
    here = sleeve_position()
    report.append(
        f"placed: {np.linalg.norm(here[:2] - seat[:2]) * 1000:.1f} mm off the peg axis,"
        f" {(here[2] - seat[2]) * 1000:+.1f} mm above the seat, tilt {sleeve_tilt_deg():.1f} deg"
    )
    return report


# ------------------------------------------------------------------------------- main loop ---
task = arena_env.task
overall: list[str] = []
recorded = 0
for demo in range(args.num_demos):
    print(f"\n--- demo {demo + 1}/{args.num_demos} ---")
    if demo:
        if recording:
            env.recorder_manager.reset()
        env.reset()
        if interface is not None:
            interface.sync_from_robot()
        for executor in executors.values():
            executor._gripper_target = executor.planner.cfg.gripper_open_pos
    executors["left"].step(steps=max(1, 30 // (DECIMATION if recording else 1)))
    for planner in arms.values():
        planner.set_obstacle_enabled(SLEEVE_KEY, True)
        planner.update_obstacle_pose(SLEEVE_KEY)

    try:
        report = run_demo()
    except RuntimeError as error:
        report = [f"aborted: {error}"]
    for line in report:
        print(f"  {line}")

    success = bool(task.is_success(env)[0].item())
    print(f"  task success predicate: {success}")
    overall.append(f"demo {demo + 1}: {'success' if success else 'FAILED'}")

    if success and recording:
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
        env.recorder_manager.export_episodes([0])
        recorded += 1
        print(f"  exported demo {recorded} to {args.record_dir}/{args.dataset_name}.hdf5")

print("\n=== outcome ===")
for line in overall:
    print(f"  {line}")
if recording:
    print(f"  {recorded}/{args.num_demos} demonstrations exported to {args.record_dir}/{args.dataset_name}.hdf5")
if writer is not None:
    writer.close()
    print(f"  wrote {frame_counter[0]} frames to {video_path}")
simulation_app.close()
