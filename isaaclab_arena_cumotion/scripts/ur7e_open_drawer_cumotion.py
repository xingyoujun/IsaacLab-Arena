# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pull the drawer open on the UR7e workcell with cuMotion-planned motions, optionally recording demos.

One demonstration is: open the gripper, plan to a standoff above the pull knob, descend onto it
with the fingers pointing straight down, close the gripper, pull horizontally along the drawer's
opening axis, release, lift and return home. The jaws close across the drawer's width (perpendicular
to the opening axis) so both fingers stand in front of the drawer face beside the knob; the only
rotation that varies with the drawer's placement is the yaw about the vertical.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/ur7e_open_drawer_cumotion.py \\
        --video /home/ubuntu/playground/rr_ur/sim_renders/v5_open_drawer_cumotion.mp4

    # record demonstrations (states only; re-render cameras afterwards)
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/ur7e_open_drawer_cumotion.py \\
        --record-dir /home/ubuntu/playground/datasets/ur7e_open_drawer --num-demos 10 --seed 0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="ur7e_usdcraft_open_drawer")
parser.add_argument("--embodiment", type=str, default="ur7e_robotiq_joint_pos")
parser.add_argument("--video", type=str, default=None, help="Write the D435 view here and a scene view next to it.")
parser.add_argument("--record-every", type=int, default=3, help="Keep one video frame in N physics steps.")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--pull-distance", type=float, default=0.10, help="How far to pull the drawer out, metres.")
parser.add_argument("--standoff", type=float, default=0.08, help="Pre-grasp height above the grasp pose, metres.")
parser.add_argument(
    "--grasp-depth",
    type=float,
    default=0.01,
    help="How far the tool centre point goes below the knob centre, so the pads (not the fingertips) hold the ball.",
)
parser.add_argument(
    "--knob-forward-shift",
    type=float,
    default=0.005,
    help="Grasp this far in front of the knob centre (along the opening axis) to keep the fingers off the drawer face.",
)
parser.add_argument("--pull-speed", type=float, default=0.15, help="Playback speed fraction of the pull.")
parser.add_argument(
    "--spins",
    type=float,
    nargs="+",
    default=(0.0, 180.0),
    help="Yaw offsets (deg) about the vertical to offer; 0 and 180 are the same pinch with the wrist flipped.",
)
parser.add_argument("--pull-segments", type=int, default=8, help="IK waypoints along the straight pull.")
parser.add_argument("--fixed-drawer", action="store_true", help="Disable the drawer pose randomisation.")
parser.add_argument(
    "--drawer-pose",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "YAW_DEG"),
    help="Fixed drawer pose (implies --fixed-drawer); the env default is the far corner.",
)
parser.add_argument(
    "--init-joint-std",
    type=float,
    default=0.03,
    help=(
        "Std (rad) of the Gaussian jitter added to every robot joint at reset, so demonstrations do not all start"
        " from the identical configuration. 0 disables it."
    ),
)
parser.add_argument("--drawer-z", type=float, default=None, help="Fixed drawer height override (diagnostics).")
parser.add_argument(
    "--probe-drawer-joint",
    action="store_true",
    help="Before each demo, push the drawer joint open directly and report whether it stays.",
)
parser.add_argument("--probe-only", action="store_true", help="Run the drawer joint probe and stop (no grasp).")
parser.add_argument("--record-dir", type=str, default=None, help="Directory for the HDF5 dataset; enables recording.")
parser.add_argument(
    "--stop-after-pull",
    action="store_true",
    help="End and judge each demonstration after the pull settles, before releasing the gripper or returning home.",
)
parser.add_argument("--dataset-name", type=str, default="ur7e_open_drawer", help="HDF5 file name, without extension.")
parser.add_argument("--num-demos", type=int, default=1, help="How many demonstrations to run (and record).")
parser.add_argument("--seed", type=int, default=None, help="Seed for the per-reset drawer placement.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = args.video is not None
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.managers.recorder_manager import DatasetExportMode  # noqa: E402

import isaaclab_arena_environments  # noqa: E402,F401
from data_engine.motion.cumotion.executor import ArmExecutor, EnvActionExecutor, JointActionInterface  # noqa: E402
from data_engine.motion.cumotion.grasps import quat_wxyz_from_matrix  # noqa: E402
from data_engine.motion.cumotion.planner import CumotionArmPlanner  # noqa: E402
from data_engine.motion.cumotion.robot_description import import_cumotion  # noqa: E402
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import (  # noqa: E402
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.embodiments.ur7e.demo_recorders import ur7e_demo_recorder_cfg  # noqa: E402
from isaaclab_arena.embodiments.ur7e.ur7e import (  # noqa: E402
    TCP_OFFSET_FROM_GRIPPER_BASE_M,
    Ur7eJointRecordingActionsCfg,
)
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena_environments.ur7e_workcell_environment import TABLE_SIZE_M, TABLE_TOP_HEIGHT_M  # noqa: E402

DRAWER_EXTENTS_M = (0.20, 0.235, 0.078)
"""Drawer unit bounding box in its own frame (x width, y depth incl. knob, z height); both drawer assets."""
DRAWER_BOX_CENTRE_LOCAL = np.array([0.0, 0.0, DRAWER_EXTENTS_M[2] / 2])
TABLE_OBSTACLE = "/obstacles/table"
DRAWER_OBSTACLE = "/obstacles/drawer"

# ------------------------------------------------------------------------------------- env ---
recording = args.record_dir is not None
# The seed must reach the environment config: Isaac Lab seeds every RNG the reset events draw from
# (torch, numpy, warp) at construction, so seeding torch afterwards would not move the drawer.
_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little") % 2**31
print(f"scene placement seed: {_seed}")
arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras", "--seed", str(_seed)])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
env_cfg_kwargs = dict(
    embodiment=args.embodiment,
    # The environment's own D435 and scene_cam sensors supply the video frames.
    enable_cameras=args.video is not None,
    randomize_drawer_pose=not (args.fixed_drawer or args.drawer_pose is not None),
)
if args.drawer_pose is not None:
    env_cfg_kwargs.update(
        drawer_x=args.drawer_pose[0], drawer_y=args.drawer_pose[1], drawer_yaw_deg=args.drawer_pose[2]
    )
env_cfg = factory._legacy_argparse_cfg_type(**env_cfg_kwargs)
arena_env = factory.build(env_cfg)
# Small per-reset jitter of the start configuration (the embodiment's reset event defaults to 0).
arena_env.embodiment.event_config.reset_robot_joints.params["std"] = float(args.init_joint_std)

if recording:
    # Joint-space actions so cuMotion's planned joint paths pass through the action manager verbatim.
    arena_env.embodiment.action_config = Ur7eJointRecordingActionsCfg()
    _prev_cb = arena_env.env_cfg_callback

    def _recording_env_cfg(patched):
        """Attach the demo recorder; success is judged and exported by this script instead."""
        patched = _prev_cb(patched) if _prev_cb is not None else patched
        patched.recorders = ur7e_demo_recorder_cfg(with_cameras=False)
        patched.recorders.dataset_export_dir_path = args.record_dir
        patched.recorders.dataset_filename = args.dataset_name
        patched.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        patched.env_name = args.env
        patched.terminations.success = None
        patched.episode_length_s = 600.0
        return patched

    arena_env.env_cfg_callback = _recording_env_cfg

builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(arena_args))
env = builder.make_registered().unwrapped
DECIMATION = max(1, round(env.step_dt / env.sim.get_physics_dt()))
if recording:
    args.record_every = max(1, round(args.record_every / DECIMATION))


# ----------------------------------------------------------------------------------- video ---
VIDEO_CAMERAS = {"d435": "realsense_d435", "scene": "scene_cam"}
"""Video stream name -> scene key of the embodiment camera it is taken from."""
writers: dict = {}
if args.video is not None:
    import imageio.v2 as iio  # noqa: E402

    video_path = pathlib.Path(args.video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writers["d435"] = iio.get_writer(video_path, fps=args.fps, codec="libx264", macro_block_size=8)
    writers["scene"] = iio.get_writer(
        video_path.with_name(video_path.stem + "_scene" + video_path.suffix),
        fps=args.fps,
        codec="libx264",
        macro_block_size=8,
    )

env.sim.reset()
env.reset()

step_counter = [0]
frame_counter = [0]


def grab_frame() -> None:
    """Keep one frame in ``--record-every`` from each embodiment camera (reading the data updates it)."""
    step_counter[0] += 1
    if not writers or step_counter[0] % args.record_every != 0:
        return
    for stream, scene_key in VIDEO_CAMERAS.items():
        camera = env.scene[scene_key]
        camera.update(env.sim.get_physics_dt())
        writers[stream].append_data(camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
    frame_counter[0] += 1


# -------------------------------------------------------------------------------- planning ---
embodiment = arena_env.embodiment
planner = CumotionArmPlanner(env, embodiment)
error_m = planner.kinematics_error_m()
print(f"arm kinematics cross-check: {error_m * 1000:.2f} mm")
# The URDF carries the real robot's kinematic calibration while the USD is the nominal model; the
# two agree to ~1.3 mm at the ready pose, which is far below any tolerance the grasp needs.
assert error_m < 3e-3, "cuMotion's kinematics disagree with the simulated robot"
print(f"tool frame correction (sim tool -> cuMotion tool0):\n{np.round(planner.tool_correction, 3)}")

# The work surface, modelled 30 mm below the real top so a grasp near the table stays plannable
# while gross sweeps through it are blocked. The robot is bolted onto this table at its -y edge, so
# the box stops short of the base: with it under the base, the base link's own collision spheres
# (radius 89 mm, centred 50 mm up) sit inside the slab and every start configuration is invalid.
TABLE_OBSTACLE_Y_MIN = -0.30
_table_y_max = TABLE_SIZE_M / 2 + 0.1
planner.add_box_obstacle(
    TABLE_OBSTACLE,
    np.array([0.0, (TABLE_OBSTACLE_Y_MIN + _table_y_max) / 2, TABLE_TOP_HEIGHT_M - 0.03 - 0.06]),
    (TABLE_SIZE_M + 0.2, _table_y_max - TABLE_OBSTACLE_Y_MIN, 0.12),
    safety_tolerance_m=0.0,
)
# The drawer asset (drawer_rr or drawer_rr_gpt56) tells where its knob is and which joint slides.
drawer_asset = arena_env.task.openable_object
drawer_box_size = getattr(drawer_asset, "planning_box_size", DRAWER_EXTENTS_M)
drawer_box_center = np.asarray(getattr(drawer_asset, "planning_box_center", DRAWER_BOX_CENTRE_LOCAL))
drawer = env.scene[drawer_asset.name]
knob_body = list(drawer.data.body_names).index(drawer_asset.knob_body)
knob_offset_local = np.asarray(drawer_asset.knob_offset_local, dtype=np.float64)
drawer_joint = list(drawer.data.joint_names).index(drawer_asset.openable_joint_name)
# Sign of the joint value that opens the drawer: drawer_rr travels 0..+150 mm, drawer_rr_gpt56 0..-155 mm.
_joint_limits = wp.to_torch(drawer.data.joint_pos_limits)[0, drawer_joint].detach().cpu().numpy()
JOINT_OPEN_SIGN = -1.0 if _joint_limits[0] < -1e-6 else 1.0
print(f"drawer '{drawer_asset.name}': knob body '{drawer_asset.knob_body}', joint limits {np.round(_joint_limits, 4)}")


def drawer_pose() -> tuple[np.ndarray, np.ndarray]:
    """Drawer root position and rotation matrix in the world."""
    position = wp.to_torch(drawer.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = wp.to_torch(drawer.data.root_quat_w)[0].detach().cpu().float()
    rotation = math_utils.matrix_from_quat(quat_xyzw.unsqueeze(0))[0].numpy().astype(np.float64)
    return position, rotation


def drawer_box() -> tuple[np.ndarray, np.ndarray]:
    position, rotation = drawer_pose()
    return position + rotation @ drawer_box_center, quat_wxyz_from_matrix(rotation)


_centre, _quat = drawer_box()
planner.add_box_obstacle(DRAWER_OBSTACLE, _centre, drawer_box_size, quat_wxyz=_quat, safety_tolerance_m=0.01)


def refresh_drawer_obstacle() -> None:
    centre, quat = drawer_box()
    planner.world.update_obstacle_transforms(
        [DRAWER_OBSTACLE],
        (
            wp.array([centre.astype(np.float32)], dtype=wp.float32),
            wp.array([quat.astype(np.float32)], dtype=wp.float32),
        ),
    )


def opening_direction() -> np.ndarray:
    """World direction the drawer slides out along (the asset's -y)."""
    _, rotation = drawer_pose()
    return rotation @ np.array([0.0, -1.0, 0.0])


def knob_centre() -> np.ndarray:
    """World centre of the spherical grip: the knob body's origin plus the asset's local knob offset."""
    position = wp.to_torch(drawer.data.body_pos_w)[0, knob_body].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = wp.to_torch(drawer.data.body_quat_w)[0, knob_body].detach().cpu().float()
    rotation = math_utils.matrix_from_quat(quat_xyzw.unsqueeze(0))[0].numpy().astype(np.float64)
    return position + rotation @ knob_offset_local


def openness() -> float:
    """How far the drawer is pulled out, metres (positive regardless of the joint's sign convention)."""
    return JOINT_OPEN_SIGN * float(wp.to_torch(drawer.data.joint_pos)[0, drawer_joint])


if recording:
    interface = JointActionInterface(env)
    executor = EnvActionExecutor(env, planner, interface, "arm_action", "gripper_action", on_step=grab_frame)
else:
    interface = None
    executor = ArmExecutor(env, planner, on_step=grab_frame)


class _WaypointPath:
    """The subset of cuMotion's ``Path`` interface ``ArmExecutor.follow`` uses."""

    def __init__(self, waypoints: np.ndarray):
        self._waypoints = np.asarray(waypoints, dtype=np.float64)

    def get_waypoints(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._waypoints


def top_down_tool_frame(pull: np.ndarray, spin_deg: float) -> np.ndarray:
    """Canonical tool orientation pointing straight down with the jaw axis (+y) across the pull direction.

    Args:
        pull: Horizontal world direction the drawer opens along.
        spin_deg: Extra yaw about the vertical; 180 flips the wrist for the same pinch.
    """
    z = np.array([0.0, 0.0, -1.0])
    horizontal = np.array([pull[0], pull[1], 0.0])
    horizontal /= np.linalg.norm(horizontal)
    y = np.cross(np.array([0.0, 0.0, 1.0]), horizontal)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    spin = np.radians(spin_deg)
    c, s_ = np.cos(spin), np.sin(spin)
    x, y = c * x + s_ * y, -s_ * x + c * y
    return quat_wxyz_from_matrix(np.column_stack([x, y, z]))


def solve_ik(position: np.ndarray, quat_wxyz: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
    """Exact IK for a world-frame tool pose, seeded near ``seed``."""
    cumotion = import_cumotion()
    rotation = cumotion.Rotation3(*np.asarray(planner.to_tool_frame(quat_wxyz), dtype=np.float64)).matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=np.float64) - planner.base_pos
    ik_cfg = cumotion.IkConfig()
    ik_cfg.cspace_seeds = [np.asarray(seed, dtype=np.float64)]
    result = cumotion.solve_ik(planner.kinematics, cumotion.Pose3(transform), planner.cfg.tool_frame, ik_cfg)
    if not result.success:
        return None
    for attr in ("cspace_position", "cspace_positions", "q", "joint_positions", "solution"):
        if hasattr(result, attr):
            return np.asarray(getattr(result, attr), dtype=np.float64).reshape(-1)
    raise AttributeError(f"cuMotion IkResult exposes no known solution attribute: {dir(result)}")


def straight_line_path(start_q: np.ndarray, start: np.ndarray, end: np.ndarray, quat_wxyz: np.ndarray, segments: int):
    """IK waypoints along a straight tool-space line, or None if a waypoint has no solution."""
    waypoints = [np.asarray(start_q, dtype=np.float64)]
    for i in range(1, segments + 1):
        target = start + (end - start) * (i / segments)
        q = solve_ik(target, quat_wxyz, waypoints[-1])
        if q is None:
            print(f"    no IK at pull waypoint {i}/{segments}: {np.round(target, 4)}")
            return None
        if np.max(np.abs(q - waypoints[-1])) > 0.5:
            print(f"    IK branch jump at pull waypoint {i}/{segments} ({np.max(np.abs(q - waypoints[-1])):.2f} rad)")
            return None
        waypoints.append(q)
    return _WaypointPath(np.stack(waypoints))


def jaw_axis_check(quat_wxyz: np.ndarray) -> None:
    """Report the measured finger-separation axis against the commanded jaw axis, after a close."""
    robot = planner.robot
    names = list(robot.data.body_names)
    left = wp.to_torch(robot.data.body_pos_w)[0, names.index("left_inner_finger")].cpu().numpy()
    right = wp.to_torch(robot.data.body_pos_w)[0, names.index("right_inner_finger")].cpu().numpy()
    separation = right - left
    if np.linalg.norm(separation) < 1e-4:
        print("  jaw axis: finger frames coincide, nothing to measure")
        return
    separation /= np.linalg.norm(separation)
    commanded_y = math_utils.matrix_from_quat(torch.tensor([[quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]]))[
        0
    ].numpy()[:, 1]
    print(
        f"  jaw axis: measured {np.round(separation, 3)} vs commanded +y {np.round(commanded_y, 3)}"
        f" (|cos| = {abs(float(np.dot(separation, commanded_y))):.3f}; 1 means the jaws are where the pose asked)"
    )


def pull_trace(label: str) -> str:
    root = wp.to_torch(drawer.data.root_pos_w)[0].cpu().numpy()
    return (
        f"{label}: tool {np.round(planner.tool_position(), 3)} knob {np.round(knob_centre(), 3)}"
        f" joint {openness() * 1000:.1f} mm carcass {np.round(root, 3)}"
    )


def probe_drawer_joint() -> list[str]:
    """Write the drawer joint to 50 mm, let physics run, and report where it ends up and how it is driven."""
    j = drawer_joint
    lines = [
        "drawer joint: stiffness %s damping %s friction %s limits %s effort_limit %s"
        % (
            np.round(wp.to_torch(drawer.data.joint_stiffness)[0, j].item(), 3),
            np.round(wp.to_torch(drawer.data.joint_damping)[0, j].item(), 3),
            (
                np.round(wp.to_torch(drawer.data.joint_friction_coeff)[0, j].item(), 3)
                if hasattr(drawer.data, "joint_friction_coeff")
                else "?"
            ),
            np.round(wp.to_torch(drawer.data.joint_pos_limits)[0, j].cpu().numpy(), 3),
            (
                np.round(wp.to_torch(drawer.data.joint_effort_limits)[0, j].item(), 3)
                if hasattr(drawer.data, "joint_effort_limits")
                else "?"
            ),
        ),
        f"drawer joint target: {np.round(wp.to_torch(drawer.data.joint_pos_target)[0, j].item(), 4)}",
    ]
    ids = torch.tensor([j], device=env.device)
    drawer.write_joint_position_to_sim(torch.tensor([[JOINT_OPEN_SIGN * 0.05]], device=env.device), joint_ids=ids)
    drawer.write_joint_velocity_to_sim(torch.zeros((1, 1), device=env.device), joint_ids=ids)
    executor.step(steps=30)
    lines.append(
        f"after writing 50 mm and 30 steps: joint {openness() * 1000:.1f} mm, knob {np.round(knob_centre(), 3)}"
    )
    drawer.write_joint_position_to_sim(torch.zeros((1, 1), device=env.device), joint_ids=ids)
    drawer.write_joint_velocity_to_sim(torch.zeros((1, 1), device=env.device), joint_ids=ids)
    executor.step(steps=30)
    lines.append(f"after writing back 0 mm: joint {openness() * 1000:.1f} mm")
    return lines


def run_demo() -> list[str]:
    """One open-drawer demonstration; returns a report."""
    report: list[str] = []
    if args.probe_drawer_joint or args.probe_only:
        report.extend(probe_drawer_joint())
        if args.probe_only:
            return report
    home_q = planner.joint_positions()
    executor.open_gripper()

    pull = opening_direction()
    knob = knob_centre()
    report.append(
        f"knob at {np.round(knob, 4)}, pull direction {np.round(pull, 3)}, openness {openness() * 1000:.1f} mm"
    )

    # Tool (flange) pose for the grasp: fingers down, tool centre point just below and in front of
    # the knob centre. Offer the two wrist spins; the first with an IK solution for both poses and an
    # executable plan to the standoff wins.
    tcp_grasp = knob + pull * args.knob_forward_shift + np.array([0.0, 0.0, -args.grasp_depth])
    tool_grasp = tcp_grasp + np.array([0.0, 0.0, TCP_OFFSET_FROM_GRIPPER_BASE_M])
    tool_standoff = tool_grasp + np.array([0.0, 0.0, args.standoff])
    planner.set_obstacle_enabled(DRAWER_OBSTACLE, True)
    chosen = None
    for spin_deg in args.spins:
        quat = top_down_tool_frame(pull, spin_deg)
        reachable = planner.ik_reachable(tool_standoff, quat), planner.ik_reachable(tool_grasp, quat)
        if not all(reachable):
            print(f"    spin {spin_deg:4.0f}: IK standoff {reachable[0]}, grasp {reachable[1]} -- skipped")
            continue
        standoff = planner.plan_pose(planner.joint_positions(), tool_standoff, quat)
        if standoff is None:
            print(f"    spin {spin_deg:4.0f}: IK ok, no path to the standoff")
            continue
        if not standoff.is_executable():
            print(
                f"    spin {spin_deg:4.0f}: path not executable (margin {standoff.limit_margin_rad:.2f} rad,"
                f" travel {standoff.max_travel_rad:.2f} rad)"
            )
            continue
        chosen = (spin_deg, quat, standoff)
        break
    if chosen is None:
        report.append("aborted: no wrist spin gives an executable plan to the standoff")
        return report
    spin_deg, quat, standoff = chosen
    report.append(f"wrist spin {spin_deg:.0f} deg, standoff plan margin {standoff.limit_margin_rad:.2f} rad")
    executor.follow(standoff.path)
    report.append(
        f"standoff reached, tool error {np.linalg.norm(planner.tool_position() - tool_standoff) * 1000:.1f} mm"
    )

    planner.set_obstacle_enabled(DRAWER_OBSTACLE, False)
    advance = planner.plan_pose(planner.joint_positions(), tool_grasp, quat)
    if advance is None or not advance.is_executable():
        report.append("aborted: no executable plan onto the knob")
        return report
    q_grasp = executor.follow(advance.path, speed=0.2)
    report.append(f"on knob, tool error {np.linalg.norm(planner.tool_position() - tool_grasp) * 1000:.1f} mm")

    executor.close_gripper(hold_arm_at=q_grasp)
    executor.step(arm_target=q_grasp, steps=executor.settle_steps)
    jaw_axis_check(quat)
    finger = float(wp.to_torch(planner.robot.data.joint_pos)[0, planner.gripper_joint_ids[0]])
    knob_before_pull = knob_centre()
    report.append(
        f"closed: finger_joint {finger:.3f} rad (target {planner.cfg.gripper_closed_pos:.3f}),"
        " knob-TCP distance"
        f" {np.linalg.norm(knob_before_pull - (planner.tool_position() + np.array([0, 0, -TCP_OFFSET_FROM_GRIPPER_BASE_M]))) * 1000:.1f} mm"
    )

    pull_path = straight_line_path(
        planner.joint_positions(), tool_grasp, tool_grasp + pull * args.pull_distance, quat, args.pull_segments
    )
    if pull_path is None:
        report.append("aborted: the straight pull has no IK solution")
        executor.open_gripper(hold_arm_at=q_grasp)
        return report
    waypoints = pull_path.numpy()
    report.append(pull_trace("pull start") + f"; q {np.round(waypoints[0], 3)}")
    q_pulled = waypoints[0]
    for k in range(1, len(waypoints)):
        q_pulled = executor.follow(_WaypointPath(waypoints[k - 1 : k + 1]), speed=args.pull_speed, settle_steps=1)
        if k in (1, len(waypoints) - 1):
            lag = planner.joint_positions() - waypoints[k]
            report.append(
                pull_trace(f"pull wp {k}/{len(waypoints) - 1}") + f"; servo lag max {np.max(np.abs(lag)):.3f} rad"
            )
    executor.step(arm_target=q_pulled, steps=executor.settle_steps)
    report.append(
        f"pulled: drawer openness {openness() * 1000:.1f} mm of {args.pull_distance * 1000:.0f} mm commanded;"
        f" knob moved {np.linalg.norm(knob_centre() - knob_before_pull) * 1000:.1f} mm"
    )

    if args.stop_after_pull:
        report.append("stopped after pull with gripper closed; release, retreat and home omitted")
        return report

    executor.open_gripper(hold_arm_at=q_pulled)
    retreat_target = planner.tool_position() + np.array([0.0, 0.0, args.standoff])
    retreat = planner.plan_pose(planner.joint_positions(), retreat_target, quat)
    if retreat is not None and retreat.is_executable():
        executor.follow(retreat.path, speed=0.2)
    else:
        report.append("could not plan the retreat; going home directly")
    refresh_drawer_obstacle()
    planner.set_obstacle_enabled(DRAWER_OBSTACLE, True)
    home = planner.plan_config(planner.joint_positions(), home_q)
    if home is not None and home.is_executable():
        executor.follow(home.path, speed=0.2)
    else:
        report.append("could not plan the way home")
    report.append(f"final drawer openness {openness() * 1000:.1f} mm")
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
        executor._gripper_target = planner.cfg.gripper_open_pos
    # Openable's reset writes position only. A passive slide otherwise carries
    # the previous pull velocity into the next episode and can open ungrasped.
    drawer.write_joint_velocity_to_sim_index(
        velocity=torch.zeros((1, 1), device=env.device),
        joint_ids=torch.tensor([drawer_joint], dtype=torch.int32, device=env.device),
    )
    executor.step(steps=max(1, 30 // (DECIMATION if recording else 1)))
    refresh_drawer_obstacle()

    try:
        report = run_demo()
    except RuntimeError as error:
        report = [f"aborted: {error}"]
    for line in report:
        print(f"  {line}")

    success = bool(task.openable_object.is_open(env, threshold=env_cfg.openness_threshold)[0].item())
    success = success and not any(line.startswith("aborted:") for line in report)
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
for writer in writers.values():
    writer.close()
if writers:
    print(f"  wrote {frame_counter[0]} frames per camera next to {args.video}")
simulation_app.close()
