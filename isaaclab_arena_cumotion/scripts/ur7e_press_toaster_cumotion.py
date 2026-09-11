# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Press the toaster's carriage lever on the UR7e workcell with cuMotion-planned motions, optionally recording demos.

One demonstration is: close the gripper, plan to a standoff on the approach line above and outside the
lever paddle, advance onto the paddle with the tool tilted from the vertical (pointing down and into the
toaster, so the closed fingertips rest on the paddle while the hand stays clear of the housing), push
straight down by ``--press-depth``, retreat along the approach line and return home. Every combination of
``--tilts`` and ``--spins`` is tried; only candidates whose standoff, contact and pressed configurations
are free of self-collision qualify, and the one needing the least joint travel from the start pose is executed.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/ur7e_press_toaster_cumotion.py \\
        --video /home/ubuntu/playground/rr_ur/sim_renders/toaster_v1_press.mp4

    # record demonstrations (states only; re-render cameras afterwards)
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 .venv/bin/python \\
        isaaclab_arena_cumotion/scripts/ur7e_press_toaster_cumotion.py \\
        --record-dir /home/ubuntu/playground/datasets/ur7e_press_toaster --num-demos 10 --seed 0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="ur7e_press_toaster")
parser.add_argument("--embodiment", type=str, default="ur7e_robotiq_joint_pos")
parser.add_argument("--video", type=str, default=None, help="Write the D435 view here and a scene view next to it.")
parser.add_argument("--record-every", type=int, default=3, help="Keep one video frame in N physics steps.")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--tilts",
    type=float,
    nargs="+",
    default=(60.0, 70.0, 80.0, 90.0, 50.0),
    help="Tool tilts (deg) from the vertical, towards the toaster, to offer.",
)
parser.add_argument("--press-depth", type=float, default=0.045, help="How far to push the lever down, metres.")
parser.add_argument("--standoff", type=float, default=0.08, help="Pre-press distance along the approach line, metres.")
parser.add_argument(
    "--contact-outward",
    type=float,
    default=0.004,
    help="Fingertip contact this far outside the paddle centre (away from the housing), metres.",
)
parser.add_argument(
    "--contact-sink", type=float, default=0.002, help="Command the fingertip this far into the paddle top."
)
parser.add_argument("--press-speed", type=float, default=0.15, help="Playback speed fraction of the press.")
parser.add_argument(
    "--spins",
    type=float,
    nargs="+",
    default=(0.0, 180.0, 90.0, -90.0),
    help="Rolls (deg) about the approach axis to offer; 0 and 180 swap the two fingers.",
)
parser.add_argument("--press-segments", type=int, default=6, help="IK waypoints along the straight press.")
parser.add_argument("--fixed-toaster", action="store_true", help="Disable the toaster pose randomisation.")
parser.add_argument(
    "--toaster-pose",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "YAW_DEG"),
    help="Fixed toaster pose (implies --fixed-toaster); the env default is the far corner.",
)
parser.add_argument(
    "--init-joint-std",
    type=float,
    default=0.03,
    help="Std (rad) of the Gaussian jitter added to every robot joint at reset. 0 disables it.",
)
parser.add_argument(
    "--probe-lever",
    action="store_true",
    help="Write the lever joint to 40 mm before the demo and report where it settles.",
)
parser.add_argument(
    "--no-pedestal",
    action="store_true",
    help="Put the toaster straight on the table (env default: on the drawer unit).",
)
parser.add_argument("--record-dir", type=str, default=None, help="Directory for the HDF5 dataset; enables recording.")
parser.add_argument(
    "--record-retreat",
    action="store_true",
    help="Keep recording through the retreat and the return home (by default a demo ends with the lever pressed).",
)
parser.add_argument("--dataset-name", type=str, default="ur7e_press_toaster", help="HDF5 file name, without extension.")
parser.add_argument("--num-demos", type=int, default=1, help="How many demonstrations to run (and record).")
parser.add_argument("--seed", type=int, default=None, help="Seed for the per-reset toaster placement.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
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
from isaaclab_arena_cumotion.executor import ArmExecutor, EnvActionExecutor, JointActionInterface  # noqa: E402
from isaaclab_arena_cumotion.grasps import quat_wxyz_from_matrix  # noqa: E402
from isaaclab_arena_cumotion.planner import CumotionArmPlanner  # noqa: E402
from isaaclab_arena_cumotion.robot_description import import_cumotion  # noqa: E402
from isaaclab_arena_environments.ur7e_press_toaster_environment import (  # noqa: E402
    LEVER_PADDLE_CENTRE_LOCAL,
    LEVER_TRAVEL_M,
    PEDESTAL_HEIGHT_M,
    TOASTER_EXTENTS_M,
    ToasterRR,
)
from isaaclab_arena_environments.ur7e_workcell_environment import TABLE_SIZE_M, TABLE_TOP_HEIGHT_M  # noqa: E402

TOASTER_KEY = "toaster_rr"
LEVER_BODY = "carriage_lever"
LEVER_BODY_ORIGIN_LOCAL = np.array([-0.002, -0.106, 0.124])
"""Origin of the carriage_lever body in the toaster frame, at rest."""
PADDLE_FROM_LEVER_BODY = np.array(LEVER_PADDLE_CENTRE_LOCAL) - LEVER_BODY_ORIGIN_LOCAL
"""Paddle centre relative to the carriage_lever body origin (moves with the lever)."""
PADDLE_HALF_HEIGHT_M = 0.0045
TOASTER_BOX_CENTRE_LOCAL = np.array([0.001, 0.0, TOASTER_EXTENTS_M[2] / 2])
PEDESTAL_EXTENTS_M = (0.20, 0.235, PEDESTAL_HEIGHT_M)
"""Drawer unit bounding box (x width, y depth, z height) when it serves as the pedestal."""
TABLE_OBSTACLE = "/obstacles/table"
TOASTER_OBSTACLE = "/obstacles/toaster"

# ------------------------------------------------------------------------------------- env ---
recording = args.record_dir is not None
# The seed must reach the environment config: Isaac Lab seeds every RNG the reset events draw from
# (torch, numpy, warp) at construction, so seeding torch afterwards would not move the toaster.
_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little") % 2**31
print(f"scene placement seed: {_seed}")
arena_args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras", "--seed", str(_seed)])
factory = EnvironmentRegistry().get_component_by_name(args.env)()
env_cfg_kwargs = dict(
    embodiment=args.embodiment,
    # The environment's own D435 and scene_cam sensors supply the video frames.
    enable_cameras=args.video is not None,
    randomize_toaster_pose=not (args.fixed_toaster or args.toaster_pose is not None),
    pedestal=not args.no_pedestal,
)
if args.toaster_pose is not None:
    env_cfg_kwargs.update(
        toaster_x=args.toaster_pose[0], toaster_y=args.toaster_pose[1], toaster_yaw_deg=args.toaster_pose[2]
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
assert error_m < 3e-3, "cuMotion's kinematics disagree with the simulated robot"

# The work surface, modelled 30 mm below the real top so the tool can work near the table while gross
# sweeps through it are blocked; it stops short of the robot base (see ur7e_open_drawer_cumotion.py).
TABLE_OBSTACLE_Y_MIN = -0.30
_table_y_max = TABLE_SIZE_M / 2 + 0.1
planner.add_box_obstacle(
    TABLE_OBSTACLE,
    np.array([0.0, (TABLE_OBSTACLE_Y_MIN + _table_y_max) / 2, TABLE_TOP_HEIGHT_M - 0.03 - 0.06]),
    (TABLE_SIZE_M + 0.2, _table_y_max - TABLE_OBSTACLE_Y_MIN, 0.12),
    safety_tolerance_m=0.0,
)
_inspector = import_cumotion().create_robot_world_inspector(planner.robot_description)


def self_collision_frames(q: np.ndarray) -> list[tuple[str, str]]:
    """Pairs of robot frames whose collision spheres overlap at arm configuration ``q``."""
    return list(_inspector.frames_in_self_collision(np.asarray(q, dtype=np.float64).reshape(-1, 1)))


toaster = env.scene[TOASTER_KEY]
lever_body = list(toaster.data.body_names).index(LEVER_BODY)
lever_joint = list(toaster.data.joint_names).index(ToasterRR.openable_joint_name)


def toaster_pose() -> tuple[np.ndarray, np.ndarray]:
    """Toaster root position and rotation matrix in the world."""
    position = wp.to_torch(toaster.data.root_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = wp.to_torch(toaster.data.root_quat_w)[0].detach().cpu().float()
    rotation = math_utils.matrix_from_quat(quat_xyzw.unsqueeze(0))[0].numpy().astype(np.float64)
    return position, rotation


def toaster_box() -> tuple[np.ndarray, np.ndarray]:
    """Centre and orientation of a box around the toaster, and the pedestal under it if there is one."""
    position, rotation = toaster_pose()
    centre_local = TOASTER_BOX_CENTRE_LOCAL.copy()
    if not args.no_pedestal:
        # Same footprint as the pedestal (wider than the toaster) from the table up to the toaster top.
        centre_local[2] = (TOASTER_EXTENTS_M[2] - PEDESTAL_HEIGHT_M) / 2
    return position + rotation @ centre_local, quat_wxyz_from_matrix(rotation)


OBSTACLE_EXTENTS_M = (
    (max(TOASTER_EXTENTS_M[0], PEDESTAL_EXTENTS_M[0]), TOASTER_EXTENTS_M[1], TOASTER_EXTENTS_M[2] + PEDESTAL_HEIGHT_M)
    if not args.no_pedestal
    else TOASTER_EXTENTS_M
)
_centre, _quat = toaster_box()
planner.add_box_obstacle(TOASTER_OBSTACLE, _centre, OBSTACLE_EXTENTS_M, quat_wxyz=_quat, safety_tolerance_m=0.01)


def refresh_toaster_obstacle() -> None:
    centre, quat = toaster_box()
    planner.world.update_obstacle_transforms(
        [TOASTER_OBSTACLE],
        (
            wp.array([centre.astype(np.float32)], dtype=wp.float32),
            wp.array([quat.astype(np.float32)], dtype=wp.float32),
        ),
    )


def outward_direction() -> np.ndarray:
    """World direction the lever paddle sticks out along (the asset's -y, away from the housing)."""
    _, rotation = toaster_pose()
    return rotation @ np.array([0.0, -1.0, 0.0])


def paddle_centre() -> np.ndarray:
    _, rotation = toaster_pose()
    position = wp.to_torch(toaster.data.body_pos_w)[0, lever_body].detach().cpu().numpy().astype(np.float64)
    return position + rotation @ PADDLE_FROM_LEVER_BODY


def lever_travel() -> float:
    return float(wp.to_torch(toaster.data.joint_pos)[0, lever_joint])


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


def tilted_tool_frame(outward: np.ndarray, tilt_deg: float, spin_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Tool orientation pointing down and into the toaster, jaw axis (+y) along the paddle's length.

    Args:
        outward: Horizontal world direction the paddle sticks out along.
        tilt_deg: Angle of the approach axis from the vertical; 0 is straight down.
        spin_deg: Extra roll about the approach axis; 180 swaps the two fingers.

    Returns:
        The orientation as a wxyz quaternion and the unit approach direction (tool +z).
    """
    horizontal = np.array([outward[0], outward[1], 0.0])
    horizontal /= np.linalg.norm(horizontal)
    tilt = np.radians(tilt_deg)
    z = -horizontal * np.sin(tilt) + np.array([0.0, 0.0, -np.cos(tilt)])
    y = np.cross(np.array([0.0, 0.0, 1.0]), horizontal)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    spin = np.radians(spin_deg)
    c, s_ = np.cos(spin), np.sin(spin)
    x, y = c * x + s_ * y, -s_ * x + c * y
    return quat_wxyz_from_matrix(np.column_stack([x, y, z])), z


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
            print(f"    no IK at press waypoint {i}/{segments}: {np.round(target, 4)}")
            return None
        if np.max(np.abs(q - waypoints[-1])) > 0.5:
            print(f"    IK branch jump at press waypoint {i}/{segments} ({np.max(np.abs(q - waypoints[-1])):.2f} rad)")
            return None
        pairs = self_collision_frames(q)
        if pairs:
            print(f"    self-collision at press waypoint {i}/{segments}: {pairs}")
            return None
        waypoints.append(q)
    return _WaypointPath(np.stack(waypoints))


def press_trace(label: str) -> str:
    tcp = planner.tool_position()
    robot = planner.robot
    names = list(robot.data.body_names)
    tips = [
        wp.to_torch(robot.data.body_pos_w)[0, names.index(n)].cpu().numpy()
        for n in ("left_inner_finger", "right_inner_finger")
    ]
    torque = wp.to_torch(robot.data.applied_torque)[0, planner.arm_joint_ids].cpu().numpy()
    return (
        f"{label}: tool {np.round(tcp, 3)} paddle {np.round(paddle_centre(), 3)} lever {lever_travel() * 1000:.1f} mm"
        f" fingers {np.round(tips[0], 3)} {np.round(tips[1], 3)} arm torque {np.round(torque, 1)}"
    )


def probe_lever() -> list[str]:
    """Write the lever joint to 40 mm, let physics run, and report where it ends up (a jam test)."""
    ids = torch.tensor([lever_joint], device=env.device)
    lines = [
        f"lever probe: limits {np.round(wp.to_torch(toaster.data.joint_pos_limits)[0, lever_joint].cpu().numpy(), 4)}"
    ]
    toaster.write_joint_position_to_sim(torch.tensor([[0.04]], device=env.device), joint_ids=ids)
    toaster.write_joint_velocity_to_sim(torch.zeros((1, 1), device=env.device), joint_ids=ids)
    executor.step(steps=30)
    lines.append(f"lever probe: after writing 40 mm and 30 steps the joint reads {lever_travel() * 1000:.1f} mm")
    toaster.write_joint_position_to_sim(torch.zeros((1, 1), device=env.device), joint_ids=ids)
    toaster.write_joint_velocity_to_sim(torch.zeros((1, 1), device=env.device), joint_ids=ids)
    executor.step(steps=30)
    lines.append(f"lever probe: after writing back 0 mm the joint reads {lever_travel() * 1000:.1f} mm")
    return lines


def run_demo() -> list[str]:
    """One press-toaster demonstration; returns a report."""
    report: list[str] = []
    if args.probe_lever:
        report.extend(probe_lever())
    home_q = planner.joint_positions()
    executor.close_gripper()

    outward = outward_direction()
    paddle = paddle_centre()
    report.append(
        f"paddle at {np.round(paddle, 4)}, outward {np.round(outward, 3)}, lever {lever_travel() * 1000:.1f} mm"
    )

    # Fingertip contact point on the paddle's top, slightly outside its centre; the tool (flange) sits
    # back along the tilted approach axis by the closed-finger TCP offset.
    contact = paddle + outward * args.contact_outward + np.array([0.0, 0.0, PADDLE_HALF_HEIGHT_M - args.contact_sink])
    planner.set_obstacle_enabled(TOASTER_OBSTACLE, True)
    q_now = planner.joint_positions()
    candidates = []
    for tilt_deg in args.tilts:
        for spin_deg in args.spins:
            label = f"tilt {tilt_deg:3.0f} spin {spin_deg:4.0f}"
            quat, approach = tilted_tool_frame(outward, tilt_deg, spin_deg)
            tool_contact = contact - approach * TCP_OFFSET_FROM_GRIPPER_BASE_M
            tool_standoff = tool_contact - approach * args.standoff
            tool_pressed = tool_contact + np.array([0.0, 0.0, -args.press_depth])
            q_standoff = solve_ik(tool_standoff, quat, q_now)
            if q_standoff is None:
                print(f"    {label}: no IK at the standoff")
                continue
            q_contact = solve_ik(tool_contact, quat, q_standoff)
            q_pressed = solve_ik(tool_pressed, quat, q_contact) if q_contact is not None else None
            if q_contact is None or q_pressed is None:
                print(f"    {label}: no IK at the contact or pressed pose")
                continue
            colliding = [pairs for pairs in map(self_collision_frames, (q_standoff, q_contact, q_pressed)) if pairs]
            if colliding:
                print(f"    {label}: self-collision {colliding[0]}")
                continue
            # Plan to the IK configuration, not the pose: a pose target lets cuMotion pick another IK branch
            # (a folded one, wrist beside the shoulder, that then pressed the lever with the shoulder alone).
            standoff = planner.plan_config(q_now, q_standoff)
            if standoff is None:
                print(f"    {label}: IK ok, no path to the standoff configuration")
                continue
            if not standoff.is_executable():
                print(
                    f"    {label}: path not executable (margin {standoff.limit_margin_rad:.2f} rad,"
                    f" travel {standoff.max_travel_rad:.2f} rad)"
                )
                continue
            if self_collision_frames(standoff.q_end):
                print(f"    {label}: planned standoff configuration self-collides")
                continue
            margin = min(
                standoff.limit_margin_rad,
                float(
                    np.min(np.minimum(q_pressed - planner.joint_limits[:, 0], planner.joint_limits[:, 1] - q_pressed))
                ),
            )
            travel = float(np.max(np.abs(q_pressed - q_now)))
            print(f"    {label}: ok, limit margin {margin:.2f} rad, travel {travel:.2f} rad")
            candidates.append((
                travel,
                margin,
                tilt_deg,
                spin_deg,
                quat,
                approach,
                standoff,
                tool_standoff,
                tool_contact,
                tool_pressed,
            ))
    if not candidates:
        report.append("aborted: no tilt/spin gives a self-collision-free, executable plan")
        return report
    # Least joint travel from the start pose wins: it keeps the arm in the ready pose's extended IK branch.
    # The branch with the most limit margin was a folded one (wrist beside the shoulder) that pressed poorly.
    _, _, tilt_deg, spin_deg, quat, approach, standoff, tool_standoff, tool_contact, tool_pressed = min(
        candidates, key=lambda c: (round(c[0], 1), -c[1])
    )
    report.append(
        f"tilt {tilt_deg:.0f} deg, wrist spin {spin_deg:.0f} deg ({len(candidates)} candidates),"
        f" standoff plan margin {standoff.limit_margin_rad:.2f} rad"
    )
    executor.follow(standoff.path)
    report.append(
        f"standoff reached, tool error {np.linalg.norm(planner.tool_position() - tool_standoff) * 1000:.1f} mm"
    )

    planner.set_obstacle_enabled(TOASTER_OBSTACLE, False)
    advance = straight_line_path(planner.joint_positions(), tool_standoff, tool_contact, quat, args.press_segments)
    if advance is None:
        report.append("aborted: the approach onto the paddle has no IK solution")
        return report
    q_contact = executor.follow(advance, speed=0.2)
    report.append(
        press_trace("on paddle")
        + f", tool error {np.linalg.norm(planner.tool_position() - tool_contact) * 1000:.1f} mm"
    )

    press = straight_line_path(planner.joint_positions(), tool_contact, tool_pressed, quat, args.press_segments)
    if press is None:
        report.append("aborted: the straight press has no IK solution")
        return report
    waypoints = press.numpy()
    q_pressed = waypoints[0]
    for k in range(1, len(waypoints)):
        q_pressed = executor.follow(_WaypointPath(waypoints[k - 1 : k + 1]), speed=args.press_speed, settle_steps=1)
        if k in (1, len(waypoints) - 1):
            lag = planner.joint_positions() - waypoints[k]
            report.append(
                press_trace(f"press wp {k}/{len(waypoints) - 1}") + f"; servo lag max {np.max(np.abs(lag)):.3f} rad"
            )
    executor.step(arm_target=q_pressed, steps=executor.settle_steps)
    report.append(
        f"pressed: lever {lever_travel() * 1000:.1f} mm of {LEVER_TRAVEL_M * 1000:.0f} mm travel"
        f" ({args.press_depth * 1000:.0f} mm commanded)"
    )
    if recording and not args.record_retreat:
        # The demonstration ends with the lever down; lifting the hand off again is not part of the task.
        return report

    retreat = straight_line_path(
        planner.joint_positions(), planner.tool_position(), planner.tool_position() - approach * args.standoff, quat, 3
    )
    if retreat is not None:
        executor.follow(retreat, speed=0.2)
    else:
        report.append("could not plan the retreat; going home directly")
    refresh_toaster_obstacle()
    planner.set_obstacle_enabled(TOASTER_OBSTACLE, True)
    home = planner.plan_config(planner.joint_positions(), home_q)
    if home is not None and home.is_executable():
        executor.follow(home.path, speed=0.2)
    else:
        report.append("could not plan the way home")
    report.append(f"final lever travel {lever_travel() * 1000:.1f} mm")
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
    executor.step(steps=max(1, 30 // (DECIMATION if recording else 1)))
    refresh_toaster_obstacle()

    try:
        report = run_demo()
    except RuntimeError as error:
        report = [f"aborted: {error}"]
    for line in report:
        print(f"  {line}")

    success = bool(task.openable_object.is_open(env, threshold=env_cfg.pressed_threshold)[0].item())
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
