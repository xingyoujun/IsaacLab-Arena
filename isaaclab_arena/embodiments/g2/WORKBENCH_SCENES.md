# G2 workbench scene previews

Three independent scene previews reuse Arena's object library and the G2
stack-bowls table, room, robot and keyboard controller. They have **no task
success predicate, automatic policy or collection pipeline yet**.

| Environment | Arena objects | Intended next step |
| --- | --- | --- |
| `g2_peg_insert` | `peg`, `hole` | Confirm these provisional assembly assets match the desired sleeve task |
| `g2_organize_tools` | `red_hammer_robolab`, `spring_clamp_ycb_robolab`, `cordless_drill_ycb_robolab`, two `bin_b04_vomp_robolab` bins | Define tool destinations and grasp strategy |
| `g2_turn_knob` | `stand_mixer`, including `knob_speed_joint` | Define the desired angle/level and test physical grasping |

The peg/hole pair uses Arena's existing 3x scale: the nominal 8 mm peg becomes
24 mm wide and 150 mm tall. It is a provisional insertion scene, not a claim
that the original Agibot sleeve asset has been ported. Both bins use an explicit
1.3x scale to accommodate the long hammer. Other task objects retain their
library scales. External source USDs are not edited.

The tabletop stays at world z=0, 0.75 m above the room floor. The robot base is
at (-0.65, 0, -0.75). Objects are placed using their rotated bounding boxes so
their lowest geometry starts 3 mm above the tabletop. Tools occupy the near
edge; the two bins occupy the far half. The scenes preserve the library assets'
rigid-body and articulation settings.

## Launch

Use the dev-container skill to discover this clone's running container. Run as
the host user inside it, with the G2 assets available at `/datasets/GenieSimAssets`
(or set `GENIESIM_ASSETS_DIR`). Arena library USDs must be accessible through
the configured asset service/cache.

```bash
cd /workspaces/isaaclab_arena
/isaac-sim/python.sh submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
  --device cpu --viz kit --num_envs 1 \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task g2_organize_tools --teleop_device keyboard --arena_teleop_device keyboard
```

Replace `g2_organize_tools` with `g2_peg_insert` or `g2_turn_knob` to switch scenes.
Tab selects the arm; K toggles its gripper. Movement and rotation keys match
[the G2 keyboard reference](README.md). R resets the scene. Pass
`--enable_cameras` to enable the native head and wrist sensors.

Common scene options are `--table_height_m`, `--robot_position x y z`,
`--arm_mode dual_arm|left|right`, `--hdr`, and `--light_intensity`.

Scene definitions and explicit layouts live in
`isaaclab_arena_environments/g2_workbench_environments.py`.

## Validation

On 2026-09-16, each scene was built in the workspace's container with CPU
physics and native cameras enabled. Each ran 90 idle control steps, then reset
and ran another 30 steps. Object root positions remained on the table, all three
scenes exposed the 14-value dual-arm action interface, and the mixer exposed
`knob_speed_joint`. Head and wrist images were captured. Scoped pre-commit
checks passed. Full-suite tests, grasp reachability and task completion have not
been validated for these new scenes.

The diagnostic process prepended
`/workspaces/isaaclab_arena/submodules/IsaacLab/source/isaaclab` to `PYTHONPATH`
to avoid the current container's Isaac Lab namespace import issue during Kit
startup. If startup reports that `_deprioritize_prebundle_paths` cannot be
imported, prefix the Python command above with
`PYTHONPATH=/workspaces/isaaclab_arena/submodules/IsaacLab/source/isaaclab`.
