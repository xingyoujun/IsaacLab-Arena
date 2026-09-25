# Articraft drawer qualification

Source: `/home/ubuntu/playground/rr_ur/drawer_articraft/model.urdf`, with both OBJ
meshes under its `assets/meshes/` directory. The exported single-file USD is
`/home/ubuntu/playground/rr_ur/articraft_drawer/articraft_drawer.usd`.
Geometry and materials are embedded; the USD does not reference the OBJ files
or the intermediate importer output.

The USD was subsequently normalized in place to the original drawer's measured
198 mm enclosure width using a uniform factor of 0.495 on all three axes.
The enclosure is now 198 x 247.5 x 74.25 mm; slide travel is 168.3 mm and knob
diameter is 33.66 mm. Visuals, collisions, translations, joint anchors and travel,
and the planner's knob/bounding-box metadata use this same factor. Masses remain
unchanged; centres of mass scale by 0.495 and inertias by 0.495 squared. No shape,
aspect-ratio, friction, damping or grasp-policy tuning accompanies normalization.
The source URDF/OBJ files retain their original dimensions.

The following qualification describes the **earlier, unscaled** export; its
videos and HDF5 must not be replayed against the now-scaled USD.

The original dimensions were: enclosure 400 x 500 x 150 mm, 340 mm slide
travel, 68 mm knob diameter. The closed asset including the knob spans
x [-0.2, 0.2], y [-0.349, 0.25], z [0, 0.15] metres. The knob centre is
(0, -0.315, 0.065) in the drawer frame. The joint `drawer_slide` opens along -Y.

`ur7e_open_drawer_articraft` inherits the existing workcell, camera calibration,
robot configuration, and drawer pose distribution. Its planner box uses the
actual larger asset bounds. The existing 50% success threshold requires more
than 170 mm opening, so qualification commands a 200 mm pull. This is a
different physical scale from the earlier comparison, not a scale-matched ablation.

Import adaptation:

- Keep the importer-generated world fixed joint.
- Remove the importer's Newton articulation root and use a single PhysX/USD
  articulation root encompassing the links and joints.
- Disable internal articulation collisions, as in the existing drawer assets.
- Use convex decomposition for the concave enclosure and drawer collision meshes.
- Keep zero spring stiffness and 3.0 joint drive damping. The exported USD stores
  URDF joint friction 1.2, but the Arena runtime probe reports friction 0.0;
  the probe does not establish full URDF dynamics equivalence.

On this host, add the Isaac Lab source directory to PYTHONPATH as well as this
worktree, because AppLauncher can otherwise reload `isaaclab` as a namespace
without its initialization helpers. No shared installation changes are needed.

```bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH=/home/ubuntu/code/IsaacLab-Arena-tasks:/home/ubuntu/code/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab
.venv/bin/python isaaclab_arena_cumotion/scripts/ur7e_open_drawer_cumotion.py \
  --env ur7e_open_drawer_articraft --device cuda:0 --stop-after-pull \
  --pull-distance 0.20 --num-demos 3 --seed 0
```

The joint-only probe (`--probe-only`) held 50 mm and returned to 0 mm at all
three seed-0 randomized placements. Its success predicate is deliberately false
because the probe returns the joint to closed before evaluating it.

Full cuMotion qualification on 2026-09-11 (seed 0, three randomized placements,
original 0.03 rad robot jitter, 200 mm pull, stop after pull): one success out of
three attempts. Attempt 1 selected the 180-degree wrist spin and pulled the
drawer 200.4 mm. Attempts 2 and 3 had reachable IK targets but neither offered
wrist spin produced a plan to the pre-grasp standoff. This validates one complete
opening, not collection reliability across the shared placement distribution.

Artifacts alongside the USD:

- `cumotion_test.mp4`: 640x480 D435 view, 311 frames at 15 fps.
- `cumotion_test_scene.mp4`: third-person view.
- `qualification/drawer_articraft.hdf5`: one successful state/action recording.

Those planning results apply only to the earlier unscaled asset. For the current
width-normalized asset, the ordinary 100 mm pull exceeds the unchanged 50%
openness threshold (84.15 mm). Use `--pull-distance 0.10` for subsequent trials;
normalization alone does not establish a new collection success rate.

## Randomized collection (width-normalized asset)

The scaled seed-0 qualification succeeded with a 100 mm opening. A one-episode
pipeline smoke test also passed randomized D435 rendering, LeRobot validation,
EEF conversion and diffusion-policy Zarr validation (249 training frames).

Run or resume the 200-success pipeline with:

```bash
bash /home/ubuntu/playground/datasets/collect_ur7e_open_drawer_articraft_v2.sh
```

It uses two workers, the original pose distribution and 0.03 rad initial-joint
jitter, and stops recording immediately after pulling. Only successful episodes
ending above 84.15 mm opening with the gripper still closed are merged. The shared
visual randomizer supplies backgrounds, lights, table/floor materials and 0–6
distractors per episode (seed 0). LeRobot videos remain 640x480 at 15 fps; the
separate training Zarr uses the existing 320x240 setting.

Outputs are isolated under `rr_sim2real_raw/open_drawer_articraft_v2` and
`rr_sim2real/articraft_open_drawer` in `/home/ubuntu/playground/datasets`.
The raw directory's `complete.json` is written only after all stages validate.
The launcher supports the same target, batch-size and seed overrides as the
GPT56 pipeline; resume with the original settings. The single-episode smoke
outputs are separate and are not mixed into the 200-demo dataset.
