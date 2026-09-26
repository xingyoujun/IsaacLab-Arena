# Pine WM task dashboard

Read-only dashboard for the 20 priority PhysX tasks. Start from this checkout:

```bash
python3 tools/pine_wm/dashboard/server.py --host 0.0.0.0 --port 8090
```

The page reads the external first20 experiment directory on every refresh. It
separates development success, fixed-version qualification and available videos.
Current videos require a passing live-preview manifest and a matching layout snapshot. No HDF5 datasets,
workspace directory listings or arbitrary filesystem paths are served. MP4 byte
ranges support browser seeking. Bind to 127.0.0.1 when using an SSH tunnel only.

Current collection is native Isaac Sim cuMotion GraphBasedMotionPlanner plus
TrajectoryGenerator and solve_ik, followed by PhysX execution and HDF5 recording.
It is not the Python cuRobo MotionGen pipeline. The observed simulation device is
cuda:0 on NVIDIA L40S; cameras use GPU RTX. Existing NumPy/SciPy geometry and
contact auditing run on CPU. Native graph planning exposes CUDA tree acceleration
with CPU/GPU crossover; use of the cuMotion package alone is not evidence that
all IK, planning or trajectory generation executed on CUDA. No CUDA profiler
trace covering these solvers has been obtained. **All-solver CUDA acceptance is
not met.** GPU rendering does not mean that the renderer exclusively uses CUDA
rather than RTX/Vulkan and its host orchestration.

Changing to GPU collision-aware IK / trajectory optimization requires preserving
camera and printed mount collision geometry, attached payload checks, continuity
and Cartesian contact motions, then rerunning qualification. Do not relabel the
existing mixed execution results as CUDA-only data. CPU scheduling and file I/O
are separate from this solver requirement.

Authoritative references:

- https://github.com/nvidia-isaac/cumotion (Lula and cuRobo lineage)
- https://nvidia-isaac.github.io/cumotion/api/cpp_api.html (graph planner CUDA tree and CPU crossover)

## Current preview workflow

`preview_queue.py` executes tasks serially and stops each task after one success.
Current-layout videos come from live sensor capture during execution, not replay.
All 20 tasks have a successful four-camera preview. The launch gate is closed
pending user review; stability and collection remain disabled.

`pine_wm_web` serves the page and `pine_wm_monitor` updates its heartbeat.
Old PID-based round09 scheduling helpers have been removed. Historical results
remain outside the repository; they are not counted as current-layout previews.

Assets are distributed separately through the private HF USDCraft-Scene dataset;
see [asset management](../../usdcraft_scene/README.md).
