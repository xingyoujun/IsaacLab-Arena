# G2 native cuMotion release — 2026-09-26

This branch uses `tools/data_collection/run_g2.sh` for every G2 collection, raw validation,
three-camera replay and LeRobot export. Active implementation and task tests are in
`isaaclab_arena_cumotion/g2_collection/`, including `workcell/`, `dataset_io.py` and `metadata.py`.
Retired cuRobo collectors, old launchers, separate replay tools, unused trajectory retiming and
historical task guides were removed. Existing robot/environment definitions and keyboard support remain.

## Published assets

HF dataset: [xingyoujun/USDCraft-Scene](https://huggingface.co/datasets/xingyoujun/USDCraft-Scene/tree/082b4f1164ddd47ef5f0e775bdfeb14401691575).
Immutable revision: `082b4f1164ddd47ef5f0e775bdfeb14401691575`.
Manifest SHA-256: `7199f94aae34a15a1784a95c1e1b76b30ca34bec8157a8ddc228ecfeab0e1090`.

The release contains 89 manifest-listed files (132,291,408 bytes), plus `manifest.json`.
It adds the G2 robot, planning descriptions, collision spheres, room and task objects,
three task definitions and matching configuration/calibration snapshots. Existing Pine WM asset
entries and payloads are preserved. README and LFS metadata are shared repository metadata.
Room and aluminum-block USD duplicates were removed from Git. Raw HDF5, video, training data,
checkpoints and credentials are excluded from the asset release and Git commit.

USD dependency closure passed for 31 entrypoints; the Isaac runtime provides `OmniPBR.mdl`.
The first upload revealed HF-added JPG LFS rules changing `.gitattributes`. The final revision
reconciles those rules, and every file was downloaded and verified before pinning this revision.
The publisher now verifies remote bytes before writing `release.json`; staging registers JPG/JPEG
LFS rules before hashing. The earlier unverified upload is superseded by the pinned revision.

## Validation and timing

Evidence remains local under `outputs/g2_cleanup_20260926/`. No Git push was performed.
Earlier complete pilots remain under `outputs/g2_framework_20260926/` and are historical qualification,
not proof of a 200-episode native cuMotion batch success rate.

The cleanup regression exposed a right-arm wrist-branch failure during bowl descent. Additional
shoulder IK seeds were added while retaining the 2.6-radian travel cap and collision-checked planning.
The two earlier attempts are retained as failures, not exported as successful data.
One subsequent success does not establish the reliability of the expanded seed search.

The cleanup passed 45 targeted tests and host `pre-commit run --all-files`.
The full Arena three-phase test suite was not run. The fresh stack-bowls pipeline `stack_003`
passed raw task validation, all three camera streams and LeRobot export (1,044 frames per camera).
LeRobot 0.3.3 loaded the first/middle/last samples successfully; all three camera midpoints were
visually inspected. All three registered tasks passed preflight against the downloaded HF revision,
and the three earlier qualified raw recordings also passed the relocated audit implementation.
The same normal container retained its shader cache:

| Stage | Measured wall time |
| --- | ---: |
| Raw collection, including startup | 165.91 s |
| Camera process initialization | 20.2 s |
| Actual three-camera render loop | 74.27 s |
| Entire run through export artifact | 296.21 s |

Stage times except the renderer's own counters are estimated from run/log/artifact timestamps.
For 200 successful fixed-layout episodes with similar timings, serial raw collection extrapolates to
9 h 13 m and the entire pipeline to 16 h 27 m. This excludes failed attempts, recovery, hardware
contention and changed scenes. It is a single-success extrapolation, not a measured batch average.
Planning is a small part of the overall simulation/rendering cost.

The fresh `workcell_001` raw-collection regression completed all three objects and passed its
independent raw audit in 429.97 s including startup. This cleanup run did not repeat workcell rendering;
the shared renderer/exporter was exercised end-to-end by `stack_003`. The earlier full workcell
qualification remains documented in the development guide. The fresh result and exact device-audit
limitations are indexed in `outputs/g2_cleanup_20260926/qualification.json`.

## Historical 200-episode comparison

The previous stack-bowls dataset used cuRobo MotionGen, with 2 cm XY randomization.
`/home/ubuntu/datasets/agibot_dataset_v1_raw/stack_bowls/collection_manifest.json` records
208 attempts, 200 successes and 8 failures (96.15% observed success).
Logs and final artifact timestamps give these wall-clock spans (UTC):

| Stage | Start | Finish | Duration |
| --- | --- | --- | --- |
| Raw collection | 2026-09-10 19:07:54 | 2026-09-11 05:16:50 | 10 h 08 m 56 s |
| Rendering and export | 2026-09-11 05:16:51 | 2026-09-11 11:21:46 | 6 h 04 m 55 s |
| Entire batch | 2026-09-10 19:07:54 | 2026-09-11 11:21:46 | 16 h 13 m 52 s |

Average raw attempt wall time was 172.65 s, median 173.58 s. These are log/artifact-derived
wall-clock measurements, not profiler totals. See `historical_stack_timing.json` in the evidence directory.
Historical image alignment and capture implementation differ from the new protocol.

A roughly five-minute cold camera startup is first-use RTX shader compilation in a fresh isolated
container. Reusing the normal container retains caches. Each current `run` still starts independent
collection and renderer processes; process/environment initialization is paid for every episode.
Creating a fresh isolated container per episode repeatedly incurs the cold cost.

The current native task registry uses fixed qualification layouts. `--seed` alone does not produce
200 varied scenes. Large-scale collection needs scene randomization and a measured retry/success-rate
policy; a single-episode throughput extrapolation is not a guaranteed batch duration.
