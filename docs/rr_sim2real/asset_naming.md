# Four methods, two cases

Authoritative inventory: [assets.json](assets.json). Assets are named
`<method>_<drawer|toast>`; task/dataset names are
`<method>_<open_drawer|press_toaster>`. No numeric version suffix in canonical
exports. GPTSOL means the supplied GPT-5.6 Sol high output; Astra means the
supplied GPT-6 Astra low output. These are two miniworkflow baselines, not USDcraft.

All paths below are relative to `/home/ubuntu/playground/rr_ur`.

| Method | Drawer payload | Toast payload |
| --- | --- | --- |
| USDcraft (ours) | `usdcraft_drawer.usdc` | `usdcraft_toast.usdc` |
| Articraft | `articraft_drawer/articraft_drawer.usd` | `articraft_toast/articraft_toast.usd` |
| miniworkflow GPTSOL | `miniworkflow_gptsol_drawer.usd` | `miniworkflow_gptsol_toast.usd` |
| miniworkflow Astra | `miniworkflow_astra_drawer.usd` | `miniworkflow_astra_toast.usd` |

Articraft toast has now been exported from its unchanged URDF/mesh source and
uniformly scaled by 0.795522222222 to 0.143194 m width. Do not rescale the USD.
See [toast qualification](../ur7e_press_toaster_baselines.md) for measured outcomes.
The Articraft drawer's source URDF is `articraft_drawer/articraft_drawer.urdf`;
the standalone USD is already normalized by 0.495. Do not apply that factor twice.
The root-level old `drawer_articraft.urdf` is a legacy incomplete copy, not the
authoritative source; do not use it for future conversion.

Only Articraft receives uniform dimension normalization. Do not scale, reshape
or tune the other methods. Existing `_arena.usda` overlays were retained for
USDcraft drawer/toast and miniworkflow GPTSOL drawer and their references updated.
The Astra drawer now uses a minimal articulation-root overlay after its raw USD
failed fixed-base spawning; see [qualification](../ur7e_open_drawer_astra.md).
Diagnose any structure/planning failure before deciding that an Arena wrapper is needed. Missing toast decal
textures are a known source issue covered by the existing USDcraft overlay.

## Existing task and dataset mapping

| Preferred environment | Legacy environment (still works) | Dataset directory |
| --- | --- | --- |
| `ur7e_usdcraft_open_drawer` | `ur7e_open_drawer` | `usdcraft_open_drawer` |
| `ur7e_usdcraft_press_toaster` | `ur7e_press_toaster` | `usdcraft_press_toaster` |
| `ur7e_articraft_open_drawer` | `ur7e_open_drawer_articraft` | `articraft_open_drawer` |
| `ur7e_miniworkflow_gptsol_open_drawer` | `ur7e_open_drawer_gpt56` | `miniworkflow_gptsol_open_drawer` |

Each existing dataset has a sibling `<dataset>_dp.zarr`, 200 episodes, and
randomized appearance. They still show the old gripper: black-gripper replay
remains pending. New robot spawns default to black.

The Astra drawer environment `ur7e_miniworkflow_astra_open_drawer` is now registered;
qualification recordings are separate from the four existing 200-demo datasets.

The three toast baseline environments are now registered:
`ur7e_articraft_press_toaster`, `ur7e_miniworkflow_gptsol_press_toaster`,
`ur7e_miniworkflow_astra_press_toaster`.
There are no training datasets for these cases. Joint/paddle adapters are in
`ur7e_press_toaster_baselines_environment.py`; inspect the qualification report
before collecting. Both miniworkflow toasts need minimal root overlays after
their raw USDs failed fixed-base spawning. Astra also needs collision-only hull
triangulation after four PhysX cooking failures; source and visible geometry
remain unmodified.

Existing HDF5 articulation keys (`drawer_rr`, `drawer_rr_gpt56`, `drawer_articraft`,
`toaster_rr`, `pedestal`), USD internal prims and joints are deliberately unchanged.
Renaming these would invalidate recorded states. Historical raw directories,
checkpoint configs and logs also retain original names. For checkpoint resume,
override the dataset path; no old data-path symlinks remain now that training ended.

Current collectors and DP defaults use the new canonical paths; historical
collector filenames remain compatibility entry points. Toaster black replay reads
the original raw `press_toaster/press_toaster_v2.hdf5` and writes isolated
`usdcraft_press_toaster_black_gripper` raw output, never overwriting trained data.

For fresh USDcraft drawer collection, use
`isaaclab_arena_cumotion/scripts/collect_ur7e_usdcraft_open_drawer.py` (not the
historical external v1 shell script). It reuses the randomized integrated
pipeline with the original drawer's 75 mm threshold. Supply isolated `--raw`
and `--final` directories when the canonical dataset already exists. No
collection or render was launched as part of the naming migration.

Rename provenance is local: `rr_ur/asset_rename_manifest_2026-09-11.json` records
payload hashes before/after the filename move; the dataset movement manifest is
under `datasets/rr_sim2real_aux/2026-09-11_cleanup/`. All payload bytes were
preserved; only existing overlay references were then edited. Method aliases use
distinct config dataclasses (required by the registry) while keeping legacy IDs.
The Astra run exercises complete environment registration; the other aliases
have not been separately simulation-qualified by this rename.
