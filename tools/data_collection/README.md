# Unified collection

[G2 development guide](../../docs/g2_development.md) is the supported workflow.
Use `run_g2.sh` from the host, or `g2.py` inside this checkout's container.

```bash
tools/data_collection/run_g2.sh list
tools/data_collection/run_g2.sh run --task clean_workcell_table --run-dir outputs/g2/workcell_000
```

The same entrypoint owns `preflight`, `collect`, `validate`, `render`, `export`, and full `run`.
All G2 tasks use the public native cuMotion planner, CUDA simulation, the shared pre-step transition
protocol and three native cameras. Old G2 collection CLIs are retired; they do not launch collection.
Code and assets can be synchronized between developers; raw data stays on the collection host.
These commands never upload assets, push Git or overwrite a recording.

`export_g2_episode.py` is the implementation of the unified export stage, with task-specific raw
checks and raw/video SHA-256 pairing. `derive_episode.py` remains an explicit historical migration
utility; it never changes the source recording or removes task/final transitions.

Pine WM keeps its task registry and qualification gates, sharing the public planner/executor,
HF asset resolver and `arena.transitions.v1` protocol. Its export config is
`isaaclab_arena_gr00t/lerobot/config/pine_wm_config.yaml`. Preview videos are not training sidecars.
RR and historical Agibot remain separate task families.
