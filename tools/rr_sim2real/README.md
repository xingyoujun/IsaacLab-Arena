# Small migration references

Read [the agent handoff](../../docs/rr_sim2real/README.md) first.

- `asset_overlays/`: tiny USD configuration layers only. Copy beside the
  separately obtained object payloads, retaining relative references.
- `diffusion_policy_overlay/`: source-host DP extensions/config snapshots, with
  formatting/license headers normalized for this repo. They are not installed
  into Arena. Compare with your DP checkout before copying relative paths.
  The upstream DP project and derived workspace configuration are subject to the
  retained `diffusion_policy_overlay/LICENSE` notice.
- `reference/*.txt`: historical source-host scripts and calibrated scene source,
  intentionally stored as text. They have absolute paths and some scripts delete
  or replace outputs. Do not execute unchanged. They are not portable launchers.

Snapshots were taken 2026-09-11. No source-host external files or shared Python
environments were changed while creating them. Subsequent DP changes must be
explicitly synchronized; this directory is not an automatic mirror.
