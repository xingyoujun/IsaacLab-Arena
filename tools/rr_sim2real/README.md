# Small migration references

Read [the agent handoff](../../docs/rr_sim2real/README.md) first.

This directory belongs to **RR real2sim**, not Pine WM. See
[task ownership](../../docs/task_families.md). The tracked USDA layers below are
RR-specific compatibility configurations. Keep them here with their RR callers;
they are not Pine WM scene assets and are not inputs to the Pine WM HF packager.
`boxx_support.usda` is read directly by the RR toaster-knob task, so deleting it
would break that task. Other layers are copied beside external RR payloads as
described in the handoff.

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
