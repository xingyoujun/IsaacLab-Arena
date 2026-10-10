# Pine WM cable wrap around a post (USDCraft segmented cable)

`pine_wm_cable_wrap_post` adds a USDCraft cable and a static post (radius 15 mm, height 50 mm) to
the `pine_wm` workcell. The cable's root plug is pinned, standing in for a connector in a fixed
jack. `data_engine/pine_wm/collection/wrap_cable.py` collects the demonstration with cuMotion:
- grasp the free plug top-down and lift it 10 mm;
- pull it in towards the post for slack;
- drag it counter-clockwise around the post at a fixed 55 mm radius until the taut cable wraps it;
- set the plug down, release, and return to the ready configuration.

## Asset

The cable is the 2026-10-08 USDCraft Generation record
`rec_one-short-3-5-mm-audio-patch-cable-approximately_20261008_045800_06ce1e91_cca72cc4`. It is
0.25 m tip to tip: two 3.5 mm plugs (7 mm grip) and twenty 2.8 mm jacket links joined by
alternating-axis revolute springs. It is an ordinary PhysX articulation, so it loads through
Arena's `Object` with `ObjectType.ARTICULATION`; Arena's Newton-only `Cable` asset is not used.
- **Location.** The export is copied to the gitignored `local_assets/usdcraft_cables/audio_cable/`
  and selected with `ARENA_PINE_WM_CABLE_USD`.
- **Stability.** USDCraft's `validate_cable` drop, hang and drape protocols completed at dt 1/120 s
  for this file and for the Ethernet and USB-C records. The evidence is in `validation/` next to
  the asset.
- **Why the audio cable.** The Ethernet and USB-C cables reach their bend limits in the drape
  protocol: their minimum bend radius is about 35–40 mm. The audio cable bends tighter, so it is
  the one that can wrap a 15 mm post.

The joint frames give this cable a 17 mm rest bow along its local +x. The environment rotates the
cable so that the bow lies in the table plane, away from the post. With the bow pointing down,
the cable spawned into the table and buckled about 90° when PhysX pushed it out.

## Run

```bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export ARENA_PINE_WM_CABLE_USD=$PWD/local_assets/usdcraft_cables/audio_cable/isaac/model.usdc
.venv/bin/python -m data_engine.pine_wm.collection.wrap_cable --output outputs/cable_wrap/NEW_DIR \
    --enable_cameras --kit_args="--/rtx/verifyDriverVersion/enabled=false"
```

The outputs match the towel fold: `results.json`, `annotations.json`, the HDF5 file (cable
articulation states per step, plus a `cable` group with rest and final link positions and the post
geometry) and four preview videos.

## Success criteria

The criteria are checked on the settled cable after the arm has returned:
- **Winding.** The signed angle swept around the post axis along the cable, from the pinned plug
  to the free plug, must be at least 240°. The untouched straight cable already scores 130°, and
  a straight line past a post never reaches 180°.
- **Contact.** The closest link must be within 6 mm of the post surface.
- **Plug on the table.** The free plug must lie within 10 mm of the table.
- **No other robot contact.** No robot body other than the gripper may touch anything.

## Measured (2026-10-08, RTX 3090, PhysX dt 1/120 s, single runs)

| Run | Variation | Winding | Post gap | Result |
| --- | --- | --- | --- | --- |
| `demo_01` (cameras), `run_03` | defaults: radius 55 mm, end 160° | 288° | 2.7 mm | success |
| `r_end130` | end angle 130° | 266° | 10.4 mm | failed: not tight on the post |
| `r_rad065` | radius 65 mm | 179° | 29 mm | failed |
| `r_end200` | end angle 200° | — | — | failed: no safe transit back to ready |

The first attempt dragged the plug along a spiral that started 135 mm from the post. That spiral
asked for more than the 0.19 m of cable between the pinned plug and the grip, and the taut cable
pulled the plug out of the jaws. The current path pulls the plug in first to create slack.

These are development runs at one fixed layout, not a qualified success rate. The 0.25 m cable
leaves only about 2 cm of slack for one wrap. Multiple posts or full turns need a longer cable;
generating one with USDCraft needs the user's approval.
