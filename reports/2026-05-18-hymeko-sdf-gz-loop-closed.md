# HyMeKo → SDF → GZ loop closed for the rapport demo

**Date:** 2026-05-18 (evening, after the Stage G' demo ship)
**Triggered by:** user question *"The workflow is HyMeKo description → SDF/URDF → GZ, right?"*
**Verdict:** ✅ Kinematic structure of the rapport-demo agents now declared in `.hymeko` files and emitted to SDF via `hymeko emit -f sdf`; the demo runs on HyMeKo-emitted models.

## 1. Summary

The Stage G' rapport demo, as initially shipped, used hand-written
SDFormat files for the agent models. The user's question revealed
the obvious gap: HyMeKo's `hymeko_formats` crate already ships SDF /
URDF / MJCF / Gazebo emitters, but the demo bypassed them. This
report records closing that loop: the triad agents are now declared
in HyMeKo robotics files and SDF-emitted by the framework's own
toolchain.

**Honesty caveat surfaced**: the SDF emitter has two gaps that the
demo wrapper had to fill manually:

| Gap | Workaround | Proper fix |
|:---|:---|:---|
| Joint-origin xyz/rpy not propagated to per-link `<pose>` | wrapper injects link poses from .hymeko joint declarations | `hymeko_formats::sdf` propagation, ~30 LOC Rust |
| `color` directives stripped from emitted material tags | wrapper grafts `<material><ambient>...<diffuse>...</material>` | `hymeko_formats::sdf` codegen, ~50 LOC Rust |

Both are clean candidates for upstream fixes in the `hymeko_formats`
crate; the demo wrapper is a transitional bridge.

## 2. The workflow now

```
data/robotics/triad_human.hymeko       ─emit─→  kinematic SDF
data/robotics/triad_r1.hymeko          ─emit─→  kinematic SDF
                                         │
                                         │  scripts/emit_triad_sdf.py
                                         │  - inject link poses from joint origins
                                         │  - inject <material> from *_color
                                         │  - graft PosePublisher / DiffDrive / camera plugins
                                         ▼
                                  data/models/triad_human/model.sdf  (253 lines)
                                  data/models/triad_r1/model.sdf     (633 lines)
                                         │
                                         ▼
                                gz sim ─→ rapport pipeline ─→ r1 sees alice + bob
```

**Single regenerate command**:

```bash
python scripts/emit_triad_sdf.py
```

## 3. The HyMeKo robotics files

`data/robotics/triad_human.hymeko` (62 lines): cylinder body + sphere
head + fixed neck joint, using
`meta_kinematics.kinematics.elements.link` for both links and a
single `fixed_joint` for the neck. The `body_color` and `skin_color`
constants are declared at the top of the model block.

`data/robotics/triad_r1.hymeko` (95 lines): box chassis + sphere head
+ 2 cylinder wheels + fixed neck + 2 continuous wheel joints. The
`chassis_color`, `head_color`, `wheel_color` are declared at the top.
The `axis [-1.5707963, 0.0, 0.0]` joint rpy gives the wheels their
standard rolling axis (the wrapper translates this to per-link
`<pose>` since the SDF emitter doesn't yet do it).

## 4. The wrapper script

`scripts/emit_triad_sdf.py` (~225 LOC):

| Pass | What it does |
|:---|:---|
| `emit_hymeko_sdf(hymeko_path, model_name)` | Runs `cargo run --bin hymeko -- emit -f sdf <file>.hymeko` and captures stdout |
| `inject_link_pose(model, link_name, pose_xyz_rpy)` | Inserts `<pose>` at the start of `<link>` (derived from the .hymeko joint origin) |
| `inject_material(model, link_name, rgba)` | Adds `<material><ambient>...<diffuse>...</material>` to each `<visual>` (mirrors the `*_color` constant from the .hymeko file) |
| `graft_xml_into_link(model, link, snippet)` | Adds visual / sensor / arrow children to a named link |
| `graft_xml_into_model(model, snippet)` | Adds plugins (PosePublisher, DiffDrive), the camera link / joint, etc. |

The wrapper is intentionally additive: structural decisions
(geometry, joints, link names, masses, inertias) come from the
.hymeko file; runtime extensions (gz-specific plugins, camera
sensors) are layered on top. If the .hymeko file adds a new link,
the SDF emit picks it up; only the plugin attachments need to know
the link names.

## 5. Why two gaps in the emitter, not one?

Both bugs are in the same "the SDF emitter currently treats geometry
as the whole story." HyMeKo's IR has joint origins and color
constants; the SDF emitter's transform templates simply don't read
them.

Concrete locations to fix:

- `hymeko_formats/transforms/sdf/` — the template-based transform
  registry. The template that emits `<link>` would need to read the
  joint subgraph and write `<pose>` based on the parent joint's
  origin.
- `hymeko_formats/src/sdf.rs` — the codegen for `<material>` is
  either absent or doesn't read the `color -> <name>.body_color`
  reference path. A small Rust patch suffices.

Both are tractable but out of scope for the visit prep. Filed as
issues in the wrapper script's docstring.

## 6. Verification

| Property | Before | After |
|:---|:---|:---|
| Source of truth for human body shape | hand-written XML | `triad_human.hymeko` cylinder declaration |
| Source of truth for r1 wheel positions | hand-written XML | `triad_r1.hymeko` joint origin (`[0.0, 0.20, -0.05]` for left wheel) |
| Source of truth for body colour | hand-written `<material>` | `triad_human.hymeko` `body_color [0.42, 0.62, 0.83, 1.0]` |
| `gz sim` accepts the emitted SDFs | manual files only | yes, both emitted models load (verified by `gz sdf -p`) |
| Pose topics fire | yes | yes |
| Cmd_vel + diff-drive responds | yes | yes |
| HSV blob detector finds blue capsules | yes | yes (material colour matches the .hymeko constant) |

## 7. Files

### New
- `data/robotics/triad_human.hymeko` (62 lines)
- `data/robotics/triad_r1.hymeko` (95 lines)
- `scripts/emit_triad_sdf.py` (~225 lines)

### Regenerated (from HyMeKo, replacing the prior hand-written versions)
- `data/models/triad_human/model.sdf` (253 lines)
- `data/models/triad_r1/model.sdf` (633 lines)

### Backup of hand-written versions
- `/tmp/triad_sdf_backup/triad_human_handwritten.sdf`
- `/tmp/triad_sdf_backup/triad_r1_handwritten.sdf`

### CORE.YAML items touched
None. `hymeko_formats` is the non-CORE plugin crate (per
`project_formats_plugin_extraction_2026_04_20.md`); only its
output is consumed.

## 8. Open items

1. **Upstream the link-pose-from-joint-origin propagation** in
   `hymeko_formats::sdf`. ~30 LOC Rust. Removes the wrapper's
   `inject_link_pose` calls.
2. **Upstream the `color -> <material>` propagation** in
   `hymeko_formats::sdf`. ~50 LOC Rust. Removes the wrapper's
   `inject_material` calls.
3. **Migrate the world file too.** `data/worlds/triad_hri.sdf` is
   still hand-written. A HyMeKo `world` block could declare the
   floor, lighting, and model `<include>` statements with poses;
   wrapper would render the world file from it. ~half day.
4. **drchubo migration** for full anthropomorphic agents. Memory
   already has drchubo URDF import via `urdf_to_hymeko.py` (per
   `project_real_urdf_imports_2026_04_21.md`); the reverse path
   (HyMeKo drchubo → SDF for gz) is the natural next step
   once the visit demo is locked in.

## 9. Bottom line

The HyMeKo → SDF → GZ pipeline works end-to-end for the rapport
demo. The framework's own toolchain emits the agents. Two
documented emitter gaps are bridged by a 225-LOC wrapper script;
both gaps are clean candidates for upstream Rust patches but were
out of scope for visit prep. The "kinematic structures" line in
Niitsuma's brief (§4 of `niitsuma-brief-2026-05-14.tex`) is now
concrete in this demo — alice, bob, and r1 are declared as
HyMeKo signed-hypergraph structures and instantiated in the
physics simulator from those declarations.
