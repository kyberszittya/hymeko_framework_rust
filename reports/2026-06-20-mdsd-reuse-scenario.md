# MDSD single-source reuse for the ROS2 robot scenario

*2026-06-20 · Aiko (Claude Code) for Dr. Csaba Hajdu*
*Plan: [docs/plans/2026-06-20-mdsd-reuse-scenario/](../docs/plans/2026-06-20-mdsd-reuse-scenario/)*

## Summary

The ROS2 demo robot was re-authored to the MDSD single-source-of-truth pattern:
instead of re-declaring its kinematic and context-state vocabulary inline as bare
types, the new scenario **imports** the canonical shared profiles and declares only
its instance hypergraph. The paper-faithful baseline is kept untouched as the
comparison control (user direction: it is a selling point, used to measure the
length reduction).

**Key correction to the premise.** "Wiring the import path" required *no framework
work*. Cross-file import resolution already flows through the full emission pipeline
— `data/robotics_imported/wam/wam.hymeko` imports `meta_kinematics.hymeko` and
`hymeko_query/tests/test_imported_real.rs` proves extraction + 6-format emit on it.
The `xprofile-instance-refs` core enabler landed 2026-06-19. The ROS2 scenario
simply predated the pattern. The work here is a scenario + a new shared vocabulary +
a regression test, plus two requested docs.

## Files touched

| File | Δ | Note |
|------|---|------|
| `data/robotics/meta_context.hymeko` | +57 (new) | Shared context-state vocabulary (context/signal/counter/mode/reference/tool/payload/component/grasp_mode + @interpretation/@aggregation/@constraint). No prior scaffold (§6.1 discovery: `meta_hri` is a different domain). |
| `hymeko_ros2_demo/.../scenarios/hymeko_robot_reuse.hymeko` | +126 (new) | Reused scenario; imports `meta_kinematics` + `meta_context`. |
| `hymeko_query/tests/test_mdsd_reuse.rs` | +118 (new) | Regression: compile + extract + emit + LOC guard. |
| `hymeko_query/tests/mod.rs` | +1 | Register test module. |
| `docs/architecture/SYSTEM_ENGINEERING_VIEW.md` | +152 (new) | MDSD manifesto. |
| `docs/HYMEKO_ECO_CAP.md` | +98 (new) | Capability-evidence ledger. |
| `docs/plans/2026-06-20-mdsd-reuse-scenario/` | new | plan.{tex,pdf,tikz,mmd}. |

## CORE.YAML items touched

**None.** `hymeko_query` is `lockdown: implementation`; only a new test file was added
(allowlisted `tests/**`), no library edit. Vocabulary/scenario files are under `data/`
and the demo package (non-core). No dependency change.

## Test results

- `cargo test -p hymeko_query --test integration test_mdsd_reuse` — **2 passed**, 0
  failed, 0.03 s:
  - `reused_scenario_compiles_extracts_and_emits`: 5 links, 4 revolute joints, all
    joints carry unit axis vectors; URDF contains `<robot name="hymeko_robot"` and
    `<joint`; SDF non-empty.
  - `reused_scenario_is_shorter_than_baseline`: 126 < 189 lines.
- Full `hymeko_query` integration target — **214 passed**, 1 ignored (pre-existing),
  0 failed, 3.37 s. No regression.
- `cargo clippy -p hymeko_query --tests -- -D warnings` — clean (exit 0).
- `hymeko validate hymeko_robot_reuse.hymeko` — ✅ valid.

## Measurements

| Metric | Baseline (bare) | Reuse (imported) | Δ |
|--------|-----------------|------------------|---|
| Source lines | 189 | 126 | −33 % |
| Code lines (no comment/blank) | 108 | 76 | −30 % |
| URDF `<joint>` elements emitted | 0 (generic `joint` type unknown to extractor) | 4 (typed `rev_joint`) | — |
| Axis vectors recovered | none | 4 unit vectors | — |

The length saving is per-robot; the ~55-line shared vocabulary is amortised, so the
marginal saving for the *N*-th robot is the full re-declaration block. The semantic
saving (typed joints → valid URDF) is the stronger argument: the bare baseline does
not project to a working robot at all.

Performance: sub-second IR work, peak RSS ≪ 1 GB (16 GB cap not approached). Not a
perf-sensitive change; no benchmark surface.

## §6.5 anti-patterns

None introduced. The change *removes* a duplication anti-pattern (inline vocabulary
re-declaration) rather than adding one. The new `meta_context.hymeko` is a genuine
new artifact confirmed absent by §6.1 discovery. Baseline kept as an intentional,
documented frozen control — not v-suffix proliferation.

## Open / follow-up

- The live demo (`grasping_context_node.py`, launch) still consumes the baseline by
  design (control preserved). Switching the demo to the reused source is a separate,
  optional step once a context-evaluation regression is in place for the imported
  form.
- Transitive-import indexing and a cleaner `shared.dist`-style path (dropping the
  `_description` segment) remain future CORE edits (see
  `reports/2026-06-19-shared-agent-models.md`), unblocked-but-deferred.

## Provenance

- Git: branch `soma-vision`; working tree dirty (pre-existing unrelated changes, see
  `git status`). New files listed above are the contribution of this task.
- Toolchain: `cargo` (rustup), MiKTeX `pdflatex` for the plan PDF. No new deps.
- Determinism: all inputs are committed fixtures; no RNG, no GPU.
