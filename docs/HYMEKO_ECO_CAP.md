# HyMeKo — Capability Evidence Ledger

*Companion to the [System-Engineering View manifesto](architecture/SYSTEM_ENGINEERING_VIEW.md).*
*Last updated: 2026-06-20.*

## What this document is, and the standard it holds itself to

This is a calibrated inventory, not a brochure. Each row states three things kept
deliberately separate: **the specific claim**, **the artifact that checks it**, and —
where it matters — **what the artifact does not establish**. The aim is that a reader
can audit any line by opening the named file, not take it on trust.

Two honesty rules govern the table:

- A claim is only as strong as what is *mechanically* checked. "A test passes" means
  the assertions inside that test hold; it does not silently extend to properties the
  test never examines. Where a test proves structure but not, say, visual fidelity or
  numerical parity, that boundary is stated in the row or the section note.
- Provenance is marked. Results **re-verified in this session (2026-06-20)** are
  tagged *(measured)*. Results resting on a prior dated report are attributed to it
  rather than presented as freshly confirmed. A green status inherited from a report
  is a claim about that report, not an independent re-run.

**Status legend.**
`PROVEN` — a passing automated test or committed generated artifact asserts the
specific claim.
`SHIPPED` — exercised in a working demo or report, but not yet pinned by a dedicated
regression test, so it can silently regress.
`RESEARCH` — an open investigation; the result may be positive, partial, null, or
falsified, and is reported as whichever it is.

A status is never upgraded without an evidence anchor, and a row whose anchor has
gone stale is a defect, not a convenience.

---

## 1. Hypergraph core — the single source of record

| Capability | What the evidence actually establishes | Status |
|-----------|-----------------------------------------|--------|
| Signed-incidence hypergraph; GGK 4-tuple `K = (B, G, μ, r)` with signed-incidence tensor | The core types and the `K1`–`K4` axioms they satisfy are specified in `docs/spec/g_sphf_axioms.tex` (cited in published work) and implemented in `hymeko_core` (locked, `lockdown: full`). The axioms are documentation + design contract, not a machine-checked proof. | PROVEN (impl), spec-backed |
| Canonical hashing of descriptions | `hymeko_core/src/ir/canonical_hash.rs` produces a hash that is invariant under semantics-preserving rewrites and changes under semantic ones; hash-parity is asserted across the core and query suites, so an accidental semantic change surfaces as a failing test. | PROVEN |
| Lower → resolve → intern compile pipeline (`ModuleStore`) | The production compile path is exercised by the `hymeko_core` suite (≈133 tests per the 2026-06-19 report). Not independently re-run this session. | PROVEN (per core suite) |

## 2. The HyMeKo language

| Capability | What the evidence actually establishes | Status |
|-----------|-----------------------------------------|--------|
| LALR(1) grammar + parser (`parser` crate, generated tables, locked) | Parser round-trips the fixture corpus including alias resolution (`parser/tests/using_alias.rs`). Tables are generated, never hand-edited. | PROVEN |
| Profiles, `<isa>` inheritance, tag annotations, tensor/edge values | The `data/minimal_examples/**` corpus + core import tests parse and lower each construct; this proves the language *accepts and lowers* them, not that every semantic edge case is covered. | PROVEN |
| **Cross-file imports** (`@"file.hymeko"`) sharing meta vocabulary | `hymeko_core/.../test_import/` + `import_examples/` compile a file against vocabulary declared in another. Establishes kind-sharing across files. | PROVEN |
| **Cross-profile instance references** (`using <desc>.<content> as a; (+ a.decl)`) | Core test `check_xprofile_instance_ref` compiles an importer that references another profile's *instance* decls; the referenced `dist` decl's hash is unchanged vs. the inline form, which is the concrete evidence that semantics were preserved, not merely that it compiled (report 2026-06-19, `APPROVED-CORE-EDIT: xprofile-instance-refs`). | PROVEN |
| Import-aware bundle reading on the Python side (`read_bundle`) | Mirrors the compiler's import resolution so the Python tooling sees the same merged decl set; per `reports/2026-06-19-shared-agent-models.md`. Phase-1 limitation, stated there: the importer must also import the meta vocab the shared profile uses (no transitive indexing yet). | PROVEN, with stated limit |

## 3. Robotics — kinematic extraction + multi-format emission

| Capability | What the evidence actually establishes | Status |
|-----------|-----------------------------------------|--------|
| Kinematic-model extraction | `extract_kinematic_model` recovers links, typed joints, axis vectors, limits, geometry, origins. `test_imported_real` asserts counts within a documented tolerance (zero-DOF/anchor joints may be filtered): ≥8 links / ≥6 joints for WAM, ≥50 / ≥50 for DRC-Hubo. | PROVEN |
| Emitters: URDF, SDF, MJCF, DOT, Mermaid, SysML (one IR → six views) | The six emitters produce non-empty output carrying the correct robot header on real fixtures (`test_imported_real`, `test_sysml_emit`, `test_mermaid`). `test_mdsd_reuse` additionally asserts the URDF contains real `<joint>` elements with the recovered unit axis vectors *(measured)*. | PROVEN (structure) |
| Lean4 proof-obligation emission | `hymeko_emitter/src/emit_lean4.rs` + tests emit non-empty Lean4 from the IR. Proves emission, not that the obligations discharge. | PROVEN (emit) |
| HyMeKo → HyMeKo canonical round-trip | `emit_hymeko.rs` + `test_bridge.rs` re-emit a description and check it back; the information-loss guard for any view generator. | PROVEN |
| Emission on real published robots (Barrett WAM 7-DOF, DRC-Hubo 52-link humanoid) via import | The translated URDFs compile + emit through the full pipeline; this is the "not only synthetic morphologies" evidence. | PROVEN |
| **MDSD single-source reuse** for a full scenario | `test_mdsd_reuse` *(measured 2026-06-20)*: the imported form yields 5 links + 4 revolute joints with unit axes and emits `<joint>`s; the bare baseline's generic `joint` type emits none; reuse is 126 vs 189 lines (−33 %). | PROVEN |
| Gazebo launch-bundle generation | `test_gazebo_sim_launch` builds URDF + world + launch; a committed bundle exists at `generated/gazebo_launch/moveo/`. | PROVEN |
| Robot generators (anthropomorphic arms; chain/tree/quadruped/humanoid scaling fixtures) | Generation tests build valid descriptions across morphologies and sizes (`scripts/scaling/`). | PROVEN |

**Scope — what §3 does not claim.** Emission is *structural and kinematic*, not
*visual*. The importer strips mesh references and substitutes placeholder box
geometry (documented in `test_imported_real.rs` itself), so a round-tripped robot
will **not** visually reproduce the original in Gazebo/MuJoCo; mesh-bearing emission
is a separate, unbuilt line. Lean4 output is emitted, not proven to discharge.

## 4. Query, rewrite, analysis

| Capability | What the evidence actually establishes | Status |
|-----------|-----------------------------------------|--------|
| Predicate query engine over the IR (`node`/`edge`/`inherits`/`and`) | Drives all of §3's extraction; the full `hymeko_query` integration target runs **214 tests, 214 pass, 1 ignored** *(measured 2026-06-20)*. | PROVEN |
| Structural entropy + cluster-split rewrite (emits a proposed `.hymeko`) | `test_entropy.rs`, `test_split.rs` compute entropy and a split that recompiles. | PROVEN |
| Cycle/Berge enumeration, Friedler pruning, branch-and-bound scoring | `hymeko_graph` + `test_fixture_berge.rs` enumerate against fixtures with known answers. | PROVEN |

## 5. P-graph (process-network) layer

| Capability | What the evidence actually establishes | Status |
|-----------|-----------------------------------------|--------|
| P-graph schema + A1–A5 axioms; MSG / SSG / ABB | `hymeko_pgraph` (`axiom_witness.rs`, `byproduct_filter_phase11.rs`) exhibits axiom witnesses and the algorithmic constructions. | PROVEN |
| Book validation against Friedler / Pimentel worked examples | The suite reproduces the textbook Examples (3.2, 6.1, …) from `data/pgraph/Chapter*/`, i.e. matches published answers — the strongest form of correctness evidence available here. | PROVEN |
| Audit protocols ↔ reachability rules (unifying leakage audit + P-graph axioms) | Plan + argument + tests exist (2026-06-14); a candidate unification, not yet an external result. | RESEARCH |

## 6. ROS2 / simulation demos

| Capability | What the evidence actually establishes | Status |
|-----------|-----------------------------------------|--------|
| Multi-context (grasping / maintenance / safety) context-evaluation node | Runs in `hymeko_ros2_demo` with `test_evaluate_context.py`, `test_topic_binding.py`. | SHIPPED |
| UR5e Gazebo + MoveIt2 stack; dual-robot handover | Launch files + sim assets present and demoed. | SHIPPED |
| MuJoCo RL grasp/reach env emitted from `.hymeko` | Built and run in the 2026-06-19 reports. | SHIPPED |

**Scope.** These are `SHIPPED`, not `PROVEN`: they work in the demo environment and
have Python unit tests around pieces, but no single regression test pins the
end-to-end ROS2/Gazebo path, so it can regress silently on an environment change.

## 7. Reinforcement learning — descriptions → MDP

| Capability | What the evidence actually establishes | Status |
|-----------|-----------------------------------------|--------|
| Declarative observation / reward specs in HyMeKo | The MDP's observation and reward are authored as `.hymeko`, not Python; reports `2026-06-19-hymeko-rl-declarative-*`. | PROVEN |
| `AgentSpec.from_hymeko` (obs + action bounds + reward + vertex count → one MDP) | Composes a complete MDP spec from robot + task descriptions; a parity test vs. the hand-built env passes (97 `hymeko_rl` tests green, per report 2026-06-19). | PROVEN (per RL suite) |
| Shared reward / observation models reused across agent descriptions | `arm_reach_reward.hymeko` / `arm_reach_observation.hymeko` imported by multiple tasks; the reward `dist` decl hash is unchanged vs. the duplicated form (semantics preserved). | PROVEN |
| PPO fine-tuning on the BC-pretrained policy | **Negative-to-open:** PPO currently *degrades* the behaviour-cloned policy; leading hypothesis is a truncation-bootstrap bug, not yet isolated. Paused. | RESEARCH (paused) |

## 8. Tooling surface

| Capability | What the evidence actually establishes | Status |
|-----------|-----------------------------------------|--------|
| CLI (`validate`, `compile`, `transform`, `query`, entropy, rewrite) | `hymeko_cli`; `validate` on a fresh scenario returned ✅ *(measured 2026-06-20)*. | PROVEN |
| In-browser WASM editor with relationship-enriched snapshots | Built via `wasm-pack` (tools.yaml, §1-approved 2026-06-12); runs in `docs/editor/`. No automated browser regression. | SHIPPED |
| IPC daemon + client; MCP server | `hymeko_daemon` / `hymeko_client` (API frozen, CORE), `hymeko_mcp` present. | SHIPPED |
| Benchmark corpora + size sweeps | `hymeko_bench/corpora/` + artifact generator. | SHIPPED |

## 9. Research lines — reported as found

| Line | What is actually known | Status |
|------|------------------------|--------|
| Signed-link prediction; leakage audit (label-shuffle, Gömb-strict) | Audit harness + vectorisation (~6×) + figure done; 5-seed grid running (2026-06-14 line). The contribution is the *audit protocol*, results in progress. | RESEARCH |
| HSiKAN geometric / triad attention; rotor tuning | A/B head built; closing a ~0.02–0.03 AUROC gap to SiGAT is the open question, attributed (untested) to attention, not cycles. | RESEARCH |
| HyMeYOLO / Soma-vision detection | **Mostly negative:** Soma-vision is falsified for vision in the backlog; HyMeYOLO D-3/H remain open. Held-out VOC mAP is low (~0.015). Reported as a negative result, not a capability. | RESEARCH (negative) |
| **Cayley/Clifford rotor embedding** (inductive, leakage-free) | **Positive, with honest framing:** validated on signed-link as *Pareto-efficient*, not an accuracy win — near-best AUROC at ~270× fewer parameters, leakage-free under label-shuffle, across 5 datasets × 5 seeds (`reports/rotor_biggraph_20260616.jsonl`). It beats DADSGNN/SGCN but trails SiGAT by ~0.02 at that size; that trade is the result. 9 unit tests. | RESEARCH (validated) |
| S¹ `(cos,sin)` rotor encoding of RL **joint angles** | **Falsified, narrowly:** as one arm of the reach ablation only. Revolute joints are range-limited (±2.5 rad) and never reach the ±π wrap, so the rotor's periodicity buys nothing (1-seed smoke: rotor 0.329 vs hsikan 0.240 m). This does **not** bear on the rotor *primitive* above — different mechanism, different domain. Reverted. | FALSIFIED (scoped) |

**Reading §9 fairly.** The two rotor rows are deliberately distinct: the *embedding
primitive* is a validated, parameter-efficiency result; the *RL joint-angle encoding*
is a single falsified ablation. Neither implies the other. The vision line is reported
as negative on purpose — a capability ledger that hid its null results would not be
worth reading.

---

## Maintenance rule

This ledger earns its keep only by staying true. When a capability lands or changes
status, update the row **and** its anchor in the same change. Prefer a row that says
"SHIPPED, not regression-pinned" over one that rounds up to PROVEN. A row with a dead
anchor is worse than no row.
