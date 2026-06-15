# HyMeKo — Backlog (open work)

**Last updated:** 2026-06-15 · Companion: [DONE.md](DONE.md) (completed work).

The canonical record stays in `docs/plans/` (plans) + `reports/` (acceptance). This
file is a hand-maintained **index of what's still open**, seeded from the
2026-06-15 plan audit (Soma-vision + HyMeYOLO lines), the 2026-06-15 full
plan-dir sweep (~127 non-vision plan dirs) and the project memory.

**Convention.** One line per item: `[area] description — plan/ref — STATUS — prio`.
STATUS ∈ OPEN / PARTIAL / QUEUED / BLOCKED / DECISION. When an item finishes, move
its line to `DONE.md` with the report link. Keep newest areas on top.

**Tally (2026-06-15):** 51 open line-items across 9 areas. Of these, ~11 are **P1**
(prefilter, HSIKANConfig, 3-dataset bench, tier-2/3 audit, Friedler-quotient,
reachability-rules pgraph, Phase-B shuffle, Nature-Comm audit, Stage D-3, Stage H,
cortical Brain-Score). The May signed-link items predate the leakage audit —
re-validate any AUC against the σ-masked strict protocol before reporting.

---

## Soma-vision (Gömb perception)

> **Decision context (2026-05-28 re-bench):** hypergraph-vision is *empirically
> falsified at small scale* — RicciStim/HSiKAN/HGNN all lose to a plain MLP on
> MNIST/Fashion; RicciStim Cluttered-MNIST = 0.14 mAP₅₀ (< 0.23 baseline). Code
> is correct (~3.8 kLOC, 168+ tests); the *approach* doesn't help vision here.
> So most items below are "explain / decide", not "build more architecture."

- [ ] **Explain the Cluttered-MNIST loss** — component ablation (Hodge vs Ricci vs SDRF vs quadtree vs σ-balance), one-at-a-time; inspect learned edge signs on a planted test — `2026-05-14-gomb-soma-ricci-stim` — OPEN — **P1**
- [ ] **Cortical Brain-Score run** — infra + synthetic smoke done; fetch real Cichy-92, run V1/V2/V4 ROI scoring vs ResNet-tiny (paired bootstrap) — this is the *untested* place Soma might actually shine — `2026-05-16-gomb-soma-cortical-benchmark` / `…-implementation` — OPEN — **P1**
- [x] **Base-Soma vision falsification** — DONE 2026-06-15: walk-conv base-Soma 0.52 vs linear control 0.91 on MNIST (−0.387 paired, all seeds) → walks-only also falsified for vision (→ DONE.md, `reports/2026-06-15-base-soma-vs-linear-mnist.md`). Residual: base-Soma vs Gömb on *signed-link* never run, but deprioritized given the vision falsification — P4
- [ ] **TriangleConv clarification** — original plan specified a Cartwright–Harary balance-gated triangle layer; only `hg_conv_bochner.py` exists. Confirm superseded or build it — `2026-05-14-gomb-soma` §6 — DECISION — P3
- [ ] **Faithful Soma round-trip (.hymeko ↔ vision pipeline)** — `soma_vision.hymeko` is the minimal skeleton; the real Hodge/stim/patch internals aren't emittable yet — hero Phase 3 follow-up — OPEN — P3
- [ ] **Perf P6–P8** (topology cache, polygon vectorize, GPU SegmentedSparse batching) — only justified if cortical/vision use-case survives — `2026-05-15-ricci-stim-opt-pass-5` — BLOCKED (on the decision above) — P4
- [ ] Hygiene: archive/delete superseded `2026-05-14-gomb-soma-stim` (never implemented; subsumed by ricci-stim) — P4

## HyMeYOLO (detection stack)

- [ ] **Stage D-3 nodelet head** — per-query objectness gates (decouple is-object from which-class); code drafted, **production smoke not run** (≥0.05 mAP₅₀ gate, then 5-seed ≥0.20). The linchpin for the VOC claim — `2026-05-19-stage-d3-nodelet-head` — OPEN — **P1**
- [ ] **Stage H person-only VOC** — single-class diagnostic: settles whether the head (D-3) is the *only* bottleneck or upstream (D-4) is needed; also unlocks the rapport demo — `2026-05-19-stage-h-voc-eyes-for-rapport` — QUEUED — **P1**
- [ ] **VOC ep180 full run on HPC** — current floor is ep60 = 0.0149 (under-trained, DataLoader host-bound at 37% GPU util → needs pinned-memory rewrite); ep180 sets the honest publishable number — `2026-06-10-voc-test-baseline` — QUEUED — P2
- [ ] **Stage D-3-bis / D-3-tris** — λ_gate⁻ override + matcher gate-veto + focal-gate; run only after D-3 base clears — `2026-05-18-stage-d3-{bis,tris}-*` — QUEUED — P3
- [ ] **conv-as-hypergraph** — RF-orbit hyperedges + shift-equivariance unit-test gate + Bochner variants; H1/H2/H3 mAP comparison not run — `2026-06-11-conv-as-hypergraph-hymeyolo` — OPEN — P3
- [ ] Tidy: confirm Stage A-3 + Stage B′ attribution controls ran (no standalone report found; likely folded into Stage B) — P4

## Hero demo

- [ ] **Runnable torch round-trip** — numeric parity of the emitted module vs `cascade.py` (pulls torch = §1 dep) — beyond the current structural parity — hero Phase 3 — OPEN — P2
- [ ] **Task-layer emitters** — BehaviorTree.CPP / PDDL / ROS 2 action servers for the `.hymeko` task layer (FANUC handover task validates but can't emit) — OPEN — P3
- [ ] **Editor hero-cell** done (robot_arm profile); optional: a NL→.hymeko stub front-end behind the gate — P4

## Editor (docs/editor)

- [x] **3D pan in the Kinematic view** — DONE 2026-06-16: right-button/shift-drag pan ported from `hypergraph3d.js`; 66/66 editor tests pass (`reports/2026-06-16-editor-kinematic-pan.md`)
- [ ] More Steiner/design generators (S(2,3,9) via Bose already; add Pasch, K₅³, projective planes) — 1 registry entry each — OPEN — P4
- [ ] Optional: shareable `?src=` deep-link (load arbitrary source) — would also unblock headless visual tests (e.g. the 2D bidirectional render) — P4

## Signed-link learning (HSiKAN / cycles / AC-HSiKAN)

> From the 2026-06-15 plan sweep of the May signed-link line. Most of these
> predate the leakage audit; **re-validate the operating point against the
> σ-masked strict protocol before reporting any new AUC** (`[[project-nature-leakage-paper]]`).

- [ ] **Vertex prefilter** — cycle-enumeration prefilter; gates the 3-dataset bench; measured ~30–40 % Epinions k=5 savings — `2026-05-11-vertex-prefilter` — OPEN — **P1**
- [ ] **HSIKANConfig derivation** — collapse 25+ `HSIKAN_*` env vars into one typed `HSIKANConfig` dataclass parsed at startup (kills the §6.5 #11 deep-call-site env reads) — `2026-05-12-hsikan-config-derivation` — OPEN — **P1**
- [ ] **3-dataset HSiKAN v1/v2 bench** — 4-phase Bitcoin/Epinions/Slashdot run; blocked behind vertex-prefilter — `2026-05-12-3dataset-hsikan-v1-v2` — OPEN — **P1**
- [ ] **Cycles strategy refactor** — 16 `#[pyfunction]` variants → 1 config-driven entry (§6.5 #1/#9); pure-Rust algo to `hymeko_graph` (§6.5 #2) — `2026-05-11-cycles-strategy-refactor` — PARTIAL — P2
- [ ] **CPML-XHC architectures** — cross-head-coupled CPML variants — `2026-05-11-cpml-xhc-architectures` — OPEN — P2
- [ ] **Self-evolving cycle sampling** — adaptive reservoir/ABB sampler that reweights by yield — `2026-05-14-self-evolving-cycle-sampling` — OPEN — P2
- [ ] **CN optimization pattern** — common-neighbour fast path — `2026-05-13-cn-optimization-pattern` — OPEN — P2
- [ ] External-AUC tuning / orthogonal FIR sweeps — `gomb-external-auc-tuning` · `gomb-tune` · `hsikan-cpml-fir-orthogonal` — OPEN — P3

## Architectures / fuzzy / sequence

- [ ] **Signed-link tier-2/tier-3 audit** — ~32 h GPU SGT/SiGAT baseline audit; **blocks the signed-link paper's baseline table** — `2026-05-17-signed-link-tier2-tier3` — QUEUED — **P1**
- [ ] **Friedler-quotient param reduction** — structural param compression via the Friedler quotient; sizing claim unverified — `2026-06-06-friedler-quotient-param-reduction` — OPEN — **P1**
- [ ] Sequence/text + fuzzy layers — `sequence-multichannel-v2` · `text-encoder-decoder` · `fuzzy-signature-layer` · `cycle-induced-memory` — OPEN — P2
- [ ] Fuzzy pose / balance-as-activation — `fuzzy-pose-detection` · `balance-as-activation` — OPEN — P3
- [ ] Fuzzy-pose config audit + restart options — **BLOCKED on user decision** — `fuzzy-pose-config-audit` · `restart-options` — BLOCKED — P4
- [ ] KLP skyline · AC-HSiKAN follow-ups — `klp-skyline` · `ac-hsikan-followups` — QUEUED — P3

## P-graph / reachability / robotics demos

- [ ] **Reachability-rules audit (pgraph)** — thread a `ReachabilityRule` enum through `hymeko_pgraph` + regime sweep; unifies leakage audit ↔ A1–A5/MSG/SSG/ABB; trio of publications — `2026-06-14-reachability-rules-audit-pgraph` — OPEN — **P1** (supersedes the article-only line below)
- [ ] **hymeko_pgraph Python integration** — 4 Python integration tests over the pgraph crate — `hymeko-pgraph-py` — OPEN — P2
- [ ] Robotics/demo plans — `kinematic-demo` · `comm-cliques-demo` · `gomb-np-hard-approximation` · `camera-placement` · `robotics-behavior-collaboration` · `seminar-demo-program` (+ remaining) — OPEN — P2
- [ ] Gömb-as-planner-heuristic · legged-locomotion — OPEN — P3
- [ ] Gömb belief-planning bridge — `gomb-belief-planning-bridge` — OPEN — P4
- [ ] Niitsuma dual-UR5 · MDPI Technologies live demo — `niitsuma-dual-ur5` · `mdpi-technologies-live-demo` — QUEUED — P2

## Paper / leakage / SMC (from memory — verify before acting)

- [ ] **Nature leakage paper** — 5-seed σ-masked strict grid running; finish + lock the honest operating point — `[[project-nature-leakage-paper]]` — OPEN — P2
- [ ] **Phase-B baseline shuffle audit** — 5-seed shuffle grid → Nature Table 1; harness + ~6× vectorization done 2026-06-14, grid running — `2026-06-14-phase-b-baseline-shuffle-audit` — OPEN — **P1**
- [ ] **Nature-Comm leakage audit** — baseline label-shuffles + Reddit reachability control; feeds the strict-protocol claim — `2026-05-17-nature-comm-leakage-audit` — OPEN — **P1**
- [ ] No-leakage structural benchmark E2/E3 — E1 harness smoke-clean 2026-06-11; E2/E3 still open — `[[project-no-leakage-benchmark-resume]]` — OPEN — P2
- [ ] **SMC smc_02** edits are UNCOMMITTED in the smc_02 repo (14pp, review_1/2 addressed) — commit when ready — P3
- [ ] Reachability-rules article (audit protocols ↔ P-graph A1–A5) — plan+argument+tests done; possible new article — `[[project-reachability-rules-article]]` — P3 (impl tracked under the pgraph item above)

## Infrastructure / other (from memory)

- [ ] HSMM → FPGA: Nagare dataflow substrate → HSMM abstract machine → Zynq (frozen compiler plan exists) — `CRITICAL-nagare-to-hsmm-compiler` — OPEN — P3
- [ ] **Codebase rehaul Phases 2–7** — `RuntimeConfig` + unified `Experiment` framework (kills §6.5 #3 run_*.py scaffold dup, #11 globals); Phase 1 partial — `2026-05-11-codebase-rehaul` — PARTIAL — P2
- [ ] `04_graph_query` tests T13–T21 — `hymeko_query` integration suite — OPEN — P2
- [ ] `05_hre_extraction` — HRE extraction spec tests — OPEN — P2
- [ ] `06_wasm_editor` — full 7-crate WASM editor spec (current editor is the ad-hoc subset) — OPEN — P3
- [ ] G-SPHF Rust crate — awaiting CORE-edit token (new crate = §1 dep add) — `gsphf-rust-crate` — BLOCKED — P3
- [ ] Seminar deck: open `HyMeKo_Seminar.with_refs.pptx` and eyeball the faded heatmap-background opacity (one-line tweak in `insert_into_deck.py`) — `2026-06-15-seminar-deck-additions` — OPEN — P4
