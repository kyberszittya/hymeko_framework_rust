---
name: project-reachability-rules-article
description: "Reachability rules unifying the signed-link leakage audit with the P-graph axioms (MSG/SSG/ABB) — plan+argument+tests done 2026-06-14, possible standalone article"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a8e98b2-6e30-4006-a8d4-ecf2fe971782
---

User's idea (2026-06-14): the leakage-audit's transductive protocols
(topology-only, full) are **reachability rules** that could also serve in
`hymeko_pgraph`'s ABB/MSG/SSG **alongside the five Friedler axioms (A1–A5)**.
Asked for "a plan and argument," then "test cases as well… maybe basis of a new
article." All delivered 2026-06-14.

**Artifacts:** plan dir `docs/plans/2026-06-14-reachability-rules-audit-pgraph/`
(`argument.md` = the article seed; `plan.{tex,pdf,tikz,mmd}` compile); report
`reports/2026-06-14-reachability-rules-audit-pgraph.md`.

**The argument (grounded in real code):** reachability is *already* axiomatic in
pgraph — A4/S4 and E-StrictNoExcess are BFS-to-product predicates; the ABB bound
is `msg::close_producible` (forward fixpoint seeded by raws); `axiom_extensions.rs`
is the precedent for orthogonal rules slotting alongside S1–S5; `regime.rs` is a
ready `Regime` Strategy seam (Canonical/NoExcess). The audit protocols parameterize
*which edges/labels seed the closure*: `strict ⊆ topo ⊆ full` (monotone lattice).

**Code + tests (all green):** Python `signedkan_wip/src/baselines/reachability.py`
(`ReachabilityRule` enum + `reachable_edges`) + `tests/test_reachability.py` (8
tests: reduction, lattice, leakage-invariant). Audit wired: `run_baseline_audit`
gained `reachability=`/`--reachability` (default `strict` = bit-identical
reduction). Rust `hymeko_pgraph/src/reachability.rs` (`close_producible_under_rule`)
+ 4 tests (reduction = canonical closure, candidate-unlocks-product, monotone,
flags). pgraph suite 41/41, ruff/clippy/fmt clean.

**KEY EMPIRICAL FINDING (refines the thesis, argument §5a):** on bitcoin_alpha,
sgt drops to chance under shuffle even under `full` (0.514) — full-adjacency
inclusion does NOT leak for diffuse node-embedding readouts. Contrast cycle-based
path leaked at 0.73 (2026-06-11). So **leakage = reachability rule × readout
locality**: the label being reachable-in-graph is necessary but not sufficient;
the readout must expose it (per-edge cycle features do; aggregated node embeddings
don't). This two-factor account is the candidate article headline.

**EXPANDED 2026-06-14 to a multi-article program** (user: "plan + mathematical report,
multiple articles from P-graph optimization, neural networks, cognitive agents").
Delivered: **mathematical framework report** `reports/2026-06-14-reachability-mathematical-framework.{tex,pdf}`
(closure-operator algebra; Defs reachability-rule + lattice R_strict⊑R_topo⊑R_full; **Thm 1
monotonicity, Thm 2 reduction [=regression test], Thm 3 bound-admissibility** all proved; 3
instantiations: A P-graph producibility closure, B msg-passing closure w/ leak = rule×readout-locality
factorization [Prop, empirical], C AKOIRE refinement closure under parser gatekeeper [framing]; §8
epistemic-status honesty). **Multi-article plan** `docs/plans/2026-06-14-research-program-reachability/`
(4 fmts compile, mermaid valid): Article B (NN leakage, MOST MATURE, lead — subsumes Nature paper,
upgrades "transformers leak" → "which readouts leak & why"), Article A (P-graph regimes, code done),
Article C (cognitive agents, least mature — formalize first), Article 0 (umbrella, last). Sequencing
B→A→C→0. **R_topo now FULLY WIRED+TESTED** across readout families (sgcn/sgcl/dadsgnn both-channels,
sigat/sigformer both-buckets, sgt/sesgformer neutral-bias + degree-skip; 10 reachability tests incl 2
per-model wiring tests). Risk flagged: over-unification; native-readout caveat; R_topo sweep is
load-bearing for B.

**NOT done (follow-up):** (1) Phase 2 — thread `close_producible_under_rule` into
the ABB bound + a `ReachabilityRegime` with the admissibility test (ABB optimum
invariant across rules). (2) Per-model `topo` neutral-sign wiring (SGCN pos/neg
split drops a 0-sign edge → topo doesn't yet include test topology for signed-split
models). (3) The `R_topo` empirical sweep across readout families = the article's
experimental section. **Status: plan+argument+tests done; further code awaits user
go-ahead.** Related: [[project-nature-leakage-paper]] (the audit), the SMC/PNS line.

**CYCLE-METHOD R_topo VERDICT 2026-06-15** (report reports/2026-06-15-cycle-leak-reachability-verdict.md). cell_signed_graph+HSiKAN, bitcoin_alpha 60ep, 3-way (strict/topo/full)x(real/shuffle): strict 0.500/0.500 (excludes test-edge cycles->no features, degenerate); topo 0.806/0.467 (cycles kept as topology, test SIGN masked to +1 in sigma via HSIKAN_REACH_TOPO); full 0.901/0.735 (LEAK reproduced). KEY: the 0.735 leak is the DIRECT sigma-sign channel NOT structural -- masking only the test sign (topology intact) drops shuffle 0.735->0.467 chance. BONUS: cycle topology is a LEGITIMATE non-leaking feature (topo real 0.806, shuffle chance); strict OVER-CORRECTS (0.500 real, kills real signal). => R_topo is a BETTER protocol than strict (keeps structural signal, leak-free). Lattice: leak only at R_full (label reachable). Ties readout-locality: local-readout cycle method leaks at full (diffuse baselines do not) but NOT at topo => leak needs LABEL reachable + local readout. Code: runtime_config.reach_topo (HSIKAN_REACH_TOPO env, mirrors strict_protocol) + run_final_cell mask g.signs[te_idx]=1 after s_te/y_te captured. Additive/env-gated/default-off. Follow-up: multi-seed/dataset R_topo for error bars; this 3-way = candidate central mechanism figure.

**ARTICLE DRAFT WRITTEN 2026-06-15:** D:/hakiko_ai_ws/01_analysis/articles/reachability_leakage_audit/article.{tex,pdf} (self-contained article class, 5pp, compiles clean, author=Csaba Hajdu solo, marked DRAFT). Sections: reachability-rule lattice (strict<=topo<=full), two-factor account (leak = label-reachability x readout-locality), experiments (7-baseline 5-seed: no structural leak Rtopo~=Rstrict +-0.6pp; cycle-method 3-way Table: strict 0.50/0.50, topo 0.806/0.467, full 0.901/0.735 -> leak is DIRECT sigma-sign channel; Rtopo better protocol), discussion (Rtopo recommended; strict over-corrects), limitations (cycle 3-way single-seed; reimpl readouts). This = Article B (NN leakage) of the multi-article program, now with the mechanism nailed. Follow-up to article-grade: multi-seed/dataset cycle 3-way for error bars; add the leakage_audit figure; cite leakage lit (Kapoor/Roberts).

**TPAMI VERSION + 5-SEED MEASUREMENTS 2026-06-15** (user chose TPAMI venue): article_tpami.tex (IEEEtran compsoc journal, 3pp, compiles; reachability_leakage_audit/ dir, IEEEtran.cls copied). Cycle 3-way grid DONE 5 seeds x {bitcoin_alpha,bitcoin_otc} (signedkan_wip/experiments/results/cycle_reachability_grid.jsonl, 60 rows): strict 0.500/0.500 both; topo BA 0.804+-.017/0.474+-.027, OTC 0.840+-.016/0.490+-.018; full BA 0.890+-.023/0.743+-.041, OTC 0.864+-.012/0.664+-.030. LEAK ROBUST both datasets (full-shuf 0.74/0.66); topo-shuf=chance (leak-free); topo-real strong. Folded into Table 2 (error bars), abstract, limitations (no longer single-seed). SMC review (smc_02/review_2, ChatGPT, Hungarian) sep: #2 any-instantiation qualifier DONE; #3 rho<0.7 NON-ISSUE (current source rho>=1 everywhere, reviewer saw old draft/conflated w/ 0.75ms latency or 0.70 scaling exp); #1 perf-tone + #5 soften comparison tables + #4 add measurable non-robotics eval = await user nod.
