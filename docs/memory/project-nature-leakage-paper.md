---
name: project-nature-leakage-paper
description: "Nature/Springer signed-link leakage-audit paper (HSiKAN/Gömb-strict) — refs + integrity fixes done 2026-06-14, Phase B shuffle runs still missing"
metadata: 
  node_type: memory
  type: project
  originSessionId: 51a9ecdd-e07f-4cb9-b952-6f107a810e81
---

**Paper:** `D:\hakiko_ai_ws\01_analysis\articles\superweapon\nature_hsikan\` — single
`sn-article.tex` (Springer Nature `sn-jnl`, `sn-mathphys-num`), Overleaf-synced git.
Title: *"A label-shuffle audit reveals systematic leakage in signed-link-prediction
benchmarks, and a strict protocol that holds."* Authors: Hajdu, Csapó, Kovács.
Thesis: 2018–2025 signed-link methods leak test-edge signs via transductive cycle
features; a train-only label-shuffle audit exposes it; **Gömb-strict** (3-shell
cascade: Clifford-FIR / HSiKAN / CPML + ABB + Cartwright–Harary pruner) survives it.

**Done this session (2026-06-14), UNCOMMITTED in the nature repo** (`sn-article.tex`,
`sn-bibliography.bib`); compiles clean (0 undefined cites/refs, 10pp). Rebuild on Overleaf.
- **References built**: the bib was the Springer template stub (`bib1`–`bib13`); replaced
  with real entries for all 10 cited keys + leakage lit (`kapoor2023leakage` Patterns 2023,
  `roberts2021common` Nat Mach Intell 2021), cited the leakage refs in Background, filled
  `\keywords`. Several entries flagged `% VERIFY` (SiGformer/SE-SGformer/DADSGNN authors+DOI,
  own `hajdu2026hsikan` venue) — confirm before submission.
- **Table 1**: SiGAT real filled `0.903 / 0.932` (†, 5-seed mean from
  `signedkan_wip/experiments/results/master_table.md` `sigat_attn`); shuffle TBD.
- **Table 4**: param range `30–50k → 24–30k` (exact 30487/23815).
- **Table 2 WikiElec cell filled (2026-06-14)**: Gömb-strict WikiElec **87.30%** acc
  (Δ +6.67 vs SE-SGformer), AUROC **0.9114±0.0013** also added as 6th col of Table 3.
  Derived from 5-seed strict run `gomb_wiki_datasets_20260514T150002Z/wiki_elec_seed{0-4}.log`
  (confirmed `[protocol] STRICT: cycle pool over 82940 training edges only`); accuracy back-solved
  from logged per-class P/R + n_test via precision-identity (same method as other Table-2 cells).
  Prose added to "Epinions cell" para. Rebuilt clean (10pp, 0 undefined). Still uncommitted.
- **Table 2 integrity fix**: caption rewritten — Gömb-strict accuracy is **derived** from
  logged per-class P/R (5-seed, strict, same-machine); SGCN+SE-SGformer are **published
  transductive** transcriptions (now `\cite`'d + ‡ legend). Removed the false "all results on
  one RTX 2070". Epinions paragraph rewritten honestly (SE-SGformer loses to SGCN in *its own
  published* numbers; comparison is cross-convention, conservative for us).
- **Param claim fix** (§3.5 + Discussion): Gömb-strict cascade is embedding-dominated
  (355k–2.1M on disk, node_embed scales with |V|); reworded to shells+core 10⁴–10⁵, full count
  low millions, only the *architectural* budget is "orders below" transformers.

**Data provenance** (all under `signedkan_wip/experiments/results/`):
- Gömb shuffle audit: `hsikan_rescore_audit_20260517T191841Z/task_b_shuffle_audit.jsonl`.
- 5-seed strict cascade (Table 3): `gomb_strict_benchmark_tuned_20260514T010516Z/` (+ reddit
  `phase_c_20260517T172421Z/`). `005041Z` dir is all-OOM — DO NOT cite.
- Table 2 source: `docs/GOMB_SOTA_COMPARISON_2026_05_17.md` (SGCN/SE-SGformer published;
  Gömb accuracy "algebraically derived"). HSiKAN-Optuna 10-seed: `bitcoin_optuna_best_5seed_2026_05_13.jsonl` (despite name, seeds 0–9).

**STILL BLOCKING (the headline's evidence gap):**
- **Phase B**: transformer-baseline SHUFFLE runs (SiGAT/SGCL/SiGformer/SE-SGformer/DADSGNN)
  are genuinely ABSENT — never run. The abstract's *"2025 transformers drop 6–40 pp under
  shuffle"* has NO supporting cell; only SGCN/Gömb/HSiKAN have shuffle numbers. This is the
  one thing that lets the central claim stand on its own evidence.
  **DECISION 2026-06-14 (user): RUN ALL 4 — reverses the 2026-05-17 "cite, don't run" Phase B
  call.** Plan: `docs/plans/2026-06-14-phase-b-baseline-shuffle-audit/`.
  **HARNESS BUILT + VERIFIED 2026-06-14** (report `reports/2026-06-14-phase-b-baseline-shuffle-audit.md`,
  this machine = RTX **3070 Laptop 8 GB**, NOT 2070; torch 2.12+cu132): `SignedLinkBaseline`
  registry (`src/baselines/registry.py`), 3 legacy wraps (sgcn/sigat/sgt), 4 reimpls
  (`{sgcl,sigformer,sesgformer,dadsgnn}.py`, each reuses existing blocks), ONE CLI
  `run_baseline_audit.py --model <m>` (seed+100003 shuffle, `test_auroc`+`n_params` JSON),
  `--baselines` grid mode in `run_no_leak_benchmark.py` (per-runner `Runner.arg_builder`, gomb
  path byte-identical). 14 pytest green; SGCN reproduces **bit-exact** 0.8547/135585; E1 smoke
  regression intact; ruff clean. All 7 learn (bitcoin_alpha 60ep real: sgcl .879 sigformer .901
  sesgformer .893 dadsgnn .854; shuffled→~.52-.55). **STILL NOT RUN: the 5-seed×5-dataset grid.**
  **§11 COST FINDING (disagrees with plan's 24 GPU-h):** python per-node attention in
  sigat/sgt/sigformer/sesgformer = ~23 s/epoch on Epinions (no OOM, GPU peak 1.93 GB) →
  full-epoch grid ≈ **80–90 GPU-h**, dominated by the 4 transformers on Slashdot+Epinions; the
  3 sparse-mm methods (sgcn/sgcl/dadsgnn) are cheap (~hours). `--patience` early-exit is NOT free
  (sgcl improves to ep185/200; patience=12 loses 1.5pp → undertrains a baseline = integrity risk),
  so the grid trains full epochs. **3 GRID OPTIONS open for the user** (report Open Issues): (a) run
  3 cheap methods ×5 ds now (~hrs), defer transformers; (b) 3 seeds for transformer×large cells;
  (c) **vectorize the attention** → ~80h becomes single-digit GPU-h.
  **USER CHOSE (c) "vectorize, then decide" 2026-06-14; DONE + MEASURED.** Report
  `reports/2026-06-14-vectorize-signed-attention.md`, plan `docs/plans/2026-06-14-vectorize-signed-attention/`.
  New shared `src/baselines/_attention.py` (`build_csr` + `segment_attention`: cached CSR segment
  attention, scatter_reduce(amax)+index_add, O(E) mem, no padding); `MotifAttention`(sigat_model) +
  `SignedAttention`(sgt) swapped to it, **encode_nodes signatures UNCHANGED** (other runners
  run_inference_bench/run_sgt_sweep/test_sota_compare untouched). 19 pytest green incl. parity vs
  naive ref @1e-5 (both bias/no-bias paths), ruff clean. **MEASURED ~6× speedup: sesgformer Epinions
  ~22.9→~3.7 s/epoch, GPU peak 2.78 GB (no OOM), AUROC 0.903.** Grid now **~10-15 GPU-h, feasible
  locally** (overnight) — cloud now optional (asked re GCP; bottleneck was CPU-side python so a
  bigger GPU wouldn't help, only horizontal fan-out would; turnkey GCP kit offered, not built).
  **5-SEED STRICT GRID DONE 2026-06-14** (350 runs, ~2.3h, peak RSS 524 MB,
`no_leak_baselines.jsonl`; figure `signedkan_wip/paper/figures/leakage_audit.{pdf,png,eps}`
regenerated w/ error bars). Per-method drop 33–37pp; shuffled means 0.515–0.545. **Leak cells
(shuf>0.55) concentrate on reddit_body: 26/35; bitcoin_alpha 12, bitcoin_otc 6, epinions 1,
slashdot 0.** Shuffled-AUROC by dataset: epinions **0.500**, slashdot **0.514±0.006** (CLEAN to
chance), bitcoin_alpha 0.534, **reddit_body 0.564±0.024 (stable residual, NOT noise)**. So strict
holds cleanly on the large balanced graphs; reddit retains ~0.56 under shuffle. Class-imbalance
explanation REFUTED (bitcoin_alpha pos_frac 0.936 > reddit 0.926 yet lower residual). **Hypothesis
(unconfirmed):** label-shuffle preserves topology, so a node-topology sign prior (popular nodes →
positive links) survives it; needs a degree-preserving rewire control or per-node-prior baseline to
isolate. KEY caveat for the leak half: reimpls use diffuse node-embedding+MLP readout → full
transductive does NOT leak (sgt 0.514); only per-edge cycle features leak (0.73). The decisive
remaining experiments: R_topo wiring+sweep across readout families, reddit rewire control,
native-readout check. **NEXT decision point — user to choose which.** Reproduction-parity
  caption still required (our reimpls, not authors' weights). SiGAT shuffle cell (Table 1) now comes
  from the grid (sigat is in the 7 methods). **Audit figure script BUILT 2026-06-14**:
  `signedkan_wip/src/paperkit/build_leakage_audit_figure.py` (reads no_leak_baselines.jsonl → per-method
  real-vs-shuffled AUROC bars + chance line @0.5, emits paper/figures/leakage_audit.{pdf,png,eps};
  tolerates partial data; ruff clean, validated on synthetic). **GRID LAUNCHED 2026-06-14 (1-seed pass,
  bg)** → no_leak_baselines.jsonl, resumable; run 5-seed after to finish. After grid: regenerate figure,
  fill Table 1 cells from the JSONL verdicts, rebuild paper.
- SiGAT shuffle TBD (Table 1). (Table 2 WikiElec Gömb cell now DONE — see above.)
- **No figure in body** (`fig.eps` unused) — the audit (real-vs-shuffled bars, chance line at
  0.5) is the obvious missing figure, a cheap high-value add.

**Why:** brought by the user 2026-06-14 "in the meantime"; distinct from the
[[project-smc-paper-additions-queue]] SMC/TSMC paper. **How to apply:** the integrity fixes
make the tables trustworthy; the remaining work is empirical (Phase B shuffles) + the figure +
bib VERIFY. Don't fabricate baseline shuffle numbers — they must be run.

**5-SEED R_topo DONE 2026-06-14** (no_leak_baselines_topo.jsonl, 350 distinct cells; 420 rows = 70 dup seed-0, dedup by key). VERDICT: per-method topo-shuffle ~= strict within +-0.6pp -> baselines do NOT leak structurally (confirms readout-locality). Localized: on EPINIONS 4/7 (sgcn,dadsgnn,sigformer,sgcl) topo lifts shuffled +0.037-0.052 (0.48->0.51-0.54), 5-seed-consistent but SUB-GATE = reachability not leakage. The 0.73 leak was cell_signed_graph (LOCAL readout). TODO: target cell_signed_graph + R_topo masking.

**CYCLE-METHOD R_topo VERDICT 2026-06-15** (report reports/2026-06-15-cycle-leak-reachability-verdict.md). cell_signed_graph+HSiKAN, bitcoin_alpha 60ep, 3-way (strict/topo/full)x(real/shuffle): strict 0.500/0.500 (excludes test-edge cycles->no features, degenerate); topo 0.806/0.467 (cycles kept as topology, test SIGN masked to +1 in sigma via HSIKAN_REACH_TOPO); full 0.901/0.735 (LEAK reproduced). KEY: the 0.735 leak is the DIRECT sigma-sign channel NOT structural -- masking only the test sign (topology intact) drops shuffle 0.735->0.467 chance. BONUS: cycle topology is a LEGITIMATE non-leaking feature (topo real 0.806, shuffle chance); strict OVER-CORRECTS (0.500 real, kills real signal). => R_topo is a BETTER protocol than strict (keeps structural signal, leak-free). Lattice: leak only at R_full (label reachable). Ties readout-locality: local-readout cycle method leaks at full (diffuse baselines do not) but NOT at topo => leak needs LABEL reachable + local readout. Code: runtime_config.reach_topo (HSIKAN_REACH_TOPO env, mirrors strict_protocol) + run_final_cell mask g.signs[te_idx]=1 after s_te/y_te captured. Additive/env-gated/default-off. Follow-up: multi-seed/dataset R_topo for error bars; this 3-way = candidate central mechanism figure.
