# `reports/` aggregate index

Machine-generated inventory plus one-line pointers into each narrative `*.md`.  
Regenerate counts with: `find reports -type f | wc -l` and `find reports -type f | sed 's/.*\.//' | sort | uniq -c`.

**Scope:** only `reports/` (not repo-root `RESULTS_VIEWS_SUITE.md`; cross-linked below).

**Evidence / tone contract:** `docs/RESULTS_DISCIPLINE.md` (also `.cursor/rules/results-discipline.mdc` for assistants).

---

## 1. Volume by extension

| ext | count (approx.) |
|-----|-----------------|
| err | 108 |
| json | 105 |
| log | 47 |
| pdf | 30 |
| tex | 23 |
| md | 21 |
| jsonl | 12 |
| out | 11 |
| aux | 11 |
| txt | 10 |
| png | 4 |
| py | 3 |
| other | 15 |

**Total files (find):** 388 (snapshot at index creation).

---

## 2. Subdirectories (experiment drops)

| directory | json | err | note |
|-----------|------|-----|------|
| `abb_global_smoke/` | 18 | 18 | ABB / bench smoke artifacts |
| `overnight_2026_05_11/` | 26 | 26 | main overnight grid |
| `overnight_2026_05_11_aborted_ulimit/` | 20 | 20 | aborted (VAS `ulimit` class) |
| `overnight_2026_05_11_stage4/` | 10 | 10 | staged Epinions / Bitcoin |
| `overnight_2026_05_11_stage5/` | 17 | 17 | walks / kitchen-sink queue (`MASTER.log`) |
| `overnight_2026_05_11_stage6/` | 0 | 0 | empty |
| `overnight_2026_05_11_stage7/` | 10 | 10 | HymeYOLO ricci / edge_cr etc. |
| `overnight_2026_05_11_stage7_5/` | 0 | 0 | empty |
| `overnight_2026_05_11_stage8/` | 0 | 0 | empty |
| `overnight_2026_05_11_stage9/` | 0 | 0 | empty |
| `overnight_2026_05_11_voc_gomb_matrix/` | 0 | 4 | errors only in snapshot |
| `cpml_5seed_2026_05_11/` | 3 | 3 | Epinions CPML 5-seed |
| `cpml_factorial_2026_05_11/` | 0 | 0 | empty |
| `figures/` | — | — | plots / PNG |
| `pdf/` | — | — | built PDF mirrors of some reports |

---

## 3. Root-level machine artifacts (`reports/*.jsonl`, `*.meta.txt`, …)

Representative (not exhaustive — use `ls reports/*.{jsonl,meta.txt}`):

- `gomb_tune_20260512_*.jsonl` — Gömb tuning sweeps
- `hsikan_lean_bitcoin_overnight_20260512_004029.jsonl` (+ `.meta.txt`) — 90-cell lean profile grid (complete in snapshot)
- `hsikan_lean_*003928*` — aborted / no-torch variants
- `hsikan_otc_pvk128_h12_5seed_*.jsonl` — OTC 5-seed HSiKAN variants
- `hsikan_chain_wait_both_20260512_010400.meta.txt` — orchestration meta
- `hsikan_pro_wait_otc5seed_20260512_075153.meta.txt`

Pair every `.json` with same-stem `.err` under overnight dirs for full logs.

---

## 4. Thesis IV numeric aggregate (outside this folder)

| artifact | path |
|----------|------|
| Tabulated paired Δ / t / W/L for 111 runs | `../RESULTS_VIEWS_SUITE.md` (repo root) |
| LaTeX source for suite write-up | `reports/thesis_iv_views_suite.tex` |
| companion table / PDF | `thesis_iv_views_table.tex`, `thesis_iv_views_suite.pdf`, `thesis_iv_executive_summary.pdf` |

---

## 5. Narrative reports (`reports/*.md`) — title + gist

- **`2026-05-10-abb-global-topk.md`** — Global top-K ABB in `hymeko_graph::topk_cycles`; new `BoundedScorer` + `enumerate_top_k_cycles_bb` / `_par_bb`; global paths opt-in, per-vertex APIs unchanged.
- **`2026-05-10-abb-hsikan-smoke-and-builder.md`** — Post-ABB HSiKAN smoke + `TopKBuilder` refactor + entropy-heuristic plan follow-ups.
- **`2026-05-10-csr-sign-lookup.md`** — CSR-aligned sign lookup, scratch hoist, inline heap entries; Epinions k=4 profile speed target.
- **`2026-05-10-degree-adaptive-mv.md`** — `HSIKAN_TOPK_MODE=per_vertex_adaptive`; smoke **passed** all c tested; best c=1 **+6.7 pp** vs fixed m=128 on abbreviated Epinions (single-seed); **5-seed promotion not claimed** in §1.
- **`2026-05-10-entropy-vertex-uniform-cycles.md`** — Entropy / inverse-degree uniformity heuristics + ABB; **negative** HSiKAN smoke vs gate.
- **`2026-05-10-epinions-lift-studies.md`** — c-sweep cycle distribution + lever matrix to stack on adaptive m_v; CPU probes + plots in `figures/`.
- **`2026-05-10-hybrid-alpha-scorer.md`** — α-blended cycle scorer theory + sweep; builds on ABB + uniformity negative results.
- **`2026-05-11-abb-global-fullness.md`** — Per-vertex ABB v2: global-min witness, fullness gating, composite scorer; addresses OTC/Bitcoin AUC regressions vs v1.
- **`2026-05-11-anti-pattern-audit.md`** — CLAUDE §6.5 sweep counts (PyO3 Cartesian surface, algorithm-in-Py, `run_*` duplication, etc.).
- **`2026-05-11-clippy-workspace-green.md`** — Workspace `cargo clippy -- -D warnings` brought to green.
- **`2026-05-11-hymeko-gomb-sphere.md`** — Gömb three-shell cascade feasibility; tests + smoke + OTC 5-seed + ablations.
- **`2026-05-11-hymeko-gomb-slashdot-sota-attempt.md`** — Slashdot vs `edge_cr` SOTA: **loss ~−2.3σ**, Gömb mean **~0.9031** vs **0.9067** reference; architectural ceiling note.
- **`2026-05-11-phase2-runtimeconfig.md`** — All `HSIKAN_*` / `HYMEKO_*` reads in `signedkan_wip/src/` routed through `runtime_config.RuntimeConfig`.
- **`2026-05-11-walks-augmented-epinions-5seed.md`** — Epinions 5-seed: baseline **0.7392** vs kitchen-sink **0.8145**, paired Δ **+0.0753**; single-seed ablation table for walks / CPG / h.
- **`2026-05-12-gomb-outer-perf.md`** — Batched Gömb outer FIR + `scatter_mean`; benchmark `gomb_outer_timing`; optional `torch.compile`.
- **`2026-05-12-hsikan-lean-enumeration-harness.md`** — `run_hsikan_lean_profile.py` + shell wrapper; JSONL harness; 90-cell Bitcoin overnight + `--python` for systemd/cron.
- **`2026-05-12-memory-close-experiment-benchmark-plan.md`** — Protocol to snapshot `free`, `/proc/meminfo`, summed RSS before/after closing heavy consumers.
- **`ph12_path_i_negative_result.md`** — Path I (`total_correlation_mi`) at λ=0.1: **strong negative** on spirals (Δ −1.014 pp, ***); λ scan; faint signal at λ=0.03.
- **`phase7c_brief.md`** — Spec / commands for Phase 7c breadth (ResMLP-20, Fashion, KMNIST, SVHN, EMNIST letters, Highway-20); **expectations** in §4 — validate against `RESULTS_VIEWS_SUITE.md`, not this file alone.
- **`phases_11_12_13_brief.md`** — Activation-side programme (Paths F/I, ph11–14): configs, logs under `/tmp/thesis_iv_views_ph*.log`, CSVs under `data/benchmarks/`.
- **`sanchez_giraldo_framework.md`** — Math note: matrix Rényi entropy, Hadamard joint kernel, Path F formalism (cross-layer MI).

---

## 6. PDF / TeX compendium (filenames only)

Built papers and briefs live alongside markdown; includes `phase7c_brief.pdf`, `phases_and_paths.pdf`, `regularizer_taxonomy.pdf`, `hsikan_hymeko_brief.pdf`, Triton kernel reports (`triton_kernel_*.{tex,pdf}`), `cycle_cache_report_2026_05_10.*`, `student_lecture_2026_05_06.*`, `meeting_pimentel_outline.*`, `lyapunov_entropy_limits.*`, `generation_and_emission.*`, `topk_cycles_brief.*`, `pgraph_hymeko_brief.*`, `test_scenario_objectives.*`, duplicates under `pdf/` for subset of 2026-05-10 reports.

---

## 7. How to use this index

1. **Human narrative:** start at §5 by date or subsystem name.  
2. **Numbers:** thesis IV table → `RESULTS_VIEWS_SUITE.md`; HSiKAN / Gömb / overnight grids → §2–§3 JSON(+`.err`).  
3. **SignedKAN experiment JSON/JSONL ledger (parallel programmes):** `signedkan_wip/experiments/results/AGGREGATE_index.md`.  
4. **Regenerate §1 counts** whenever the tree changes; §5 prose is hand-curated and can drift — prefer the source `.md` for detail.
