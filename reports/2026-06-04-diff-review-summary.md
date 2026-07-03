# 2026-06-04 diff-review summary (for user-handled commits)

Branch: `feat/fuzzy-signature-tnorm-pooling`
Total: **25 modified** + **84 untracked** ≈ **109 paths**.

> **YOU handle commits.** This is a categorised review; no `git add`
> / `git commit` was run.

## A. 2026-06-04 (this overnight) — clean, scoped, tested

### A1. New crate `hymeko_pgraph_py/` (PyO3 binding for P-graph trio)

- `hymeko_pgraph_py/{Cargo.toml,pyproject.toml,README.md,src/lib.rs}`
- Workspace add: `Cargo.toml` (one-line member insert)
- Plan: `docs/plans/2026-06-04-hymeko-pgraph-py/{plan.tex,plan.pdf,plan.tikz,plan.mmd}`
- Test: `hymeko_neuro/tests/test_hymeko_pgraph_py.py` — **12/12 PASSED**
- CORE.YAML: none touched.

### A2. KLP sort-and-sweep skyline (D=2)

- `hymeko_neuro/hyperedge/abb_walks.py` — refactored `ssg_pareto_filter`
  to dispatch; added private `_ssg_pareto_filter_brute` (reference)
  + `_ssg_pareto_filter_sweep_2d` (new, O(N log N)).
- Test: `hymeko_neuro/tests/test_ssg_pareto_filter.py` — **31/31 PASSED**
- Plan: `docs/plans/2026-06-04-klp-skyline/{plan.tex,plan.pdf,plan.tikz,plan.mmd}`
- Report: `reports/2026-06-04-klp-skyline.md`
- Empirical: 247× speedup at N=10K.

### A3. Komondor parallelism showcase

- New: `docs/komondor_setup/submit_hsikan_edge_cr_array.sh` (40-cell array)
- New: `docs/komondor_setup/launch_k_sweep.sh` (K=20/10/5 chained launcher)
- New: `scripts/komondor_morning_pull.sh` (one-shot remote→local pull)
- New: `scripts/komondor_parallelism_analysis.py` (per-cell wall histogram + LPT)
- New: `scripts/komondor_audit_metrics.py` (existing, already on disk earlier; included for completeness)
- Plan: `docs/plans/2026-06-04-komondor-parallelism-showcase/{plan.tex,plan.pdf,plan.tikz,plan.mmd}`
- Report: `reports/2026-06-04-komondor-parallelism-showcase.{md,tex,pdf}`
- Submitted on Komondor (with user "mehet" auth): jobids
  **13885808 (K=20)** → **13885809 (K=10)** → **13885810 (K=5)**,
  all PENDING after chain 13885723 finishes.

### A4. Unified MSG/ABB/SSG reference (earlier today)

- `reports/2026-06-04-msg-abb-ssg-unified-implementation.{md,tex,pdf}` —
  18 pages with App. A code listings for the algorithm trio.

## B. Earlier-today work (already discussed)

- `reports/2026-06-03-pimentel-benchmark-reply.{md,tex,pdf}` — 9-page
  Pimentel external reply.
- `reports/2026-06-03-pimentel-benchmark-validation.md` — internal
  Pimentel benchmark validation trail.
- Pimentel benchmark fixture: `data/pgraph/Chapter6/pimentel_distractors.hymeko`
- Test: `hymeko_pgraph/tests/pimentel_distractors.rs` (cargo-runnable
  assertion: 7-unit MSG, 19 SSG, top-3 = 9/12/13).

## C. 2026-06-03 work (yesterday) — already in plan-doc form

- `hymeko_neuro/hyperedge/{reservoir.py,path_scorers.py,abb_walks.py}` —
  Vitter Algo L reservoir + ABC scorers + ABB walk enumerator + SSG Pareto.
- `hymeko_neuro/graph/cycle_cache/strategies.py` — TupleEnumerator
  Strategy + ABBWalkEnumerator dispatch.
- `hymeko_graph/src/{topk_walks.rs,walks.rs}` — Rust port of ABB walks.
- `hymeko_py/src/cycles/` — PyO3 surface for `enumerate_top_k_walks_rs`.
- Plan: `docs/plans/2026-06-03-abb-msg-ssg-walk-enumeration/`
- Plan: `docs/plans/2026-06-03-tuple-enumerator-strategy/`

## D. Modified files (full list, 25)

```
CLAUDE.md                                          (the §0 step-2 discovery rule + §6.5 #12)
Cargo.lock                                          (workspace member add)
Cargo.toml                                          (hymeko_pgraph_py member add)
SMC_TUTORIAL.md
hymeko_graph/src/lib.rs                             (topk_walks pub mod)
hymeko_pgraph/src/abb.rs                            (top_k_with_regime + improved scoring)
hymeko_pgraph/src/bin/hymeko_pgraph_dump.rs         (SSG-DM CLI flag)
hymeko_pgraph/src/bin/pgraph.rs                     (--style friedler|pse + --format pdf)
hymeko_pgraph/src/cli.rs                            (to_friedler_dot)
hymeko_pgraph/src/dump.rs                           (SsgAlgorithm enum, full options)
hymeko_pgraph/src/lib.rs                            (new pub uses)
hymeko_py/src/cycles/{mod,unsigned}.rs              (path-closure refinements)
hymeko_py/src/lib.rs                                (enumerate_top_k_walks_rs export)
scripts/pgraph/verify.sh
hymeko_neuro/experiments/runs/run_final_cell.py   (n_t bugfix at line 508)
hymeko_neuro/experiments/runs/run_phase12_position_regression.py
hymeko_neuro/hyperedge/n_tuples.py                  (subsample fix)
hymeko_neuro/hyperedge/walks.py                     (open-walk enumerator)
hymeko_neuro/graph/cycle_cache/{__init__,api,stats}.py  (cache type fix)
hymeko_neuro/experiments/hsikan_pgraph_mapping.py
hymeko_neuro/experiments/vision/hsikan_vision.py
hymeko_neuro/tests/test_cycle_cache.py             (9 new tests)
```

## E. Suggested commit bundle (you decide)

Three coherent commit groups:

```
# 1. 2026-06-03 reservoir + walks + Strategy refactor
git add hymeko_neuro/hyperedge/{reservoir,path_scorers,abb_walks,walks,n_tuples}.py \
        hymeko_neuro/graph/cycle_cache/{__init__,api,strategies,stats}.py \
        hymeko_neuro/tests/test_{cycle_cache,reservoir,path_scorers,abb_walks}.py \
        hymeko_graph/src/{lib,topk_walks,walks}.rs \
        hymeko_py/src/cycles/ hymeko_py/src/lib.rs \
        docs/plans/2026-06-03-*/
git commit -m "Reservoir + ABB walks + Strategy refactor"

# 2. 2026-06-04 hymeko_pgraph_py + KLP skyline
git add hymeko_pgraph_py/ Cargo.toml Cargo.lock \
        hymeko_neuro/hyperedge/abb_walks.py \
        hymeko_neuro/tests/test_{hymeko_pgraph_py,ssg_pareto_filter}.py \
        docs/plans/2026-06-04-{hymeko-pgraph-py,klp-skyline}/ \
        reports/2026-06-04-klp-skyline.md
git commit -m "hymeko_pgraph_py PyO3 wrapper + KLP O(N log N) skyline"

# 3. 2026-06-04 Komondor parallelism showcase + setup
git add docs/komondor_setup/ scripts/komondor_{morning_pull,parallelism_analysis,audit_metrics}.* \
        docs/plans/2026-06-04-komondor-parallelism-showcase/ \
        reports/2026-06-04-komondor-parallelism-{analysis,showcase}.* \
        reports/2026-06-04-msg-abb-ssg-unified-implementation.* \
        reports/2026-06-03-pimentel-benchmark-{reply,validation}.* \
        data/pgraph/Chapter6/pimentel_distractors.hymeko \
        hymeko_pgraph/{src/{abb,cli,dump,lib}.rs,src/bin/,tests/} \
        CLAUDE.md
git commit -m "Komondor parallelism + Pimentel benchmark + pgraph CLI/CR rendering"
```

(The exact groupings can flex; the principle is "one logical theme per commit".)

## F. Pending / parked

- Komondor K=20 jobid 13885808: results expected ~9.5 h after chain
  13885723 finishes (~7 h from now). Pull via
  `bash scripts/komondor_morning_pull.sh` in the morning.
- Komondor K=10 + K=5: chained after K=20; results during the day.
- Fuzzy-pose re-run (user said "esetleg majd"): H3 ceiling 0.555
  var_expl confirmed 2026-06-02; options for next attempt parked
  until user reactivates.

## G. Test inventory (today's additions)

- `hymeko_neuro/tests/test_hymeko_pgraph_py.py` — **12 tests**, GREEN.
- `hymeko_neuro/tests/test_ssg_pareto_filter.py` — **31 tests**, GREEN.
- `hymeko_pgraph/tests/pimentel_distractors.rs` — cargo test, GREEN
  (asserted MSG=7, SSG=19, top-3 costs (9,12,13)).
