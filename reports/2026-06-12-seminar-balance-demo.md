# Seminar demo program — plan-item 5: `balance` demo (affective structural balance)

**Date:** 2026-06-12 · **Plan:** `docs/plans/2026-06-10-seminar-demos-remaining/` §Item 5
· **Spec:** `hymeko_neuro/demos/SEMINAR_DEMOS.md` §1

## Summary
Added the `balance` demo — the seminar opener (slide 12). Loads a signed
(affective) graph, enumerates its short cycles via the Rust
`enumerate_top_k_cycles_rs` path, classifies each cycle by **sign-product**, and
reports **structural balance** = fraction of cycles with positive sign-product;
frustration = 1 − balance. The most-frustrated cycles are surfaced and drawn.
Runs as `python -m hymeko_neuro.demos.seminar balance [--graph planted|camps|karate]`.

The per-cycle sign-product is a Python reduction over the Rust-enumerated pool
(CLAUDE.md §6.5 #2: enumeration is the Rust algorithm; the reduction is trivial
and stays Python). Balance is classified from the **product**, never from the
`fraction_negative` score — a 2-negative-edge cycle is balanced despite a high
fraction_negative. The Rust scorer only ranks which cycles survive the `k_keep`
cap; a binding cap is detected and reported as a lower-bound caveat (§2 contract).

## Files touched
| LOC | File | |
|---:|---|---|
| 246 | `hymeko_neuro/demos/seminar/demos/balance.py` | new — `BalanceDemo`, `balance_statistics`, `cycle_sign_products`, fixtures |
| +60 | `hymeko_neuro/experiments/demo/plotting.py` | new `frustration_figure` + 3 private draw/prune helpers |
| 152 | `hymeko_neuro/tests/test_seminar_balance.py` | new — 13 tests |
| +2 | `hymeko_neuro/demos/seminar/demos/__init__.py` | register `BalanceDemo` |
| ±6 | `hymeko_neuro/demos/PRESENTER_RUNBOOK.md` | balance + latency → READY |

## CORE.YAML items touched
**None.** `src/demo/plotting.py`, `datasets`, and `demos/seminar` are all
non-core (verified against CORE.YAML). Additive demo + a reusable figure.

## Test results
Runner: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest -p no:randomly`.

| Layer | Tests | Result | Notes |
|---|---|---|---|
| Unit — `cycle_sign_products` | 4 | pass | balanced=+1, one-neg=−1, **two-neg=+1** (proves product ≠ score), non-edge→KeyError |
| Unit — fixtures | 2 | pass | camps balanced by construction; planted flips one edge, names {0,1,2} |
| Figure (no engine) | 1 | pass | `frustration_figure` renders |
| Integration (`DemoRunner`, engine) | 6 | pass | registration, camps→1.0, planted→0.5 + triad surfaced, run+figure+provenance, `--no-figures`, `--k<3` raises |
| **Total** | **13** | **pass** | 18 s |
| Regression (harness 13 · latency 8) | 21 | pass | registry change did not break the other demos |

Coverage (§3): every new public/private function is exercised; both failure
branches (non-edge cycle, `--k < 3`) have dedicated tests; the regression test
`test_sign_product_two_negatives_is_balanced` would fail against a
score-based (wrong) classifier.

## Performance results
End-to-end CPU, seed 0:

| Graph | balance | n_cycles | Peak RSS | Wall |
|---|---|---|---|---|
| `planted` | 0.5 (1/2 frustrated, triad {0,1,2}) | 2 | 761 MB | 11.1 s (cold hymeko import) |
| `camps` | 1.0 (0 frustrated) | 2 | 761 MB | 5.5 s |
| `karate` | 1.0 (0 frustrated) | 45 | 760 MB | 5.8 s |

Well under the 16 GB cap. Wall dominated by the one-time `hymeko` import; the
enumeration + reduction is sub-second at these sizes.

## Acceptance gate
Plan §5 / SEMINAR_DEMOS §1: hand-built balanced graph → 1.0 & empty frustrated
set; planted-unbalanced → < 1.0 & the planted cycle surfaced. **PASS** —
`camps` 1.0 / 0 frustrated; `planted` 0.5 with triad {0,1,2} in the surfaced
list (`planted_cycle_surfaced=True`). Karate (real faction signing) is
Harary-balanced by construction → 1.0 over 45 triangles, used as the honest
real-world aside.

## Honest framing (printed by the demo)
- balance/frustration is the **cycle sign-product statistic, not a learned
  quantity** — it sets up the `link` demo (same cycles drive prediction);
- a binding `k_keep` cap makes the reported balance a **lower bound** over the
  kept (most-frustrated) pool — surfaced as a WARNING note, never hidden.

## Static analysis
- `ruff check`: clean (all changed files).
- `mypy --strict` (balance.py): `Success: no issues found`. One scoped
  `# type: ignore[import-untyped]` on `import hymeko` (compiled PyO3 module, no
  py.typed marker) with an inline reason.
- `mypy --strict` (plotting.py): the module carries **pre-existing** findings
  (no `networkx` stubs; the module-wide `"plt.Figure"` forward-ref idiom used by
  `roc_figure`/`alpha_figure`/`subgraph_figure`). This change adds **no new
  error class** — the two new draw helpers are annotated (`no-untyped-def`
  cleared); `frustration_figure` follows the existing return idiom.
- `radon cc -nc`: `frustration_figure` was refactored (extracted
  `_cycle_elements`, `_prune_for_legibility`, `_draw_signed_edges`,
  `_draw_nodes`) to land below the C threshold. The only D-rated block in
  plotting.py is the **pre-existing** `subgraph_figure` (untouched here).
- No §6.5 anti-patterns: one `SeminarDemo` in the dict (#1/#13); enumeration not
  re-implemented (#2); no globals (#11); enum-like `--graph` choices, not a
  free-form string into Rust (#7); files < 250 LOC (#4).

## Provenance
- Git SHA `af803ee` (working tree dirty — pre-existing untracked seminar work +
  the files above). Python 3.12.13; `hymeko` 0.1.0 editable (maturin).
- Seed 0 (fixed); deterministic — fixtures hand-built, `karate_faction_signed`
  is deterministic (NetworkX Zachary), enumeration is seeded.
- Artifacts: `demo_out/balance/frustration_{planted,camps,karate}.png`.

## Open issues / follow-ups
- Remaining seminar build items: **mesh + Sinkhorn** (item 6 — the only
  genuinely new algorithm; discovery-grep `sinkhorn|optimal.transport` first per
  §6.1), then **bridge** (item 7, closer).
- `karate` balance = 1.0 is correct (faction signing is balanced by
  construction). If a real-data frustration example is wanted on stage, an SBM
  with `noise > 0` (`datasets.sbm_signed`) plants genuine frustration — a
  one-line `--graph sbm` addition, deferred until asked.
