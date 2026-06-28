# HyMeKo machine-verified proofs — collected bundle

The proof companion to the T-SMC article *HyMeKo: A Canonical Hypergraph Intermediate Representation for Cross-View
Consistency in Cyber-Physical Model Generation*. Every one of the five propositions is machine-verified; this
bundle gathers the long write-up, the captured machine outputs, and pointers to the canonical scripts.

## PDFs (in this folder)

| file | pages | contents |
|---|---|---|
| **`hymeko_machine_verified_proofs.pdf`** | 25 | the master document: all five propositions (P1–P5), formal proofs, methods, verbatim machine outputs, the Z3 reduction, the storage regime, the dispatch-bug fix, and **full verbatim script listings** (Appendix A, read from the canonical sources at compile time so they cannot drift) |
| `proposition-verification.pdf` | — | the earlier P1–P4-only report (`reports/2026-06-28-proposition-verification.pdf`), kept here for the record; superseded by the master |

Rebuild the master: `pdflatex hymeko_machine_verified_proofs.tex` (run twice for the ToC). Requires
`listingsutf8` (MiKTeX). The figure and the `results/*.txt` machine outputs are read from `results/`.

## Proof scripts (canonical locations — NOT copied here, to avoid drift)

The master PDF embeds these verbatim via `\lstinputlisting`; edit them in place.

| proposition | script | method | result |
|---|---|---|---|
| P4 storage overhead | `verification/propositions/p4_storage_overhead.py` | sympy symbolic proof | proved |
| P2 content-address (collision) | `verification/propositions/p2_content_birthday.py` | sympy birthday bound | proved (+ assumption) |
| P1 alias invariance | `verification/propositions/p1_alias_invariance.py` | hypothesis vs. real compiler | hash 800/800; emit corrected |
| P3 proj.–emit independence | `verification/propositions/p3_emit_purity.py` | determinism vs. real emitters | 4/4 |
| P5 cross-view consistency | `verification/cross_view_consistency/cross_view.py` (+ `extract.py`) | extraction functions vs. real emitters | 16/16 exact, 16/16 topo |
| P5 entailment (logical) | `verification/cross_view_consistency/commute_z3.py` | z3 SMT reduction | T1 unsat, T2 sat |
| P5 second domain | `verification/cross_view_consistency/trace_witness.py` | SysML = DOT = Q | 4 req / 4 blocks / 7 edges |
| P4 regime reframe | `verification/cross_view_consistency/storage_regime.py` | sympy regime table | controlled, →1 high arity |

Tests: `verification/cross_view_consistency/tests/test_cross_view.py` (19 pass). Reports:
`reports/2026-06-28-cross-view-consistency.md`, `reports/2026-06-28-proposition-verification.md`. Plans:
`docs/plans/2026-06-28-cross-view-consistency/`.

## Captured machine outputs (`results/`)

`p1_alias_invariance.txt`, `p2_content_birthday.txt`, `p3_emit_purity.txt`, `p4_storage_overhead.txt`,
`cross_view.txt`, `commute_z3.txt`, `storage_regime.txt`, `trace_witness.txt`, and the figure
`cross_view_consistency.png`. Regenerate with `PYTHONIOENCODING=utf-8 python <script> > results/<name>.txt`
(then ASCII-transliterate for clean listings).

## Tooling (uv-pinned)

sympy 1.14, z3 4.16, hypothesis 6.15, matplotlib 3.11, numpy 2.4; the real CLI emitter
`target/release/hymeko.exe`.
