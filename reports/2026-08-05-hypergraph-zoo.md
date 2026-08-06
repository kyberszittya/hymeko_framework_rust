# Hypergraph zoo — canonical constructions as verifiable HyMeKo / Nagare benchmarks

**Date:** 2026-08-05
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `b78cf55f`)
**In response to:** the proposed hypergraph catalog — building the first increment of a "hypergraph zoo".

---

## Summary

A HyMeKo model **is** a (signed-incidence) hypergraph, so a catalogue of canonical hypergraphs gives principled
benchmarks for the signed / holonomy / cycle-consistency line, for constraint & matroid structure, and for
hypergraph spectral / neural methods. `scenarios/hypergraph_zoo.py` provides generators for the flagged families,
each returning a `Hypergraph` whose **defining combinatorial property is checkable** (and pinned by the tests) —
not just "a hypergraph was produced".

## Families (all verified)

| generator | defining property (asserted) |
|---|---|
| `fano_plane` / `projective_plane(q)` | `PG(2,q)`: `q²+q+1` points & lines, `(q+1)`-uniform, `(q+1)`-regular, linear, `2-(v,q+1,1)`; Fano also self-dual |
| `affine_plane(q)` | `AG(2,q) = S(2,q,q²)` |
| `steiner_triple_system(v)` | `S(2,3,v)`: every vertex pair in exactly one triple (verified for `v = 7,9,13,15,19,21`) |
| `complete_uniform(n,k)` | all `C(n,k)` `k`-subsets |
| `kneser(n,k,r)` | disjoint-`k`-subset edges (`KG_2(5,2)` = the Petersen hypergraph, 10 v / 15 e) |
| `loose_cycle` / `tight_cycle` | consecutive edges share exactly `1` / `k−1` vertices |
| `sunflower` | every two edges meet in a constant core (Δ-system) |
| `random_uniform` | `H^{(k)}(n,p)`, seeded & reproducible |
| `graphic_matroid_circuits` | the simple cycles = minimal dependent edge sets (`K₄` → 7: 4 triangles + 3 four-cycles) |
| `simplex_boundary` | the `(n−1)`-facets of the simplex |

The `Hypergraph` type carries the operations these benchmarks need: incidence matrix, uniformity / regularity,
linearity, pair-coverage & `2-design` test, and the **dual** / **self-duality** (the incidence transpose).

## Why these, for HyMeKo–Nagare

- **Fano / projective planes** — minimal and scalable *regular, linear, signed / holonomy* examples (the thesis's
  basic illustrative hypergraph); self-duality is a clean symmetry to test signed-consistency methods on.
- **Steiner systems** — global order from a *local covering rule*: exactly the local→global consistency question.
- **Loose / tight cycles** — feedback / temporal / holonomy loops (the Nagare cycle structure), with two
  inequivalent overlap regimes.
- **Matroid-circuit hypergraphs** — *minimal dependencies*: "which constraints together are dependent?" — the
  robotics constraint-schema view, in one formalism.
- **Sunflowers** — a shared interface/core with disjoint variants (redundant rule families / option branches).
- **Random `H^{(k)}(n,p)`** — phase transitions & robustness, the discrete analogue of the Nagare depth×data phase
  diagrams.

## Files touched

| File | LOC | notes |
|---|---|---|
| `scenarios/hypergraph_zoo.py` | +250 (new) | `Hypergraph` + 12 canonical generators + property methods |
| `tests/test_hypergraph_zoo.py` | +100 (new) | 12 tests, each asserting a family's defining property |
| `reports/2026-08-05-hypergraph-zoo.md` | new | this report |

## CORE.YAML items touched
None. numpy-only (+ stdlib `itertools`/`random`); no dependency change.

## Test results
- `pytest tests/test_hypergraph_zoo.py -p no:randomly` → **12 passed in 0.09 s**; `ruff check` → clean.

## Honest scope
- **STS** for larger admissible `v` uses a most-constrained-first randomised greedy (verified to close for
  `v ≤ 21`); a designed Bose/Skolem construction would guarantee closure for all admissible `v` (flagged).
- **Projective/affine planes** are built for **prime** `q` (Fermat inverse in `GF(q)`); prime-power `q` (e.g. 4, 8,
  9) needs the field arithmetic of `GF(q)` — flagged, not implemented.
- **Matroid circuits** enumerate simple cycles by edge-subset scan (capped at size 10) — exact for small
  benchmark graphs, exponential in general.

## Open issues / follow-up
- **Emit each zoo hypergraph as a HyMeKo model** (signed-incidence) and round-trip through `hymeko validate` /
  `hymeko_pgraph` — connecting the zoo to the native sparse-tensor path.
- **Spectra & signed variants**: incidence/Laplacian spectra, signed edges, and the holonomy of the cycle
  families — the actual Nagare experiments the zoo is meant to feed.
- **Designed STS** (Bose/Skolem) and **prime-power planes** for completeness.

## Provenance
Git SHA at start `b78cf55f`. Env: HyMeKo `.venv` (Python 3.11, NumPy 2), macOS (darwin 25.5). Deterministic
(seeded random hypergraph + STS). No GPU, no dataset.
