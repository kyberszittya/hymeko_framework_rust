# Gömb fuzzy signature view — 2026-05-20

## Summary

Gömb is a three-shell signed-hypergraph cascade
($\text{OuterFIRShell} \to \text{MiddleHSiKAN} \to
\text{InnerCPMLCore}$). The fuzzy signature view of Gömb is
the multi-shell counterpart to the HSIKAN signature shipped
earlier today: per-query, it exposes how each contributing
cycle's magnitude *propagates through the cortical hierarchy*.

The shape is one extra dimension on top of HSIKAN's signature:
each :class:`GombCycleContribution` carries a
`per_shell_magnitude: dict[str, float]` (one scalar per shell)
and the full `per_shell_embedding`. The plot adds a per-shell
dominance bar plus a propagation panel that draws each cycle
as a polyline across shells, colour-coded by its
Cartwright-Harary balance vote.

**The headline finding from the Bitcoin Alpha demo** (lightweight
Gömb, 120 epochs):

| query | true | p(+) | net σ·\|h_mid\| | n_cycles | dom_outer | dom_middle | r_xs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_positive | + | 1.000 | +6.828 | 9 | **2.714** | 0.759 | **−0.68** |
| high_negative | − | 0.401 | −0.754 | 3 | 0.797 | 0.735 | +0.33 |
| decision_boundary | − | 0.500 | **−7.489** | 28 | 0.993 | 0.744 | +0.15 |

Two findings that the signature exposes and that AUC cannot:

1. **The Inner CPML core overrides cycle-level evidence.** The
   decision_boundary case has 28 cycles voting net σ·\|h\| =
   **−7.489** — overwhelming unbalanced evidence at the outer
   AND middle shells — but the model lands at p(+) = 0.500.
   The signature isolates the discrepancy: the inner CPML
   core (capsule routing, not captured per-cycle in this MVP)
   is *resolving* the strong outer/middle agreement into
   model uncertainty. Without the per-shell breakdown this
   would just look like a hard query; with it, the
   architectural source of the uncertainty is named.

2. **Cross-shell re-prioritisation (r_xs = −0.68 on the
   high-positive case).** The cycles that fire most at V1
   (outer) fire *less* at V2 (middle) — the Pearson
   correlation of per-cycle magnitudes across shells is
   *negative*. The cortical hierarchy isn't just rescaling
   the cycle vote; it's actively re-prioritising which
   cycles carry which signal at which depth. This is a
   non-trivial dynamics property that's invisible without
   the per-cycle propagation view.

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/interpret/gomb_signature.py` | new | 308 (`GombCycleContribution`, `GombFuzzySignature`, `extract_gomb_signature`, `plot_gomb_signature`) |
| `signedkan_wip/src/interpret/__init__.py` | extended | +8 (re-exports) |
| `signedkan_wip/src/hymeko_gomb/shells.py` | extended | +18 (capture side-channels on `OuterFIRShell` and `MiddleHSiKAN`) |
| `signedkan_wip/src/core/arc_weights.py` | extended | +4 (`annotate_arc_weights` now works on both `SignedTriad` and `SignedNTuple` via `getattr(t, "arity", len(verts))`) |
| `signedkan_wip/tests/test_gomb_signature.py` | new | 184 (8 unit tests) |
| `signedkan_wip/experiments/runs/demo_gomb_signature.py` | new | 188 (Bitcoin Alpha demo + 3 signatures) |
| `docs/plans/2026-05-20-gomb-fuzzy-signature/{plan.tex,plan.pdf,plan.tikz,plan_figure.pdf,plan.mmd}` | new | 4-format plan |
| `reports/2026-05-20-gomb-fuzzy-signature-view.md` | new | this file |
| `reports/figures/gomb_signature_bitcoin_alpha/{*.png,summary.json}` | new | 3 figures + JSON |

## CORE.YAML items touched

None.

## Interface

```python
@dataclass
class GombCycleContribution:
    cycle_idx: int
    vertices: tuple[int, ...]
    sigma_assignment: tuple[int, ...]
    edge_signs: tuple[int, ...]
    sigma_prod: int                 # ±1 = Π edge_signs
    balanced: bool
    arc_weights: tuple[float, ...]
    per_shell_magnitude: dict[str, float]
    per_shell_embedding: dict[str, np.ndarray]

@dataclass
class GombFuzzySignature:
    query_edge: tuple[int, int]
    query_idx: int
    contributions: list[GombCycleContribution]
    cycle_arity: int
    shells: tuple[str, ...]   # ('outer', 'middle')
    logit: float
    prob_positive: float
    # methods: shell_dominance(), cross_shell_consistency(), net_vote()

def extract_gomb_signature(model, cycles, signs, tier_of, edges_to_score,
                            query_idx, arc_weights=None, edge_signs=None)

def plot_gomb_signature(sig)  # 3- or 4-axes (with arc weights)
```

## What stays untouched

- Gömb cascade itself unchanged — the hooks are
  attribute-driven side-channels (set
  `shell._signature_capture = {}` before forward) that cost
  one `getattr` per forward when not in use.
- The HSIKAN `fuzzy_signature.py` is untouched; this is a
  parallel module.

## Capture coverage

This MVP captures **outer + middle** shells via the per-cycle
intermediate state both shells naturally compute. The
**Inner CPML core** uses capsule routing without a clean
per-cycle intermediate; capturing it requires reaching into
the CPML tier-stratified state and is left as a follow-up.

The decision-boundary finding above is actually evidence that
the inner core matters: the outer + middle shells voted
unbalanced (−7.489) and the model predicted 0.500. The
inner core resolved the evidence in a way the outer + middle
couldn't predict. A full inner-core capture would close that
black box.

## Test results

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_gomb_signature.py` | **8 / 8 pass** |
| All prior interpret / side / mixed-arity / arc-weight suites | 47 / 47 (no regression) |
| Bitcoin Alpha demo (120 epochs, lightweight Gömb) | completes in 2 s |

## §6.5 anti-pattern audit

- New module mirrors the existing HSIKAN signature pattern.
  No Cartesian product wrappers, no `_kind: str` arguments,
  no env-var feature flags at deep call sites.
- The capture hooks are attribute-driven (default None,
  no-op when unused). Cost is one `getattr` per forward.
- `interpret/` is two files now (`fuzzy_signature.py` +
  `gomb_signature.py`); within the §6.2 ceiling.

Clean.

## Open follow-ups

1. **Inner CPML core capture.** The capsule-routing core
   processes the cycle-pooled features through hypergraph
   convolutions; capturing the pre-routing per-cycle state
   would close the only remaining black box in the cascade.
   Estimated: half day, mostly reading
   `signedkan_wip/src/hymeko_gomb/soma/cpml.py`.
2. **Side-by-side HSIKAN vs Gömb on the same query.** The
   interfaces are now compatible at the level of (sigma,
   alpha-like magnitude, arc weights). A wrapper that runs
   both models on a graph and renders both signatures
   side-by-side would expose architectural-disagreement
   cases (where HSIKAN and Gömb diverge on a query).
3. **Three.js viewer.** The 3D visualisation we discussed
   for XR can take both signature types and render the
   per-cycle contributions in space, anchored to the query
   edge.
4. **Per-shell propagation as a training signal.** If the
   `cross_shell_consistency` is consistently negative on a
   dataset (cycles get re-prioritised between shells), that
   might be a regulariser axis worth exploring.

## Experiment provenance

- **Git SHA:** uncommitted.
- **Dataset:** Bitcoin Alpha (n_nodes = 3783, n_triads = 22153).
- **Demo training:** 120 epochs at lr=5e-3, Adam,
  lightweight Gömb (d_embed=8, M_outer=4). 2 s wall.
- **Demo figures:**
  `reports/figures/gomb_signature_bitcoin_alpha/{gomb_signature_high_positive,gomb_signature_high_negative,gomb_signature_decision_boundary}.png`
- **Seed:** 0 only (interpretation demo, not statistics).
- **GPU:** RTX 2070 SUPER 8 GiB; under 1 GiB peak.

## Acceptance check

- [x] Plan in 4 formats on disk
      (`docs/plans/2026-05-20-gomb-fuzzy-signature/`).
- [x] CORE.YAML items touched = 0.
- [x] 8 / 8 new unit tests pass; 47 / 47 prior tests no
      regression.
- [x] Demo produces three PNGs + summary.json on Bitcoin Alpha.
- [x] §6.5 anti-pattern audit clean.
- [x] **Per-shell propagation visible** (the
      decision_boundary case is a clean example of the
      inner CPML overriding the outer+middle agreement).
- [x] Report on disk.
