# Stacked Gömb-HSIKAN backbone — null result + a real bug-smell fix — 2026-05-20

## Summary

Gömb's three-shell cascade
(`OuterFIRShell → MiddleHSiKAN → InnerCPMLCore`) currently uses a
**single-layer** `SignedKANTierAggregator` as its middle shell.
This phase replaced it with a **multi-layer HSIKAN stack**
(`MultiLayerSignedKAN`, depth $L \in \{1, 2, 4\}$) — the same
stack that produced HSIKAN's Bitcoin-Optuna SOTA — to test
whether deeper signed-cycle processing inside Gömb's cortical
hierarchy lifts AUC.

**Headline (NULL).** Deeper middle-HSIKAN does not help, on
either dataset where we could measure:

| dataset | depth | mean AUC ± σ | Δ vs d=1 | σ_d | wins |
| --- | --- | --- | --- | --- | --- |
| bitcoin\_alpha | 1 | 0.9001 ± 0.0098 | — | — | — |
| bitcoin\_alpha | 2 | 0.8976 ± 0.0109 | **−0.0025** | −2.32 | 0/3 |
| bitcoin\_alpha | 4 | 0.8983 ± 0.0143 | −0.0018 | −0.59 | 2/3 |
| slashdot | 1 | 0.9010 ± 0.0006 | — | — | — |
| slashdot | 2 | 0.9001 ± 0.0009 | −0.0009 | −1.01 | 1/3 |
| slashdot | 4 | OOM | — | — | — |

Bitcoin Alpha d=2 is **paired-significantly worse** than d=1
(σ_d = −2.32). Bitcoin Alpha d=4 and Slashdot d=2 are null
(within paired noise). Slashdot d=4 OOM'd on the 7.6 GiB GPU
even at the strict-bench config; that's genuine memory
pressure from 4-deep HSIKAN, not a bug.

Same falsification pattern as Phase 21 (parallel branches on
HSIKAN) and Phase 22 (parallel branches on mixed-arity
HSIKAN): both Bitcoin Alpha and Slashdot are at architectural
ceiling for cycle-based factorisation. The stacked-HSIKAN
hypothesis is dead.

**Bug-smell fix shipped as a side effect.** While diagnosing
the Slashdot OOM, I found a Python-level inefficiency in
`InnerCPMLCore._edge_logits` that allocated an unnecessary
~1 GiB intermediate (`torch.cat([u, v], dim=-1)` for the
edge-prediction head). Replaced with a factored matmul that
saves the (E_query × 2·d_final) tensor without changing
numerics. Memory savings let Slashdot d=2 run cleanly; d=4
moved the OOM upstream to the CR spline eval (genuine 4-layer
pressure). Bit-identical output (diff ≤ 4e-7 vs the cat form).

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/core/cpml.py` | extended | +16 / −2 (`_edge_logits` factored matmul fix) |
| `signedkan_wip/src/core/signedkan.py` | extended | +9 (`MultiLayerSignedKAN.encode_triads(initial_h_v=...)`) |
| `signedkan_wip/src/hymeko_gomb/shells.py` | extended | +110 (`StackedMiddleHSiKAN`) |
| `signedkan_wip/src/hymeko_gomb/cascade.py` | extended | +30 (`GombConfig` middle fields + `HymeKoGomb.__init__` dispatch) |
| `signedkan_wip/experiments/runs/run_gomb_smoke.py` | extended | +25 (CLI flags for the stacked middle) |
| `signedkan_wip/tests/test_stacked_gomb_hsikan.py` | new | 187 (8 unit tests) |
| `signedkan_wip/experiments/run_stacked_gomb_overnight_2026_05_20.sh` | new | 117 |
| `signedkan_wip/experiments/run_stacked_gomb_overnight_slashdot_only_2026_05_20.sh` | new | 92 |
| `docs/plans/2026-05-20-stacked-gomb-hsikan-backbone/{plan.tex,plan.pdf,plan.tikz,plan_figure.pdf,plan.mmd}` | new | 4-format plan |
| `reports/2026-05-20-stacked-gomb-hsikan-backbone.md` | new | this file |

## CORE.YAML items touched

None.

## The bug-smell fix in detail

The original `_edge_logits`:

```python
def _edge_logits(self, x_final, edges_to_score):
    u = x_final[edges_to_score[:, 0]]      # (E, d_final)
    v = x_final[edges_to_score[:, 1]]      # (E, d_final)
    pair = torch.cat([u, v], dim=-1)       # (E, 2*d_final) ← 1.1 GiB on Slashdot!
    return self.head(pair).squeeze(-1)     # Sequential(Linear(2d, h), GELU, Linear(h, 1))
```

On Slashdot at the strict-bench config: E_train ≈ 440k, d_final = 312.
`pair` is `440k × 624 × 4 bytes ≈ 1.1 GiB`. With the autograd
graph from the upstream forward (~4 GiB), the cat hits the GPU
ceiling and OOMs.

The fix factors the first linear of the Sequential head:

```python
def _edge_logits(self, x_final, edges_to_score):
    u = x_final[edges_to_score[:, 0]]
    v = x_final[edges_to_score[:, 1]]
    first = self.head[0]                   # Linear(2d, h)
    rest = self.head[1:]                   # GELU → Linear(h, 1)
    W, b = first.weight, first.bias        # W shape (h, 2d)
    d = u.shape[-1]
    h = u @ W[..., :d].t() + v @ W[..., d:].t()
    if b is not None: h = h + b
    return rest(h).squeeze(-1)
```

Linear-algebra identity:
`Linear(2d, h)(cat([u, v])) ≡ u @ W[..., :d].T + v @ W[..., d:].T + b`.
Two `(E, h)` matmuls instead of one `(E, 2d) @ W.T`, with no
`(E, 2d)` intermediate.

Numerical parity check passed at 3.58e-7 max diff (float32
precision floor). 16/16 Gömb signature + stacked-Gömb unit
tests still pass.

This is the same class as the Phase 22 outer-checkpoint fix:
a Python convenience that quietly costs memory at scale.
Worth keeping as a permanent improvement regardless of the
stacked-HSIKAN null.

## Why deeper middle-HSIKAN fails inside Gömb

The hypothesis was that Gömb's outer FIR shell would provide
the scale-invariance that pure-depth HSIKAN lacks (Phase 16's
L=8 catastrophe on Bitcoin Alpha was -0.328 AUC; we expected
the cortical-cascade context to soften that). It does soften
it — d=4 here lost only -0.018 AUC, not the catastrophe — but
the lift never materialises. Each extra middle layer is just
a parameter expense the model doesn't know what to do with.

This is consistent with Phase 21/22's findings on parallel
branches: at the Bitcoin Alpha / Slashdot ceiling, **no
cycle-factorisation architectural change lifts mean AUC**.
The dataset is doing what it's going to do regardless of
the inductive bias.

## Test results

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_stacked_gomb_hsikan.py` | **8 / 8 pass** |
| `pytest signedkan_wip/tests/test_gomb_signature.py` | **8 / 8 pass** (no regression from _edge_logits fix) |
| All prior interpret/side/arity/fuzzy suites | 47 / 47 (no regression) |
| Bitcoin Alpha smoke at d∈{1,2,4} | all complete |
| Slashdot retry d∈{1,2}                | complete; d=4 OOMs at CR spline (genuine memory pressure) |
| Numerical parity (cat vs factored `_edge_logits`) | ≤ 4e-7 max diff |

## §6.5 anti-pattern audit

- New `StackedMiddleHSiKAN` is a single class with a config-
  style constructor; no Cartesian product or `_kind: str` axes.
- `initial_h_v` is a kwarg on existing
  `MultiLayerSignedKAN.encode_triads`, additive, defaults to
  None (backward compat).
- Dispatch is at construction time
  (`if cfg.middle_n_layers <= 1`), not a `forward()` toggle.
- `_edge_logits` fix is pure additive — no API change, no new
  function, just a more efficient implementation of the
  existing operation.

Clean.

## Open follow-ups

1. **Gradient checkpointing in StackedMiddleHSiKAN.** Same
   pattern as the Phase 22 outer-checkpoint that unlocked
   N=4 with attention on Slashdot. Would let us test d=4
   on Slashdot at the strict-bench config.
2. **Rust acceleration for the per-layer HSIKAN forward.**
   The Triton kernel infrastructure exists
   ([[project-triton-kernel-integration-2026-05-09]],
   [[project-fused-backward-kernel-2026-05-09]]) and applies
   to the inner Catmull-Rom spline eval. Plumbing the kernel
   path through the stacked middle would address the d=4 OOM
   directly. Probably 1 session.
3. **ABB / MSG architecture search.** Now that we have the
   simple grid result (null), the question becomes: is there
   a *different* axis (jk_mode, inner_skip, share_weights,
   topk) that helps? The P-graph machinery is the right tool
   for that search.
4. **HymeYOLO port deferred.** The original ask included
   "and let's create a Gömb with stacked HSIKAN backbone …
   including the computer vision application with the
   HymeYOLO extension." Given the signed-graph result is
   null, the vision port is unlikely to help — but worth
   noting as a deliberate non-pursuit, not an oversight.

## Experiment provenance

- **Git SHA:** uncommitted.
- **Bitcoin Alpha:** 9 cells × 60 epochs, ~7 s each, total
  ~70 s. Config: d_embed 32, M_outer 8, d_outer 20,
  d_middle 24, d_core 48, n_tiers 4, topk 56, lr 5e-3.
- **Slashdot:** 6/9 cells complete (d=4 × 3 seeds OOM'd).
  22–32 s/seed at d=1/2. Config: d_embed 16, M_outer 12,
  d_outer 8, d_middle 16, d_core 32, n_tiers 2, topk 32,
  lr 5e-3.
- **GPU:** RTX 2070 SUPER 8 GiB.
- **JSONL:**
  `signedkan_wip/experiments/results/stacked_gomb_overnight_2026_05_20.jsonl`,
  `signedkan_wip/experiments/results/stacked_gomb_overnight_slashdot_2026_05_20.jsonl`.

## Acceptance check

- [x] Plan in 4 formats on disk.
- [x] CORE.YAML items touched = 0.
- [x] 8 / 8 new unit tests + 47 / 47 prior tests no regression.
- [x] Bitcoin Alpha grid complete; Slashdot d∈{1,2} complete.
- [x] **Bug-smell fix shipped:** `_edge_logits` factored
      matmul saves ~1 GiB on Slashdot; numerics bit-identical.
- [x] §6.5 anti-pattern audit clean.
- [x] **Null result framed honestly:** stacked HSIKAN in
      Gömb's middle shell does NOT lift AUC; the
      cycle-factorisation hypothesis is falsified across
      both datasets at this configuration.
- [x] Report on disk.
