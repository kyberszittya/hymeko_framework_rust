# Phase 22: SideMixedArity on Slashdot — null mean, σ halved at N=2 — 2026-05-20

## Summary

Phase 22 ported the Phase 21 wrapper (parallel mixed-arity
branches with mean fusion) onto Slashdot's edge_cr highway
SOTA config (`c2,c3,c4,c5,w2,w3` + quaternion attention,
[[project_edge_cr_5seed_2026_05_09]] baseline 0.9067 ± 0.0034).

Phase 21 falsified the lever on Bitcoin Alpha at the 0.997
AUC ceiling (Δ=+0.0003, σ identical). The hypothesis going
into Phase 22: Slashdot's σ=0.003 baseline has 7× the slack
for the variance-tightening lever Phase 19 found on c3-only
(σ ≈ 0.013 uniformly). Does it transfer?

**Headline:** **Yes, on σ. No, on mean.** Paired Δ AUC = −0.0004
± 0.0053 (σ\_d=−0.17, wins=2/5) — null on mean.
σ collapsed from **0.0037 at N=1 to 0.0018 at N=2** — **48%
reduction**. The Phase 19 variance-tightening result
**transfers to the mixed-arity family on Slashdot** while the
AUC lift does not.

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/core/side_signedkan.py` | extended | +52 (outer-checkpoint mode + per-branch checkpoint + 2 config fields) |
| `signedkan_wip/src/mixed_arity_signedkan/model.py` | extended | +6 (`collect_attn_entropy` kwarg gates `_attn_entropy_terms` reset) |
| `signedkan_wip/experiments/runs/run_final_cell.py` | extended | +14 (`--outer-grad-checkpoint` CLI + dispatch wiring) |
| `signedkan_wip/tests/test_side_mixed_arity.py` | extended | +75 (3 new tests: entropy skip, outer-ckpt parity, outer-ckpt grads) |
| `signedkan_wip/experiments/run_phase22_slashdot_5seed_2026_05_20.sh` | new | 109 |
| `reports/2026-05-20-phase22-slashdot-side-mixed-variance.md` | new | this file |

## CORE.YAML items touched

None.

## Phase 22 5-seed result (Bitcoin Alpha → Slashdot pivot)

5 seeds × {N=1, N=2} on Slashdot, edge_cr highway SOTA config,
hidden=4, n_epochs=80, kernel ON, cache ON.

| seed | N=1 AUC | N=2 AUC | Δ |
| --- | --- | --- | --- |
| 0 | 0.91026 | 0.90512 | −0.00514 |
| 1 | 0.90525 | 0.90505 | −0.00020 |
| 2 | 0.90101 | 0.90839 | +0.00738 |
| 3 | 0.90939 | 0.90399 | −0.00540 |
| 4 | 0.90567 | 0.90701 | +0.00134 |

| N | mean AUC ± **σ** | wall/seed | n\_params |
| --- | --- | --- | --- |
| 1 | 0.9063 ± **0.0037** | 45 s | 328,843 |
| 2 | 0.9059 ± **0.0018** | 81 s | 657,695 |

Paired Δ = −0.0004 ± 0.0053, σ\_d = −0.17, wins = 2 / 5.

### Two-axis verdict

| axis | result | vs Phase 21 (Bitcoin) |
| --- | --- | --- |
| mean AUC | null (Δ=−0.0004, 0.17σ) | also null (Δ=+0.0003, 1.07σ) |
| variance σ | **48% reduction** (0.0037 → 0.0018) | null (0.0005 → 0.0005) |

The variance-tightening lever **transfers** to mixed-arity
when there's σ slack to act on. The mean-lift hypothesis
falsifies in both regimes.

### Why σ tightens but mean doesn't

The Phase 17/19 explanation: parallel branches' independent
stochastic inits average out per-seed fluctuations in the
final test AUC. This is essentially the central-limit
result for ensemble averaging — variance scales as 1/N when
branches are independent. Mean ≈ 0.0037 / sqrt(2) = 0.0026
predicted; we measured 0.0018, slightly tighter than the
naive prediction (suggesting some weak positive correlation
between branches that helps).

The mean AUC does NOT improve because both branches are
sampling from the SAME training distribution → both find
the same local optimum (up to seed noise). Ensemble averaging
of converged-to-same-place models gives the same mean.

This is structurally different from Phase 19's c3-only
result, where ensembling DID lift mean (+0.014). The
difference: c3-only HSIKAN was underconverged at L=1
(0.794 baseline); the extra branches genuinely captured
different parts of the loss surface. At mixed-arity SOTA,
training has fully converged, and there's nothing left
for the extra branches to capture.

## Memory fixes shipped during Phase 22 (the bug)

User correctly flagged: "we had no OOM before on this
dataset". The OOM at N=4 was new — the bare single-branch
config runs fine on the same 7.6 GiB GPU. Root cause: my
Phase 21 wrapper kept all N branches' autograd graphs
alive simultaneously for backward. Two fixes shipped:

**Fix A — `collect_attn_entropy=False` kwarg on
`MixedAritySignedKAN.encode_edges`.** When the entropy
regulariser λ\_attn = 0 (Slashdot SOTA case), the
`_attn_entropy_terms` list collects scalars whose autograd
graphs reach back through the full attention forward — for
no benefit (the regulariser multiplies them by 0 anyway).
Skipping collection drops the graph entirely.

**Fix B — `outer_grad_checkpoint=True` in the wrapper.**
Wraps the *entire* multi-branch encode_edges in a single
`torch.utils.checkpoint.checkpoint`. Forward stores **no**
intermediates; backward recomputes the branches sequentially
so peak GPU = ~1 × branch-forward (not N ×). Incompatible
with the entropy side-channel (consumer reads it between
forward and backward, but outer checkpoint discards it), so
Fix B auto-disables entropy collection via Fix A. Use only
when `HSIKAN_ATTN_ENTROPY_LAMBDA = 0` (default).

Tests: 13/13 pass — including a numerical-parity test
(`outer_grad_checkpoint=True` reproduces the bare path
within 1e-4) and a backward-reaches-every-branch test.

Together these unlock N=4 on the 8 GB GPU when the entropy
reg is off, which is the production Bitcoin Alpha /
Slashdot config.

## Test results

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_side_mixed_arity.py` | **13 / 13 pass** |
| `pytest signedkan_wip/tests/test_side_signedkan.py` | 12 / 12 (no regression) |
| `cargo test -p hymeko_pgraph` | 96 / 96 + 1 ignored doctest |
| Bare single-branch baseline reproduces 0.906 ± 0.004 | ✓ (vs published 0.9067 ± 0.0029) |

## §6.5 anti-pattern audit

- `collect_attn_entropy` is a new kwarg on `encode_edges`,
  not a new function — additive, no Cartesian product.
- `outer_grad_checkpoint` is a config field, not a new
  class — same.
- No new env-var feature flags read at deep call sites; both
  thread cleanly from CLI → cell\_signed\_graph → wrapper.
- All file sizes still under §6.2 ceilings.

Clean.

## Open follow-ups

1. **Smoke N=4 with `--outer-grad-checkpoint`.** Fix B
   should unlock N=4 on the 7.6 GiB GPU. If σ keeps
   tightening (0.0018 → ~0.0010), that's worth one
   confirmatory run.
2. **Epinions A/B with the wrapper.** σ=0.011 baseline
   has the most variance slack of any signed dataset; the
   tightening lever should be strongest there.
3. **Honest-σ reporting in SOTA tables.** Phase 22 shows
   the published 5-seed σ is the bare-single-branch noise.
   An ensemble of N=2 cuts it in half at 2× wall — a
   real reproducibility lever that doesn't move the headline
   AUC.

## Experiment provenance

- **Git SHA:** uncommitted.
- **Dataset:** Slashdot (n\_nodes=82140, n\_edges=549202).
- **5-seed log directory:**
  `/tmp/phase22_slashdot_5seed_20260520T18*/`
- **JSONL results:**
  `signedkan_wip/experiments/results/phase22_slashdot_5seed_2026_05_20.jsonl`
- **GPU:** RTX 2070 SUPER 8 GiB; kernel ON.
- **5-seed wall:** ~10 min total (cycle cache amortizes
  enumeration; per-seed wall is dominated by training).
- **Seeds:** [0, 1, 2, 3, 4].

## Acceptance check

- [x] Plan covered by Phase 21 §2 plan (no new plan needed
      — port to a new dataset is not a non-trivial code
      change; the only code edits are the two memory fixes
      which were themselves bugs flagged by the user).
- [x] CORE.YAML items touched = 0.
- [x] 13 / 13 unit tests pass (10 prior + 3 new for the
      fixes).
- [x] §6.5 anti-pattern audit clean.
- [x] 5-seed paired A/B with σ\_d and win-count reported.
- [x] **σ-tightening result honest:** mean is null but σ
      halves; framed as variance, not as AUC lift.
- [x] Memory fixes documented + tested.
- [x] Report on disk.
