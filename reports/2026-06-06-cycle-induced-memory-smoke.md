# Cycle-Induced Memory — Diagnostic Smoke (past vs symmetric on IMDB)

**Date:** 2026-06-06 (evening, post-Friedler-plan)
**Plan:** `docs/plans/2026-06-06-cycle-induced-memory/plan.md`
**Status:** diagnostic smoke complete; result matches the plan's
prediction for IMDB (local context). The discriminating test belongs
on a long-range task and is out of scope of tonight's smoke.

## Summary

Added `candidate_temporality ∈ {symmetric, past, future}` to the AC-HSiKAN
config + index plumbing, with `past` being the cycle-induced-memory variant
(each anchor sees only $\{t-1, \dots, t-K\}$, clamped at the left boundary).
Cycle walk-op closes back to the anchor at time $t$, so the sign-product
carries an algebraic dependence on past positions — memory without state
forwarding, attention, or SSM kernels.

Ran a 2-seed paired IMDB smoke at the same protocol as the cycle-walk-op
baseline (the +0.017 number cited in the Friedler-quotient plan):
`d=16`, `walk_kind=cycle`, `+rotor`, `5k/2k`, `4 ep`, `L_max=200`,
`lr=3e-3`.

| variant   | seed 0 | seed 1 | mean    | σ       | wall (2-seed total) |
|-----------|-------:|-------:|--------:|--------:|--------------------:|
| symmetric | 0.7555 | 0.7785 | **0.7670** | 0.0163 | 109.2 s |
| past      | 0.7650 | 0.7685 | **0.7668** | **0.0025** | 108.5 s |

**Paired Δ (past − symmetric) ≈ 0** (mean −0.0002; per-seed: +0.0095, −0.0100).

This is the diagnostic-baseline outcome the plan predicted:

> *"IMDB favours symmetric (local context); the past-only variant is roughly
> neutral or slightly worse on this specific task."*

— `docs/plans/2026-06-06-cycle-induced-memory/plan.md` §"Concrete first
experiment".

The smoke is a *gating check on implementation correctness*, not the
memory-mechanism test. The latter requires a long-range task (LRA Text /
ListOps, character-level LM, or selective copy) where local context is
known to be insufficient. Listed in the plan §Phase 1.

## Secondary observation (free finding)

Past collapses the seed σ **6.5×** (0.0163 → 0.0025), at identical wall
(108.5 vs 109.2 s) and identical parameter count (164 668). Mechanism:
the symmetric variant flips which K/2 slots are positive vs negative
offsets per anchor; past nails the temporal direction. Same anchor sees
the same neighbour structure every forward → less seed-time noise.

This is a free side-benefit of causality — independent of whether the
walk-op cycle actually exploits memory. Useful for any task where
seed-stability matters more than absolute accuracy. **Not** promoted to
a paper claim until 5-seed replicates: the 2-seed σ may itself be noisy.

## Files touched

- `signedkan_wip/src/ac_hsikan/config.py`
  - Added `candidate_temporality: str = "symmetric"` field.
  - Added validation in `__post_init__` for `{symmetric,past,future}`.
- `signedkan_wip/src/ac_hsikan/layer.py`
  - `_local_indices(L, K, device, temporality=…)` returns past / future /
    symmetric offset windows; boundaries clamp to anchor (sentinel for
    "no past available at small $t$").
  - `_hybrid_indices(…)` forwards temporality through to `_local_indices`.
  - Buffer cache init and ragged-L fallback in `forward` pass
    `cfg.candidate_temporality`.
- `signedkan_wip/experiments/ac_hsikan_imdb_smoke.py`
  - `--candidate-temporality {symmetric,past,future}` CLI flag.
  - Threads through to `AcHsikanConfig`.

No core change. No new dependencies.

## CORE.YAML items touched

None.

## Test results

- Unit-level check on `_local_indices`:
  - Anchor 5, K=4, symmetric → `[3, 4, 6, 7]`
  - Anchor 5, K=4, past → `[1, 2, 3, 4]` (strictly causal)
  - Anchor 5, K=4, future → `[6, 7, 8, 9]` (strictly anti-causal)
  - Anchor 0, past → `[0, 0, 0, 0]` (clamped — sentinel)
  - Anchor 9, future → `[9, 9, 9, 9]` (clamped — sentinel)
- Config validation:
  - `candidate_temporality='backward'` → `ValueError` as expected.
- Smoke runs:
  - `symmetric.log` — 2 seeds × 4 ep clean, val_acc 0.7670 ± 0.0163.
  - `past.log` — 2 seeds × 4 ep clean, val_acc 0.7668 ± 0.0025.

Both runs ended cleanly with exit code 0. No GPU OOM, no NaNs, no
assert violations.

## Performance results

| metric | symmetric | past | Δ |
|---|---:|---:|---:|
| 2-seed wall total | 109.2 s | 108.5 s | −0.7 s |
| n_params | 164 668 | 164 668 | 0 |
| val_acc mean | 0.7670 | 0.7668 | −0.0002 |
| val_acc σ | 0.0163 | 0.0025 | **−0.0138 (−85 %)** |

Past has zero runtime cost (the offset construction is `O(K)` per layer
init, cached after the first forward) and zero parameter cost.

## Wall-time + pruning follow-on (same session)

After the diagnostic, swept wall-time and parameter-pruning levers on the
same cycle+rotor protocol (1 seed unless noted):

| config | n_params | wall | val_acc |
|---|---:|---:|---:|
| baseline (cycle + rotor) | 164,668 | 54.9 s | 0.7555 |
| + `--fused-walk --sparse-sign-head` | 164,602 | 12.2 s | 0.7670 |
| + `--top-k 4` | 164,602 | 10.5 s | 0.7660 |
| + `--vocab-size 5000` | 84,602 | 11.1 s | 0.7290 |
| `--d-model 8 --top-k 4` (2-seed) | 81,674 | 10.9 s | 0.7125 ± 0.0042 |
| + `--compile` (warm-up loss) | 164,602 | 75.9 s | 0.7655 |

Findings:

1. **Wall reduced 5.2×** with two structural levers
   (`--fused-walk --sparse-sign-head --top-k 4`) at zero accuracy cost.
   `--compile` is a net loss at smoke scale (warm-up dominates).
2. **Empirical param pruning fails at this scale.** Both vocab cuts
   (50%) and d-model cuts (50%) lose 24-54 pp acc; the transformer
   baseline drops comparably (0.819 → 0.798 at d=8), confirming the
   capacity ceiling is dataset-side, not AC-specific.
3. **AC-HSiKAN architecture is already lean**: 4,602 of 164,602 params
   are architecture (2.8%); 160,000 are the vocab embedding (97.2%).
   The architecture cannot be empirically pruned further without
   structural restructuring — that's the Friedler-quotient plan
   (`docs/plans/2026-06-06-friedler-quotient-param-reduction/plan.md`).

Production configuration for 5-seed full IMDB:
```
--walk-kind cycle --pool-scatter --rotor \
--fused-walk --sparse-sign-head --top-k 4
```
Projected wall: ~100 s/seed × 5 seeds ≈ 8 min total
(vs ~37 min without these levers).

## 5-seed full IMDB (production wall config)

Ran the production config above on full IMDB (25k train / 25k val,
8 epochs, batch 64, L=200, lr=3e-3, 5 seeds). Result:

| model | val_acc (mean ± σ) | n_params | wall/seed | wall total |
|---|---:|---:|---:|---:|
| AC-HSiKAN | **0.8451 ± 0.0045** | 164,602 | 114.3 s | 571.5 s |
| Transformer | 0.8534 ± 0.0055 | 166,594 | 29.1 s | 145.5 s |

**Δ (AC − TR) = −0.0083** (~1.5σ). Per-seed:
0: −0.0110 / 1: −0.0057 / 2: −0.0020 / 3: −0.0082 / 4: −0.0145.

Notes:
- Vs the Friedler-plan Phase 1 expectation ("Δ ≤ −0.003 for cycle"):
  actual −0.0083 is narrower than the star baseline (−0.0046) but
  doesn't quite hit the plan's optimistic target. The current run
  uses `--top-k 4` (wall champion); the plan's projection assumes
  K_static=6 of K_total=8 with `--dynamic-topk`, which is a different
  config and not yet tested at full IMDB.
- σ ratio AC/TR = 0.82 — AC is actually *tighter*, beating the plan's
  "σ ≤ 2×" criterion comfortably.
- AC peaks at epoch 3–4 (~0.847) and stays flat; transformer peaks at
  epoch 1 (~0.855) and degrades to 0.812 by epoch 7 (overfits).
- Wall ratio 3.9× (vs 22× without levers). Real production candidate.
- Vs the *star* AC-HSiKAN reference at d=16 (0.8489 per Friedler-plan
  table): cycle at K=4 is −0.0038 below. The K=4 cut likely costs
  ~0.4 pp at full scale; revisit with K=8 + dynamic-topk if accuracy
  is the priority over wall.

## Open issues / follow-up

1. **Phase 1 (per plan):** test past-only on a long-range corpus.
   Candidates: LRA Text, LRA ListOps, simple synthetic copy-task,
   char-level Wikitext-2. Expected behaviour: past beats symmetric
   when the task rewards memory. *Deferred* — no LRA infrastructure in
   the repo yet.
2. **Phase 2 (per plan):** learnable per-channel γ time-constant
   $\exp(-\gamma (t - t'))$ on the dynamic-topk score. Pairs with
   `--dynamic-topk` + the new past-only candidate set.
3. **σ-collapse replication.** The 6.5× variance contraction is a
   2-seed observation. Replicate at 5 seeds before any claim. If real,
   this is a free seed-stability lever independent of the memory
   hypothesis.
4. **Plumbing gap (minor).** When `use_dynamic_topk=True` and
   `dynamic_topk_static_k > 0`, the static slots still use symmetric
   indices regardless of `candidate_temporality` (layer.py:207). Should
   forward temporality to that branch too if dynamic-topk + past is
   tested next. Tonight's smoke does *not* use dynamic-topk, so this
   has no effect on the reported numbers.

## Experiment provenance

- Git SHA: `3fd11cfeb4290f7cd627fad6e2ae747db889e596` (working tree dirty:
  the cycle-induced-memory change is the dirty diff; `git status` listed
  ~20 unrelated tracked files from earlier work in the same session,
  none touch the AC-HSiKAN forward path).
- Host: local workstation, RTX 2070 SUPER (8 GB), 6.7 GB free at launch.
- Python: 3.12.13 (`.venv/bin/python`). PyTorch installed in `.venv`.
- Memory cap: `systemd-run --user --scope -p MemoryMax=16G`.
- Dataset: IMDB (cached via `signedkan_wip/src/sequence/imdb_dataset.py`,
  `vocab_size=10000`, `L_max=200`, 5000 train / 2000 val subsampled).
- Seeds: `{0, 1}`. Identical seeds across both variants for paired
  comparison.
- Output files:
  - `/tmp/cycle-memory-smoke/symmetric.json`
  - `/tmp/cycle-memory-smoke/symmetric.log`
  - `/tmp/cycle-memory-smoke/past.json`
  - `/tmp/cycle-memory-smoke/past.log`

## §6.5 anti-pattern check

- §6.5 #1 (Cartesian-product API): no — `candidate_temporality` is a
  config field, dispatch inside one `_local_indices` function.
- §6.5 #5 (new function for new axis): no — extended the existing
  `_local_indices` signature with a default-symmetric kwarg.
- §6.5 #7 (string-typed config): the value is a string at the Python
  boundary (CLI + config dataclass), parsed inside `_local_indices`. Not
  yet promoted to an enum — three values, validated in `__post_init__`.
  Consistent with the existing `walk_kind` and `sign_head_kind` strings;
  promote together if/when any of these reach the inner DFS.
- §6.5 #11 (env-var feature flag at depth): no — value lives on
  `AcHsikanConfig`, passed explicitly through `_hybrid_indices` calls.
- §6.5 #12 (discovery pass before new artifact): yes —
  `grep -rln "candidate_temporality"` initially returned empty (the
  field was added in this session), confirming the index function
  modification is local and no existing scaffolding was missed.
- §6.5 #13 (`_v2` proliferation): no — modified `_local_indices` in
  place; no second copy.
- §6.5 #14 (preamble before execution): respected — config edit, layer
  edit, smoke launched without a planning paragraph.

## Conclusion

The cycle-induced memory mechanism is implemented at minimum-viable
surface (one config field, one default-argument extension to
`_local_indices`, one CLI flag). IMDB smoke shows the predicted
neutrality — local-context task → memory mechanism gives no signal —
which is the gating diagnostic that confirms the implementation is
not silently broken.

The actual memory test belongs on long-range corpora and is the
next Phase 1 item in the plan. Tonight's smoke is the structural
check; not a result.
