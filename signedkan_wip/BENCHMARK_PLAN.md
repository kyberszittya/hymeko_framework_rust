# Benchmark plan — speedup levers verification

**Goal:** before claiming any speedup, verify correctness against
known-good reference numbers. Then measure speed on a tiered set of
fixtures from tractable (where reference still runs) to large
(where reference is impractical and we need the speedup to even
test).

## Reference (ground-truth) numbers

These are the measured outputs from the *unaccelerated* path in
the current codebase. Any new enumerator/kernel must reproduce
these exactly (counts) or within stated tolerances (training
metrics).

### k=4 cycle enumeration (`construct_k(g, k=4)`)

| fixture | nodes | edges | k=3 triads | k=4 cycles | reference time |
|---|---:|---:|---:|---:|---:|
| karate (faction)   | 34    | 78      | unmeasured | unmeasured | < 1 s |
| SBM 200/k=4 s=0    | 200   | 1,727   | 1,015      | 11,791     | < 1 s |
| SBM 400/k=5 s=0    | 400   | 6,338   | 6,381      | 134,307    | ~1 s |
| hier-SBM 240 s=0   | 240   | 2,362   | 1,704      | 21,145     | < 1 s |
| **bitcoin_alpha**  | 3,783 | 24,186  | 22,153     | **615,962**| **6.8 s** |
| **bitcoin_otc**    | 5,881 | 35,592  | 33,493     | **1,043,996** | **17.1 s** |
| **slashdot**       | 82,140| 549,202 | 579,565    | unmeasured (>10 min in pure Python) | impractical |
| epinions           | 131,828| 841,372| unmeasured | unmeasured (probably millions) | impractical |

### Correctness invariants

For any new k=4 enumerator on a fixture where reference is
tractable, the following must hold:

1. **Cycle count match**: `len(fast) == len(ref)`.
2. **Cycle set match** (after canonicalisation): `set(t.v for t in fast) == set(t.v for t in ref)`.
3. **Edge-sign match**: for each cycle, the 4-tuple `edge_signs`
   matches the reference.
4. **σ assignment match**: per-vertex `sigma` matches.
5. **Balance flag match**: `balanced` matches.

### HSiKAN training (sanity, accelerated forward / fp16 / etc.)

Reference numbers from Phase 8 SBM positivity sweep (single
canonical cell):

| fixture | recipe | AUC | F1m |
|---|---|---:|---:|
| sbmsweep_pos50_s0 | hsikan_mixed leanest, k=(3,4) | ~0.95 | ~0.85 |
| bitcoin_alpha (Phase 4) | hsikan_mixed leanest + EC | ~0.94 | ~0.79 |
| bitcoin_otc (Phase 4)   | hsikan_mixed leanest + EC | ~0.92 | ~0.80 |

**Tolerance:** any forward-pass speedup (torch.compile, fp16, fused
kernels) must produce metrics within **±0.005 AUC / ±0.010 F1m**
of these references at 3 seeds. Larger deltas indicate a
correctness bug, not noise.

## Speedup targets

| lever | target speedup | priority | est. impl. cost |
|---|---:|---|---:|
| **(1) Vectorised inner pair loop in fast k=4** | 10–100× over current `n_tuples_fast.py` on slashdot | P0 — unblocks slashdot HSiKAN-mixed | 15 min |
| (2) `torch.compile` on `_catmull_rom_eval`     | 1.3–2× per-epoch | P1 — easy free win | 5 min |
| (3) Catmull-Rom basis precompute               | 1.15–1.4× per-epoch | P1 — refactor only | 30 min |
| (4) `autocast(fp16)` training loop             | 1.3–1.5× per-epoch | P1 — needs stability check | 15 min |
| (5) Triton fused gather+blend kernel           | 2–4× per-epoch | P2 — paper-grade infra | half day |
| (6) Rust/PyO3 native enumerator                | linear scaling, parallelisable | P2 — paper-grade infra | half day |
| (7) Triad-minibatching with grad accumulation  | enables full-fixture forward at fixed memory | P2 — scaling | 1–2 hours |

## Test fixtures (tiered)

**Tier A — unit-test speed:**
- karate (34 / 78)            → both reference and accelerated must finish in <100 ms
- SBM 200 (200 / 1.7k)        → both <1 s

**Tier B — correctness verification at non-trivial scale:**
- bitcoin_alpha (3.8k / 24k)  → reference 6.8 s, accelerated <1 s expected
- bitcoin_otc   (5.9k / 36k)  → reference 17.1 s, accelerated <2 s expected

**Tier C — accelerated-only (reference impractical):**
- slashdot (82k / 549k)       → accelerated target: <60 s for k=4 enumeration
- epinions (132k / 841k)      → accelerated target: <120 s for k=4 enumeration

## Test scripts to add

- `signedkan_wip/src/benchmarks/k4_correctness.py` — runs reference
  vs accelerated on Tier A+B, asserts invariants 1–5.
- `signedkan_wip/src/benchmarks/k4_speed.py` — measures wall-clock
  + memory peak across all tiers, writes a comparison JSON.
- `signedkan_wip/src/benchmarks/training_speed.py` — measures
  training per-epoch wall-clock with each forward-path lever
  toggled; asserts AUC/F1m within tolerance.

## Acceptance criteria for shipping a lever

1. **Correctness**: passes all five invariants on Tier A + B fixtures.
2. **Speed**: meets or exceeds the target speedup on at least one
   tier where it is materially relevant.
3. **No regression**: training metrics (AUC, F1m) within ±0.005 /
   ±0.010 of reference at 3 seeds on the canonical SBM and
   bitcoin_alpha fixtures.
