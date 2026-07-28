# R9 C1 multiseed reproduction — seed 0 was NOT a lucky shot: 22/24 verified strict K6

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · immutable source `8a0c1c7b` / tag `coin-r9-first-learned-s1-k6-delivery` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · teacher-free · TD3**

## Summary

The frozen C1 authority-unlock contract, re-run on **24 independent training seeds (1–24)** with nothing tuned, reproduces the
first-learned s1 strict K6 **22 / 24 times** — rate **0.917**, Wilson 95 % CI **[0.741, 0.977]** — with **0 safety violations** and
**0 stall / 0 clamp / 0 reversal across all 24 seeds**. Both pre-declared aggregate gates **PASS**:

| gate (fixed before any seed was seen) | threshold | result |
|---|---|---|
| `MULTISEED_K6_REPRODUCTION_PASS` | ≥ 3 verified K6, 0 safety violations | **PASS** (22) |
| `RELIABLE_K6_LEARNING_PASS` | ≥ 12 / 24 verified K6, 0 safety violations | **PASS** (22) |

Every one of the 22 claimed K6 was **independently re-verified from its saved checkpoint** against the canonical
`delivery_success(m, DELIVERY_CFG)`: dwell ≥ 6 (median 27.5, range 9–37), teacher/θ/CEM absent from the deploy path, deterministic
replay (`max|Δcoin| = 0`), safe. The 2 non-K6 seeds are **clean corridor arrivals** (min_dtz 15.83 mm / 17.51 mm — inside the 20 mm
zone — but they did not hold the strict 6-frame dwell), not stalls, clamps, or unsafe runs. **Seed 0 was not a lucky basin: the C1
recipe learns teacher-free strict delivery reliably.**

## Aggregate (24 seeds)

| quantity | value |
|---|---|
| verified strict K6 | **22 / 24** (rate 0.917) |
| Wilson 95 % CI | **[0.741, 0.977]** |
| safety violations | **0** |
| stall / clamp / reversal totals | **0 / 0 / 0** |
| Phase A / Phase B K6 | **20 / 2** (seeds 3, 10 delivered in Phase B) |
| options-to-first-K6 | median **287.5**, range 150–525 |
| wall-to-first-K6 | median **11.4 s**, range 5.8–38.5 s |
| min_dtz over K6 seeds | min **6.68 mm**, median 16.6 mm, max 19.7 mm |
| K6 dwell | median 27.5, range 9–37 |
| non-K6 seeds | 1, 7 — both `CLEAN_CORRIDOR_NO_K6` (15.83 / 17.51 mm, clean, safe, no dwell) |

## Per-seed (immutable C1 contract; per-seed fresh critics / replay / RNG / expansion head)

| seed | K6 | phase | min_dtz mm | dwell | opt→K6 | | seed | K6 | phase | min_dtz mm | dwell | opt→K6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | — | B | 15.83 | — | — | | 13 | ✅ | A | 6.68 | 9 | 400 |
| 2 | ✅ | A | 19.14 | 15 | 300 | | 14 | ✅ | A | 11.29 | 32 | 300 |
| 3 | ✅ | B | 15.69 | 35 | 350 | | 15 | ✅ | A | 13.66 | 24 | 500 |
| 4 | ✅ | A | 18.36 | 37 | 150 | | 16 | ✅ | A | 14.13 | 30 | 175 |
| 5 | ✅ | A | 19.70 | 18 | 150 | | 17 | ✅ | A | 16.12 | 28 | 250 |
| 6 | ✅ | A | 15.73 | 36 | 400 | | 18 | ✅ | A | 12.21 | 26 | 300 |
| 7 | — | B | 17.51 | — | — | | 19 | ✅ | A | 19.40 | 35 | 400 |
| 8 | ✅ | A | 16.00 | 22 | 325 | | 20 | ✅ | A | 10.76 | 27 | 525 |
| 9 | ✅ | A | 19.54 | 26 | 250 | | 21 | ✅ | A | 14.85 | 30 | 325 |
| 10 | ✅ | B | 19.20 | 15 | 275 | | 22 | ✅ | A | 19.31 | 16 | 225 |
| 11 | ✅ | A | 18.16 | 36 | 175 | | 23 | ✅ | A | 19.28 | 21 | 275 |
| 12 | ✅ | A | 18.98 | 36 | 200 | | 24 | ✅ | A | 17.13 | 35 | 250 |

## Frozen contract (nothing tuned on the seeds' results)

Every seed imported the immutable C1 contract from `8a0c1c7b` unchanged: frozen K2 clone + frozen R2 residual (α0 = 0.15) +
zero-init β = 0.85 expansion head on the same 4-D basis + TD3; same reward, curriculum (80 % KINETIC-entry / 20 % healthy
frontier), safety wrappers, and champion / freeze order (strict K6 ≻ safety ≻ 0 clamp/stall/reversal ≻ dwell ≻ min_dtz ≻ release).
Per seed: fresh critics, replay, exploration RNG, and a fresh zero-init expansion head; Phase A 600 options (eval every 25, freeze
on first strict K6), Phase B +600 only for a still-clean, non-K6, corridor-progressing seed. **Config hash `d5a1056e0c154ade`**
(identical across all seeds; any drift would change it). Seed 0 was never a training seed; the seed-0 tag is untouched.

## Execution — machine reality (transparent)

The panel ran on **`Hajdus-MacBook-Pro.local` (CPU)**, not split across kato14/kato15, and here is exactly why. kato14/kato15 are
SSH-reachable and their venv **matches this Mac bit-for-version** (torch 2.12.0, mujoco 3.10.0, numpy 2.4.6), but their repos are at
a **stale state** — no commit `8a0c1c7b`, none of the R9 code path, no clone/bank checkpoints, no MuJoCo model files — and the branch
is not on `origin`. A literal remote split would need either a **public GitHub push** (declined unilaterally — it publishes the whole
R9 line) or a large tree/bundle transfer + headless-MuJoCo bring-up. Because C1 training is **CPU-deterministic** (seed 0 reproduces
bit-identically; every seed here re-verified deterministic), the machine split was only ever for wall-clock, not for the science.
The declared **1–12 / 13–24 seed allocation was preserved as two parallel Mac processes** with the mandated
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`; the frozen Mac environment is the *most* faithful reproduction of the
tagged champion. (Running on kato14/kato15 remains available as a follow-up if a code+data transfer method is chosen.)

## Execution hygiene

Each seed wrote a self-contained `seed_NN/` (hostname, commit, seed, config hash, wall time, Phase A/B history, best +
first-K6 checkpoint, per-step event/action/force/K6 trace, deterministic-replay result, safety + cleanliness). No two seeds shared
replay, checkpoints, or output paths. Peak RSS ≈ 0.34 GB per worker (two workers ≈ 0.67 GB, far under the 16 GB cap). Half walls:
1–12 = 276 s, 13–24 = 193 s.

## Files (own scientific boundary; K0–R3-C `8a0c1c7b` modules imported UNCHANGED)

| file | role |
|---|---|
| `hymeko_rl/experiments/coin_kinetic_r3c_multiseed.py` (+263) | reproduction driver: reuses the frozen `run_seed`/`_setup`; per-seed dir + independent checkpoint `verify_k6`; Wilson CI; pre-declared gates; aggregate |
| `reports/2026-07-28-coin-r9-r3c-multiseed/multiseed_combined.json` | the 24-seed aggregate + all per-seed records |
| `reports/2026-07-28-coin-r9-r3c-multiseed/{multiseed_01_12,multiseed_13_24}.json`, `seed_01…seed_24/` | per-worker + per-seed artifacts |
| `reports/2026-07-28-coin-r9-c1-multiseed-reproduction.md` | this report |

`ruff check` clean; `radon cc -a` A/B (worst `_aggregate` refactored to C-boundary via `_gates`/`_distributions`/`_clean_totals`).
No changes to any `8a0c1c7b` file. No new §6.5 anti-patterns.

## Provenance

Immutable source commit `8a0c1c7b`, tag `coin-r9-first-learned-s1-k6-delivery` (unchanged). Python 3.11.15 / mujoco 3.10.0 / numpy
2.4.6 / torch 2.12.0 / macOS-26.5.2-arm64 (Apple Silicon, CPU). Training seeds 1–24; R2 regen seed 0 (frozen, shared read-only);
cradle 14250. Config hash `d5a1056e0c154ade`. Thread-pinned; deterministic (every K6 re-verified `max|Δcoin| = 0`).

## Status & the decision this triggers

`MULTISEED_K6_REPRODUCTION_PASS` **and** `RELIABLE_K6_LEARNING_PASS` — **22 / 24, 0 safety violations, 0 stall/clamp/reversal.** Per the
pre-registered decision framework (≥ 12/24), **C1 stably works** — the learned authority-unlock recipe reliably produces teacher-free
strict s1 K6 across independent seeds, and even its 2 misses are clean corridor arrivals a frame short of dwell. This is a reproduced
RL result, not a one-off. Committed as its own scientific boundary; the seed-0 tag is untouched. **Next (post-reproduction, needs
user green):** the clone-load-bearing ablation (clone/R2-vs-zero at total authority ≈ 1.0), then the authorised dev-cradle
characterisation — explicitly deferred. STOP.
