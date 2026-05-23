# Phases 11 / 12 / 13 / 14 brief — activation-side family characterisation

**Date launched:** 2026-04-26 evening (extended 2026-04-27 with ph14)
**Phases covered:**
- **Phase 11** — `cross_layer_mi` (Path F) characterisation
- **Phase 12** — `total_correlation_mi` (Path I) sweep
- **Phase 13** — Path I × CapsMLP MNIST stress test
- **Phase 14** — Path I × deep architectures (ResMLP-20, HighwayMLP-10/20) × {MNIST, FashionMNIST} + CapsMLP×FashionMNIST companion

**Logs:** `/tmp/thesis_iv_views_{ph11,ph12,ph13,ph14}.log`
**Scripts:** `python/benches/thesis_iv_hard/run_overnight_views_ph{11,12,13,14}.sh`
**CSVs:** `data/benchmarks/thesis_iv_hard_*.csv` (one per `RUN` invocation)
**Chain status:** ph11 → ph12 auto-chained via `/tmp/.../ph11_to_ph12_chain.log` watcher.
**Abort:** `python/benches/thesis_iv_hard/abort_ph12_chain.sh` kills the watcher; ph11 untouched.

---

## Why these phases

The 2026-04-26 20:02 run of `cross_layer_mi` on spirals 100×50 at λ=0.1
returned **Δ = +0.0003 pp, t = +0.16, W/L = 28/54** — a null on the same
dataset where spectral arms (Path A/B) achieve Δ ≈ +0.4 to +0.7 pp at
*** significance. Two open questions:

- **Q11** Is activation-side regularisation fundamentally weaker, or
  are we at the wrong λ? **(Phase 11 answers.)**
- **Q12** Does a strictly more complex term — multi-information across
  all layers, with KL-driven and variance-driven λ schedules — recover
  the gap? **(Phase 12 answers.)**
- **Q13** Does the new term neutralise (or invert) the historical
  significant-negative on CapsMLP MNIST that motivated the activation-side
  family? **(Phase 13 answers.)**

---

## Phase 11 — `cross_layer_mi` (Path F) characterisation

**Hypothesis:** Path F is a *function-form* mismatch with spirals
(activation-side, mini-batch-stochastic, pairwise) rather than a λ
mismatch. Expected: null across all five λ values; modest signal on
the MNIST family where layer redundancy is non-trivial.

### 9 runs

| # | dataset       | seeds×ep | view     | λ    | reg-every-n | expected Δ | rationale |
|---|---------------|----------|----------|------|-------------|------------|-----------|
| 1 | spirals       | 100×50   | dataflow | 0.1  | 10          | null       | Anchor; reproduces 20:02. |
| 2 | spirals       | 100×50   | dataflow | 0.01 | 10          | null       | Sub-threshold λ ⇒ ≈ baseline. |
| 3 | spirals       | 100×50   | dataflow | 0.03 | 10          | null       | Mid-low λ. |
| 4 | spirals       | 100×50   | dataflow | 0.3  | 10          | weak/neg   | Above natural scale ⇒ may over-regularise. |
| 5 | spirals       | 100×50   | dataflow | 1.0  | 10          | negative   | λ overpowers task loss; expect drop. |
| 6 | circles       | 100×50   | dataflow | 0.1  | 10          | null       | circles is hard for *all* arms. |
| 7 | mnist_small   | 33×15    | dataflow | 0.01 | 50          | weak +     | redundancy plausible in 4-layer MLP. |
| 8 | fashion_mnist | 33×15    | dataflow | 0.01 | 50          | null       | Same arch as mnist_small; harder distribution. |
| 9 | kmnist        | 33×15    | dataflow | 0.01 | 50          | weak ±     | Spectral arms went negative here; activation-MI may flip. |

### Acceptance criteria

- **If runs 1–5 are all null and runs 6–9 are at most marginally
  significant**: Path F is honestly weaker than Path A/B across the
  current λ range. Phase 11 then becomes the *negative result* in the
  paper — activation-side pairwise MI is not a substitute for spectral
  entropy.
- **If any run 1–5 produces Δ > +0.10 pp at p < 0.05**: there's a λ
  regime worth a focused follow-up.
- **If runs 7–9 land Δ > +0.15 pp**: layer redundancy genuinely matters
  on these datasets and Path F is a real (if niche) tool. Worth folding
  into the universality table.

---

## Phase 12 — `total_correlation_mi` (Path I) sweep

**Path I** = strictly more complex than Path F:

```
TC_2     = Σ_l H_2(K_l) − H_2(K_join_all_layers)        # multi-information
p_t      = eigvalsh(K_join) / Σ                          # joint spectral dist
KL_step  = Σ p_{t-1} · log(p_{t-1}/p_t)                  # entropy change
σ²_t     = Var(p_t)
μ_t      = β · μ_{t-1} + (1−β) · σ²_t                    # variance EMA
inertia  = μ_t / (μ_t + 1e-3)            ∈ (0, 1)
λ_factor = exp(−η · KL_step), clamp [0.1, 10]            # KL-feedback
λ_eff    = λ_factor · var_factor(mode) · TC_2
```

**Three variance-modes (`--tc-variance-mode`):**
- `damp`     : `var_factor = (1 − inertia)` — quiet during high spread.
- `amplify`  : `var_factor = inertia` — push harder when spectrum settles.
- `mix`      : `var_factor = w·inertia + (1−w)·(1−inertia)`,
                `w = exp(−η·KL)` clipped to [0,1] — stage-aware blend.

**Hypothesis:**
- **Multi-way TC > pairwise MI** when the layers are mutually redundant
  (4+ layer plain MLPs).
- **Mode = `mix` ≥ `damp` ≥ `amplify`** in stationary regimes; `mix` is
  the principled default because it transitions from damp (transient)
  to amplify (steady state) automatically.
- **β momentum is mostly cosmetic on synthetic** (50 epochs is small
  relative to EMA reset time at β=0.9), but **β=0.99 on MNIST may
  matter** — long enough trajectory for the EMA to dominate.

### 13 runs

#### Q1 — variance-mode head-to-head (spirals 100×50, λ=0.1)

| run | mode    | expected Δ | rationale |
|-----|---------|------------|-----------|
| 1   | damp    | null/neg   | retires when spectrum settles wide ⇒ nearly baseline late-training. |
| 2   | amplify | small +    | hits the regulariser late but may over-shoot during transients. |
| 3   | mix     | small +    | stage-aware ⇒ best of both; expected to beat the other two. |

**Acceptance:** if `mix` < `amplify`, the stage-aware logic is wrong;
keep `amplify` as default.

#### Q2 — λ sweep on spirals (mode=mix)

λ ∈ {0.01, 0.03, 0.1, 0.3, 1.0}. **Expected:** dome-shaped curve peaking
near λ = 0.1; decay at extremes. Same shape as scalar_entropy, but
shifted left because TC is larger in magnitude than scalar entropy
(L sums vs single H).

#### Q3 — cross-dataset (mode=mix, baseline shipped per dataset)

| run | dataset       | seeds×ep | λ    | expected Δ | rationale |
|-----|---------------|----------|------|------------|-----------|
| 8   | circles       | 100×50   | 0.1  | small +    | TC's joint reading captures the structural effect Path F missed. |
| 9   | mnist_small   | 33×15    | 0.01 | weak +     | 4-layer MLP ⇒ multi-way redundancy ⇒ TC bites harder than pairwise. |
| 10  | fashion_mnist | 33×15    | 0.01 | null       | Same arch; harder distribution caps the effect. |
| 11  | kmnist        | 33×15    | 0.01 | small +    | Where spectral went *negative*, TC may go *positive* — orthogonal axes. |

#### Q4 — β momentum sensitivity (spirals, mode=mix)

| run | β    | expected Δ | rationale |
|-----|------|------------|-----------|
| 12  | 0.0  | null       | No memory ⇒ var_factor jumps every step ⇒ noisy λ ⇒ near-baseline. |
| 13  | 0.99 | weak −/null| Frozen inertia ⇒ var_factor effectively constant ⇒ behaves like Path F. |

**Acceptance:** if both extremes underperform β=0.9, the EMA design is
load-bearing. If β=0.0 ≥ β=0.9, the momentum is decoration.

---

## Phase 13 — Path I × CapsMLP MNIST

**Why it's separate:** CapsMLP is the historical *significant negative*
under unnormalised spectral entropy (Δ = −0.057 pp, p < 0.05 at 33
seeds × 10 epochs). Path B (`scalar_entropy_normalized`) neutralised
it (Δ ≈ −0.002 ns). The capsule architecture has explicit dynamic
routing ⇒ explicit cross-layer dependencies ⇒ activation-side MI is
the *natural* regulariser to apply here. If Path I works anywhere,
this is the strongest a-priori bet.

**Hypothesis:** Path I (TC + KL feedback + variance momentum) on
CapsMLP MNIST will produce Δ ≥ 0 — strictly better than the original
spectral-entropy negative. A modest positive (Δ > 0 but not
significant) would be a publishable *neutralisation* result; a
significant positive (Δ > +0.05 at p < 0.05) would be the
strongest argument for the activation-side family in the universality
programme.

### 5 runs

#### Q1 — variance-mode head-to-head (mnist_capsnet 33×10, λ=0.01)

| run | mode    | expected Δ | rationale |
|-----|---------|------------|-----------|
| 1   | damp    | small +    | CapsMLP's routing settles fast ⇒ damp doesn't hide for long. |
| 2   | amplify | small +    | dynamic routing creates persistent variance ⇒ amplify pushes through it. |
| 3   | mix     | small +    | best of both — should match or beat the other modes. |

#### Q2 — λ sweep at mode=mix

| run | λ      | expected Δ | rationale |
|-----|--------|------------|-----------|
| 4   | 0.001  | null       | sub-threshold ⇒ nearly baseline. |
| 5   | 0.1    | weak −     | over-pushes against capsule dynamics ⇒ may revert sign. |

**Acceptance:**
- **Best mode × best λ produces Δ > 0**: phase 8/F's failure was term-form
  not target-arch; activation-side family vindicated on its motivating
  fixture.
- **Δ remains negative at all (mode, λ)**: CapsMLP is hostile to
  activation-side regularisation full-stop; argument for keeping
  Path B as the universality default.

---

## Phase 14 — Path I × deep architectures (scouting sweep)

**Why it's separate:** TC over L=20 layers is exactly what total
correlation is *for*. If Path I outperforms anything in the universality
table, the deep architectures (ResMLP-20, HighwayMLP-10/20) are the
strongest a-priori bet — pairwise spectral entropy can't express L-way
joint structure but TC can.

**Compute philosophy:** scouting at 15 seeds × 5–15 epochs, mode=mix.
Detects effect sign and rough magnitude; does NOT chase p-values.
Follow-up with full-power 33×15 paired runs only on combos that surface
≥ +0.10 pp at the scouting budget.

### 6 runs

| run | dataset                    | seeds×ep | rationale |
|-----|----------------------------|----------|-----------|
| 1   | mnist_resnet_20            | 15×10    | 21-way joint Gram on residual depth. |
| 2   | fashion_mnist_resnet_20    | 15×10    | Sibling distribution, same arch — does the effect transfer? |
| 3   | mnist_highway_10           | 15×15    | Gated MLP at moderate depth (cheaper than -20 for first scout). |
| 4   | fashion_mnist_highway_10   | 15×15    | Highway sibling. |
| 5   | mnist_highway_20           | 15×10    | Full-depth Highway scout (skip FashionMNIST for budget). |
| 6   | fashion_mnist_capsnet      | 15×10    | Companion to ph13 — completes capsule-network coverage. |

**Acceptance:** Δ ≥ +0.10 pp on any (arch, dataset) at 15 seeds → schedule
a 33-seed power follow-up. Δ ≤ 0 across the board → activation-side
family genuinely doesn't scale to deep nets, document as a negative
result, lean on Path A/B for the universality argument.

## Combined timeline

- **20:22 (2026-04-26)** ph11 launched (background)
- **23:36** ph11 done; ph12 auto-fired via chain watcher
- **~02:54 (2026-04-27)** ph12 expected to finish (revised estimate
  given actual MNIST-family per-seed timings observed in ph11)
- **manual launch** of ph13 after ph12 (~30–40 min) — *not* chained.
- **manual launch** of ph14 after ph12 + ph13 review (~2.5–3 h). Held
  back so ph12 + ph13 results inform whether the deep-arch sweep is
  worth burning compute on. ph14 lives at
  `run_overnight_views_ph14.sh` with full configuration.

## Deliverables checklist (after results land)

- [ ] Aggregate ph11 + ph12 + ph13 + ph14 CSVs into `RESULTS_VIEWS_SUITE.md`
      via `aggregate_views_suite.py`.
- [ ] Update `reports/phases_and_paths.tex`: add §3.x Path I entry,
      §3.y phase-11/12/13 sections, headline rows for `cross_layer_mi`
      and `total_correlation_mi` in the §4 table (replacing the
      placeholder "queued" row).
- [ ] Patch the doc's known fix-me on `entropy_target_ka` ceiling
      clamp 1.0 → 0.75 if not done elsewhere.
- [ ] If Path I shows positive on any dataset: add §8.1 to
      `reports/sanchez_giraldo_framework.md` with the multi-information
      derivation and the variance-mode rationale.
