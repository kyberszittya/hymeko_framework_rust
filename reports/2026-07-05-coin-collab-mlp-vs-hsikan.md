# Coin task — collaborative MLP vs HSiKAN (CTDE/TD3), result note

**Date:** 2026-07-05 19:06 +09:00
**Scope:** frozen result note. No pivot, no SAC, no further experiments.

## Setup

Minimal single-seed comparison on the coin-delivery task under the validated pipeline:
collaborative CTDE (MADDPG privileged critic) + TD3+BC, reward `galambos_task_deliver.hymeko`
(deliver + `zone_progress`, oracle **certify=True**). Only the per-arm backbone differs (MLP vs SA-HSiKAN).
Delivery over 48 eval episodes; the other metrics over 12. Physics/reward/target-direction gates all passed
before this run (affordance: one-fingertip 0.083 vs two 0.854; scripted coin moves toward target).

## Result

| condition | delivery | dist_delta | coin_vel·target | fingertip_contact | both_contact |
|---|---|---|---|---|---|
| scripted 2-fingertip (baseline) | **0.854** | +0.067 | +0.062 | 0.356 | 0.180 |
| MLP BC step-0 | 0.208 | −0.016 | −0.005 | 0.224 | 0.050 |
| MLP post-RL (30k) | 0.125 | −0.037 | −0.015 | 0.120 | 0.005 |
| HSiKAN BC step-0 | 0.417 | +0.030 | +0.011 | 0.240 | 0.063 |
| HSiKAN post-RL (30k) | 0.292 | −0.002 | −0.001 | 0.055 | 0.016 |

Best checkpoint (peak on delivery) for both = **BC step-0** (MLP 0.208, HSiKAN 0.417); RL never beat its own clone.

## Conclusions

- The scripted two-fingertip controller remains the best controller: **delivery 0.854**.
- HSiKAN BC step-0 outperforms MLP BC step-0: **0.417 vs 0.208**.
- HSiKAN BC also has **positive** target-direction metrics (coin_vel +0.011, dist_delta +0.030), while MLP BC
  is negative.
- Therefore, HSiKAN's structural prior **helps imitation** on the genuinely collaborative two-arm contact task.
- However, **TD3 refinement degrades both** MLP (0.208→0.125) and HSiKAN (0.417→0.292).
- Therefore, learned collaborative RL **fails to beat the scripted controller** in this run.

## Verdict

- **Architecture signal: PASS** (HSiKAN structural prior helps imitation on the collaborative task).
- **TD3 learned-control success: FAIL** (learned collaborative RL did not beat the scripted controller).

## Provenance

- Reward: `data/robotics/galambos_task_deliver.hymeko` (+ `zone_progress`, certify True).
- Builds: `build_collaborative_offpolicy(kind="mlp")` vs `(kind="sa_hsikan")`, hidden 64, TD3+BC, 1 seed × 30k.
- Command: `python <scratchpad>/coin_arch_ab.py` (certify-gated at launch: delivers=True).
- Artifacts: `experiments/2026_07_05_18_34_coin_arch_ab_mlp/`, `experiments/2026_07_05_18_42_coin_arch_ab_hsikan/`
  (results.json + GIFs).
- Files changed (this coin-task line): `data/robotics/galambos_task_deliver.hymeko`,
  `hymeko_rl/agents/multichannel_ctde.py` (`DeterministicMLPMultiActor` + `kind="mlp"` branch),
  `hymeko_rl/experiments/exp_galambos_coord_ab.py` (`_coordination_metrics` → 4 metrics),
  `hymeko_rl/tests/test_multichannel_ctde.py`, `hymeko_rl/tests/test_reward.py`.

## Not done (deferred, not tonight)

- 6-DOF reach 2×2 TD3/SAC grid (MLP/HSiKAN × TD3/SAC) — Priority 2, on request.
- The cartpole (TD3, both solve 200/200; MLP ~3× cheaper) and 6-DOF reach (**BC-based**: MLP 0.452 m <
  HSiKAN 0.593 m) architecture ablations are reported separately; they are not part of this coin result.

---

## Appendix — BC fidelity audit + DAgger repair (single-seed, no RL) — FROZEN

After the frozen TD3 result above, a BC-fidelity audit localized the failure and a DAgger repair tested it. No
RL, no SAC, no new architecture.

**BC fidelity audit** (scripted states, per arm/phase; hybrid replacement rollouts): per-step action error is
small for both clones (< 2% of action_scale), so the collapse is closed-loop **compounding / covariate shift**,
not action-representation capacity. For MLP it is concentrated in the **APPROACH** (formation) phase —
scripted-approach + BC-push recovers delivery to 0.833. For HSiKAN it is distributed (no single phase/arm
dominant).

**DAgger repair** (roll the BC clone → query the scripted expert on the clone-visited states → aggregate →
retrain **BC only**, 3 iterations):

| stage | MLP delivery | HSiKAN delivery |
|---|---|---|
| scripted baseline | 0.854 | 0.854 |
| BC0 | 0.250 | 0.625 |
| DAgger-3 | **0.708** | 0.583 |

**Interpretation:**
1. The primary failure was **BC covariate shift / closed-loop compounding error** — not TD3, SAC, critic loss,
   or raw network capacity.
2. MLP confirms this sharply: DAgger improves delivery **0.250 → 0.708 without any RL**.
3. MLP's target-direction metrics flip negative → positive (dist_delta −0.069 → +0.045; coin_vel −0.024 →
   +0.021), so the policy becomes genuinely more task-directed.
4. HSiKAN starts much stronger under naive BC (0.625) — i.e. it is more robust before DAgger.
5. In this single seed, DAgger helps MLP much more than HSiKAN, so the architecture advantage must be stated
   carefully: HSiKAN helps naive-BC robustness, but DAgger can remove much of the initial gap.
6. TD3/SAC remain frozen; no RL interpretation is added here.

**Verdict:**
- DAgger for MLP: **PASS**.
- HSiKAN naive-BC robustness: **PASS**.
- Durable HSiKAN advantage after DAgger: **unresolved / not proven** (single-seed; HSiKAN BC0 varied 0.417–0.625
  across inits; DAgger-1 dipped for both before stabilizing).
- Learned TD3 control: **FAIL** (from the earlier run).
- Main diagnosed lever: **imitation distribution correction (DAgger) before RL.**

**Provenance:**
- BC clones audited/repaired: `experiments/2026_07_05_18_34_coin_arch_ab_mlp/policies/coin_arch_ab_mlp_s0.pt`,
  `experiments/2026_07_05_18_42_coin_arch_ab_hsikan/policies/coin_arch_ab_hsikan_s0.pt`.
- Reward: `data/robotics/galambos_task_deliver.hymeko` (deliver + `zone_progress`, certify True).
- Commands (scratchpad single-seed diagnostics; no source files changed): `python bc_fidelity.py`,
  `python dagger.py full` (roll clone → expert-label clone-visited states → aggregate → retrain BC only;
  MLP + `sa_hsikan`, 3 iterations).
- Status: **FROZEN single-seed result. No new run started.**

---

## Appendix 2 — Controlled 3-seed DAgger comparison (MLP vs HSiKAN), plain protocol — FROZEN

Seeds {0,1,2} × {MLP, HSiKAN}, stages BC0 / DAgger-1/2/3, 200 demos, 150 BC epochs, 60 DAgger rollout eps/iter,
per-seed seeded init, deliver+`zone_progress` reward. Delivery over 24 eval eps. No RL.

**A. Fixed-stage view — delivery mean ± std** (the deployable read):

| stage | MLP | HSiKAN |
|---|---|---|
| BC0 | 0.278 ± 0.153 | 0.514 ± 0.109 |
| DAgger-1 | 0.403 ± 0.071 | 0.417 ± 0.170 |
| DAgger-2 | 0.486 ± 0.138 | 0.556 ± 0.175 |
| DAgger-3 | **0.625 ± 0.090** | 0.486 ± 0.071 |

**B. Best-checkpoint diagnostic view** (max over DAgger stages per seed — diagnostic only; NOT a deployable
selector without a validation rule):

| | MLP | HSiKAN |
|---|---|---|
| per-seed best | 0.542 / 0.750 / 0.667 | 0.500 / **0.792** / 0.417 |
| best stage | D3 / D3 / D2 | D2 / D2 / D1 |
| best-ckpt mean ± std | **0.653 ± 0.085** | 0.570 ± 0.161 |

Scripted baseline = 0.854.

**Interpretation (recorded):**
- DAgger is confirmed as the necessary BC covariate-shift correction.
- MLP+DAgger is robust and consistent: BC0 mean 0.278 → best-checkpoint mean 0.653, improvement in **all 3 seeds**.
- HSiKAN has stronger naive-BC robustness: BC0 mean 0.514 vs MLP 0.278.
- HSiKAN advantage after plain DAgger is **not** confirmed on mean: MLP is ahead at fixed D3 (0.625 vs 0.486) and
  on best-checkpoint mean (0.653 vs 0.570).
- HSiKAN has the **highest learned peak: 0.792** (seed 1, D2), close to the scripted 0.854.
- So the architecture story is not "HSiKAN fails"; it is **"HSiKAN is high-ceiling but unstable under plain
  DAgger"** (D1 dip, non-monotonic, D3 over-correction falling below BC0).

**Reporting rule:** report both views separately — fixed-stage (MLP wins, deployable) and best-checkpoint
(HSiKAN highest ceiling, diagnostic). Do NOT overclaim best-checkpoint performance without a
validation/checkpoint-selection rule.

**Verdict:** the main lever is **DAgger / distribution correction**, not architecture (MLP+DAgger catches/passes
HSiKAN on mean). HSiKAN is high-ceiling-but-unstable; a stability variant is motivated. Status: **FROZEN.**
Provenance: scratchpad `dagger.py multiseed`, deliver+`zone_progress` reward; no source files changed.

---

## Appendix 3 — Runtime / profiling notes (DEFERRED until after the variant result freezes)

**Confirmed meaning of the "0.2–0.3 ep/s" figure** (measured from the log format, not assumed):
- It is **BC-training epochs/second** — from `behaviour_clone`'s `[bc] epoch k/N … ep/s` progress lines.
- It is **not** rollout episodes/second.
- It is **not** evaluation episodes/second.
- At 0.2 ep/s, 150 epochs take **~12–13 minutes**.
- For 3 seeds × 4 stages, HSiKAN BC training can plausibly dominate the ~2.5 h wall — but this remains a
  **profiling hypothesis until measured**.

**Profiling plan** (only after the variant result is frozen; do **not** profile during an active run — the
measurement would be contaminated). Profile the runtime split with `py-spy` (§10) on a quiet machine:
- BC training · rollout collection · expert relabeling · evaluation · metric logging.
- GIF/video excluded — not generated in this scratchpad DAgger harness.

**Safe speedups to check only after freezing** (no change to metrics, seeds, reward, evaluation protocol, or
DAgger protocol): cache scripted demonstrations · cache aggregated DAgger datasets · batch HSiKAN inference
(rollout is currently B=1/step) · precompute static hypergraph structures · avoid reparsing `.hymeko` in loops ·
parallelize seeds · use GPU only if numerically consistent.

**Contention observation (measured, provisional):** the 0.2–0.3 ep/s was measured **under contention** (the
plain 3-seed run was executing concurrently). Once that finished, the HSiKAN BC rate rose to **~1.7 ep/s** (~8×),
so HSiKAN BC training is **not inherently that slow**. **Leading runtime hypothesis: CPU contention / concurrent
runs**, not the HSiKAN path itself. Still to be confirmed by the deferred profile after the variant freezes; no
optimization during the active run.

**Refined runtime hypothesis (measured mid-variant):** BC rate is **~1.7 ep/s on small stages but back to ~0.2
ep/s on the large replay-preserved stages** — so HSiKAN BC speed is affected by **both machine contention AND
dataset size** (the preserved-replay oversampling increases the training set, raising per-epoch cost). The
deferred profile must therefore separate four factors: (1) contention, (2) dataset size, (3) model compute,
(4) Python / batching overhead.

**Runtime clarification (recorded):** the 0.2–0.3 ep/s is **BC-training epochs/second**, NOT simulator/rollout
speed — do **not** describe it as simulator speed. HSiKAN training speed is sensitive to **machine contention,
dataset size, replay-preserved DAgger data, and batching**. This must be **profiled separately after the
scientific result is frozen** (not during an active run).

**Runtime arithmetic (the key distinction, recorded):**
- 1.7 ep/s ⇒ ~0.59 s/epoch ⇒ ~88 s per 150-epoch HSiKAN BC training.
- 12 such trainings at that rate ⇒ ~18 min of BC training alone.
- But the variant actually took **~1 h 50** → **1.7 ep/s cannot represent the full average pipeline cost.**
- So the runtime issue is **not simply "HSiKAN is slow."** The full HSiKAN+DAgger pipeline wall is a **mixture**:
  dataset-size-dependent BC training (rate falls to ~0.2 ep/s on large sets), rollout collection, expert
  relabeling, evaluation, metric logging/serialization, Python overhead, B=1 rollout inference, and machine
  contention. **Do not optimize until the component profile identifies the dominant term.**

**Runtime diagnosis — component profile (measured on a quiet machine; rate-based so it separates rate from
volume). Full-pipeline estimate, 3 seeds × 4 stages:**

| component | time | share |
|---|---|---|
| **BC training** | **46.6 min** | **74.1%** |
| rollout collection (relabel+infer+physics) | 7.8 min | 12.4% |
| — expert relabeling (subset) | 0.4 min | 0.7% |
| evaluation | 6.4 min | 10.1% |
| metric logging / serialization | ~0 min | 0.0% |
| demo collection | 2.2 min | 3.5% |
| TOTAL (est., uncontended) | 62.9 min | — (measured variant wall ~110 min) |

Measured rates: BC **1.20 s/epoch @ 27k → 2.34 s/epoch @ 100k** (dataset-size-dependent); rollout/ep: relabel
0.05 s · clone B=1 infer 0.42 s · physics 0.40 s; eval/stage: measure 24 s + phase_err 7.8 s.

**Main conclusion (carefully phrased):**
- The dominant runtime component is **BC training** (~74% of the estimated **uncontended** pipeline; likely
  **~80%+** in the replay-preserved / partially contended real run, where the 63→110 min gap is entirely BC
  inflation from larger replay datasets + contention).
- The **simulator is NOT the main bottleneck.**
- Expert relabeling is **negligible** (0.7%); metric logging is **negligible** (~0%); evaluation is **secondary**
  (10%); rollout **B=1 inference is batchable but not the main target** (6%).
- **The issue is dataset-size-dependent HSiKAN BC training plus machine contention — not inherently slow
  simulation.**

**Safe engineering optimizations** (do not change metrics/seeds/reward/eval/DAgger protocol — deferred, not run):
1. Run seeds in parallel **only** if the machine has headroom and it does not recreate contention.
2. Test a clean **GPU BC-training path**, verifying numerical consistency against CPU.
3. Cache scripted demonstrations.
4. Cache aggregated DAgger datasets where it does not change the data.
5. Avoid repeated parsing / static hypergraph reconstruction.
6. Batch rollout inference if easy (secondary).

**Protocol-changing variants** (scientifically interesting and probably useful, but must be treated as **new
controlled variants, NOT speed optimizations**): bounding replay-dataset growth · APPROACH-prioritized sampling ·
dataset subsampling · checkpoint-based early stopping.

Status: profile **recorded**; **not optimizing yet**; scientific result remains **frozen**.

---

## Appendix 4 — HSiKAN frozen-trunk stability variant — COMPLETE, FROZEN (negative result)

Variant = HSiKAN only, seeds {0,1,2}, same BC0/D1/D2/D3 structure, with: lower LR on DAgger retrains
(1e-3→3e-4), preserved expert-demo replay ratio (D0 oversampled to ~50%), **trunk frozen / heads-only for
DAgger-1** then unfrozen for D2/D3, warm-start continual.

**Seed 0 (provisional — does NOT support the variant):**

| stage | delivery | trunk |
|---|---|---|
| BC0 | 0.667 | — |
| DAgger-1 | 0.125 | frozen |
| DAgger-2 | 0.208 | unfrozen |
| DAgger-3 | 0.500 | unfrozen |

Best-DAgger 0.500 = **below** BC0 0.667 — same ending as plain HSiKAN s0.

**Seed 1 (partial — the key high-ceiling seed):** BC0 0.417 → D1 **0.208** (frozen) → D2 0.708 (unfrozen) → D3
pending. Plain HSiKAN s1 was 0.417 → 0.625 → **0.792** → 0.583.

**Emerging pattern (provisional — do NOT conclude until the full table):**
- The trunk-freeze / heads-only first DAgger step has hurt **DAgger-1 in both observed seeds**: s0 frozen-D1 0.125
  vs plain 0.208; s1 frozen-D1 **0.208 vs plain 0.625** (much worse).
- On the key high-ceiling seed 1, the variant currently **suppresses the plain peak** (D2 0.708 vs plain 0.792).
- **Hypothesis:** heads-only first-step adaptation is **too restrictive** — it preserves the structural trunk but
  blocks the policy from responding to learner-distribution corrective data. If this holds through s1-D3 and s2,
  mark the variant **harmful / not useful**.

**Seed 1 — recorded as a stability-vs-ceiling trade-off (NOT a clean failure):** plain 0.417→0.625→**0.792**→0.583
vs variant 0.417→0.208→0.708→**0.708**.
- Plain HSiKAN reaches the higher peak (0.792); the variant **suppresses** it (best = 0.708).
- But the variant **avoids the late D3 collapse** (0.708→0.708, vs plain 0.792→0.583).
- So the variant may improve **fixed-final stability** at the cost of **best-checkpoint ceiling**.
- The frozen D1 **remains harmful** (0.208 vs plain 0.625).

**Provisional hypothesis:** the heads-only frozen-trunk D1 is too restrictive and blocks useful early DAgger
adaptation; the later stability likely comes from the **lower LR + preserved expert replay**, not from the
trunk-freeze.

**Final verdict must separate three axes:**
1. **Best-checkpoint** — variant is WORSE if it suppresses the peak.
2. **Fixed-final** — variant may be BETTER if it avoids the D3 over-correction.
3. **Mechanism** — trunk-freeze appears harmful at D1; replay / lower-LR may help late-stage stability.

Status: **COMPLETE — FROZEN.**

**Final variant table (3 seeds):**

| seed | BC0 | D1 (frozen) | D2 | D3 |
|---|---|---|---|---|
| 0 | 0.667 | 0.125 | 0.208 | 0.500 |
| 1 | 0.417 | 0.208 | 0.708 | 0.708 |
| 2 | 0.458 | 0.125 | 0.208 | 0.375 |

Mean ± std: best-checkpoint **0.528 ± 0.137**, fixed-D3 0.528 ± 0.137, D1 **0.153 ± 0.039**.
(Comparison: plain HSiKAN+DAgger best-ckpt 0.570 ± 0.161; MLP+DAgger 0.653 ± 0.085.)

**Corrected interpretation — this is NOT "HSiKAN+DAgger does not work" (unsupported):**
1. Plain HSiKAN+DAgger achieves the **highest learned peak observed so far** — seed 1: 0.417 → **0.792**
   (scripted baseline 0.854).
2. So HSiKAN+DAgger clearly has **high potential**.
3. The problem is **instability / variance**: mean best-checkpoint below MLP+DAgger, non-monotonic, and the high
   peak is **not reliable across seeds**.
4. The **frozen-trunk stability variant FAILED**: it suppressed the peak (best 0.528 < plain 0.570), did **not**
   beat MLP+DAgger, did **not** clearly improve plain HSiKAN, and D1 collapsed badly (0.153).
5. Therefore the negative result is specifically **"frozen-trunk heads-only DAgger-1 is harmful / too
   restrictive,"** NOT "HSiKAN+DAgger cannot work."

**Next HSiKAN direction (LATER — not run now): a NO-FREEZE stabilization** — lower LR + expert-demo replay ratio
+ possibly phase-weighted APPROACH DAgger + validation checkpoint selection; explicitly **not** frozen-trunk /
heads-only D1.

---

## Appendix 5 — No-freeze HSiKAN stabilization variant (IN PROGRESS — PROVISIONAL)

New controlled variant: HSiKAN + DAgger, **no** trunk freeze, lower LR ($10^{-3}\to3\times10^{-4}$), expert-replay
ratio 0.5, APPROACH-prioritized sampling ×2, validation checkpoint selection (val seeds 5000/12 eps; test 9000/24
eps). `device=cuda` (consistency-verified, §GPU check). Serial seeds. The scientific result appends to the LaTeX
report on completion; this is a provisional interim.

**Seed 0 (PROVISIONAL — do not conclude):** BC0 0.667 (val 0.25) → D1 0.042 → D2 0.333 → D3 0.208.
- No-freeze DAgger did **not** improve over BC0 on seed 0; BC0 remained best on test (0.667); DAgger stages degraded.
- **However, validation checkpoint selection worked correctly**: val = [0.25, 0.083, 0.083, 0.25] → selected BC0
  (declined the weaker DAgger checkpoints).
- Interim interpretation: no improvement yet, but the validation-selection mechanism functions as intended
  (prevents DAgger over-correction from being selected). **Seeds 1–2 decide** whether the no-freeze variant can
  preserve/recover the high HSiKAN ceiling.
