# option_dagger_multiseed_demo_mix_v1 — morning report (2026-07-08)

> ⚠ **SUPERSEDED (2026-07-08, same day).** The POSITIVE below was a **single training seed** (train_seed=0). The
> training-seed robustness run (8 seeds × {0.25,0.5,0.75}, kato15) came back **NOT_ROBUST**: median ft_dom is
> *below* baseline and the pooled ft_dom is statistically *worse* — the mix_25 POSITIVE was a favorable seed. The
> `mix_25_msdm.pt` checkpoint is **not** a robust artifact. See
> `reports/2026-07-08-option-msdm-trainseed-robustness.md` for the corrected verdict. Read this report only through
> that lens. (monitor_score + sustained-PUSH gains *are* robust; the delivery cost is what fails to hold.)

**Run:** `experiments/2026_07_08_option_msdm/` · wall **865.6 s** · CPU, Mac (arm64, torch 2.12.0)
**Driver:** `python -m hymeko_rl.experiments.exp_option_msdm --stage full` · `run.log` · git `5b53a92`
**Plan:** `docs/plans/2026-07-08-option-msdm/` (md/tex/pdf/tikz/mmd)

## Verdict: **POSITIVE** (mix_25) — with the trade-off resolved

The contact↔delivery trade-off from the option-level run is **resolved**. Two findings settle it:

1. **The ft_dom "drop" was single-seed noise.** Under proper multi-seed eval (4 seeds × 48), the baseline is
   ft_dom **0.578 ± 0.060** — the 0.75 we had been gating against was a lucky seed (9000). The regenerated
   option-DAgger is **0.620 ± 0.077**, statistically **tied** (2-proportion z-test p=0.40, Δ +0.042) and
   mean-above baseline — *not* a regression.
2. **A balanced demo mix wins.** Tagging expert states as sustained-PUSH vs delivery-completion and BC-fine-tuning
   at increasing sustained fractions produces a clean dose-response, with a sweet spot at **25 %**:

| candidate | frac_sustained | ft_dom (mean±std) | tie-test vs base | monitor_score | sustained-PUSH/ep | exploit | verdict |
|---|---:|---:|:--|---:|---:|---:|---|
| baseline (frozen DAgger) | — | 0.578 ± 0.060 | — | 0.243 | 0.354 | 0.0 | — |
| option_dagger (regen) | — | 0.620 ± 0.077 | tied (p=0.40) | 0.353 | 0.797 | 0.0 | reference |
| mix_0 (delivery-only) | 0.00 | 0.448 ± 0.031 | **worse** (p=0.011) | 0.353 | 0.740 | 0.0 | NEGATIVE_WITH_MECHANISM |
| **mix_25** | **0.25** | **0.594 ± 0.043** | **tied** (p=0.76) | **0.438** | **0.813** | 0.0 | **POSITIVE** |
| mix_50 | 0.50 | 0.531 ± 0.031 | tied (p=0.36) | 0.356 | 0.885 | 0.0 | PROMISING_TRADEOFF |
| mix_75 | 0.75 | 0.516 ± 0.070 | tied (p=0.22) | 0.374 | 0.839 | 0.0 | PROMISING_TRADEOFF |
| mix_100 (sustained-only) | 1.00 | 0.099 ± 0.017 | worse (p<1e-3) | −0.115 | 0.125 | 0.0 | INCONCLUSIVE_WITH_NEXT_FIX |
| curriculum (deliver→sustained) | — | 0.151 ± 0.023 | worse (p<1e-3) | 0.011 | 0.255 | 0.0 | INCONCLUSIVE_WITH_NEXT_FIX |

**mix_25** improves/preserves every headline metric and raises sustained-contact coverage without exploit:
ft_dom **tied/preserved** (0.594 vs 0.578, mean +0.016, p=0.76), monitor_score **+0.195** (0.438 vs 0.243, nearly
2×), sustained-PUSH **2.3×** (0.813 vs 0.354), both-contact and ft-progress-in-contact both up, exploit and
arm-body **0**. Deployable checkpoint saved: `mix_25_msdm.pt` (md5 `c9172989…`).

Stage-ledger sentence: **Scripted PushDemonstrator ~0.90 · tuned option expert (sustained-PUSH 1.76/ep, mon_score
0.501) · frozen DAgger baseline ft_dom 0.578 (multi-seed) · learned mix_25 ft_dom 0.594 (tied) with monitor_score
0.438 and sustained-PUSH 0.81.** For the first time in this arc, a **learned** policy improves the frozen baseline's
contact quality without losing delivery.

## The mechanism (fully measured)

- **Too little sustained contact (mix_0)** hurts ft_dom (0.448) — pure delivery-completion demos lose fingertip
  dominance. **Too much (mix_100 / curriculum)** collapses delivery (0.10 / 0.15) — the holding-overfit reproduced.
  **25 % is the balance** that keeps delivery while injecting sustained two-finger contact.
- The lever is confirmed: **not per-step RL, but option-generated sustained-contact demonstrations + demo-mix
  imitation.** The tuned option (θ*, sustained-PUSH 1.76/ep) supplies the missing contact regime; a 25 % mix
  transfers it into the learned MLP.

## Required per-branch fields

- **Part 1** (multi-seed, 4×48): baseline / option_dagger / tuned expert tables above; option_dagger tie-test
  **tied** (z=0.83, p=0.40); baseline top violation `fingertips_never_approached` (4/4 seeds).
- **Part 2** tagged pools: 300 episodes → 224 delivered → **7,886 sustained-contact states / 30,400
  delivery-completion states**; six candidates above.
- **Part 3**: best = mix_25, POSITIVE (coverage_up ✓, no_bad ✓, headline_preserved ✓, ft_dom tied with mean ≥
  base).
- **Part 4** (Branch E gate OPEN — a candidate was POSITIVE): bounded-θ option ES refinement ran, improved over
  scripted (obj 1.43→2.42); a fresh imitation round from the refined option is the documented next step (not run —
  no uncontrolled sweep).
- **guards**: tensor-contract **PASS**, policy-provenance **PASS**, actor md5 `edf4fe81…`, v2b certified
  delivers=True/25.404. **verdict: POSITIVE.**

## Honest caveats

- **mix_25's ft_dom gain is not statistically significant** — it is *tied* (preserved), with the mean marginally
  up. The POSITIVE rests on **ft_dom preserved + large, robust monitor_score/sustained-PUSH gains + zero exploit**.
  A purist could read mix_25 as the strongest PROMISING_TRADEOFF; since ft_dom did not go *down* (tied, mean up),
  POSITIVE is the honest call under the stated gate.
- **Single training seed per mixture.** Multi-seed here is on *evaluation*; each mixture was trained once (seed 0).
  The remaining rigor step is **training-seed robustness** (N training seeds × the best mixes) — the kato15 GPU
  follow-up, on a safe separate directory, not this Mac run.
- The tuned-option-expert row shows `NaN` for ft_dom/exploit — expected: ft_dom (fingertip-*dominant*) is not
  computed for the scripted stateful expert; its reference metrics are monitor_pass 0.735, monitor_score 0.501,
  sustained-PUSH 1.76/ep (all valid).

## Guards / stop rules honored

No scalar TD3/SAC/CQL, no per-step motor residual, no vector actor smoke, no reward change, no
monitor-as-reward (SearchObjective separate from the frozen TaskMonitor verifier). No CORE.YAML edit. A deployable
checkpoint was saved **only because** the gate was POSITIVE. No follow-up sweeps launched.

## Artifacts

`results.json` (all mean±std + tie-tests + provenance) · `demo_mix_grid.png` (ft_dom±std / monitor_score /
sustained-PUSH by candidate) · `best_candidate.gif` (mix_25) · `tuned_option_expert.gif` · `run.log` ·
deployable `mix_25_msdm.pt` (md5 `c9172989ff96cde8c0bd5b9578191b98`).

## Code (non-core, tested, ruff-clean)

- `hymeko_rl/train/demo_mix.py` — tagged pools (sustained vs delivery-completion) + ratio mixing + curriculum.
- `hymeko_rl/eval/multiseed.py` — multi-seed aggregation + scipy 2-proportion tie-test.
- `hymeko_rl/experiments/exp_option_msdm.py` — the gated driver (Part 1–4; reuses θ*, the DAgger loop, ledgers,
  audit, `measure_policy`).
- `hymeko_rl/eval/push_audit.py` — added `sustained_windows_raw` (shared window definition).
- `hymeko_rl/experiments/exp_vector_retest.py` — `measure_policy` gained a `seed0` param (multi-seed eval).
- Tests: `test_demo_mix.py`, `test_multiseed.py` (11 new) + prior (16 pass total for the touched set). ruff clean.
