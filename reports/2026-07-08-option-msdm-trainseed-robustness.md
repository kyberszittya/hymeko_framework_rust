# Training-seed robustness — option demo-mix (2026-07-08) — SUPERSEDES the Mac POSITIVE

**Run:** kato15 (RTX 6000 Ada, Linux, torch 2.11.0+cu128) · `…_option_msdm_20260708/experiments/2026_07_08_option_msdm_trainseeds/`
· wall **2376 s** (~40 min) · **8 training seeds × {0.25, 0.5, 0.75}** · 4×48 multi-seed eval · results pulled to
`experiments/2026_07_08_option_msdm_trainseeds/` on the Mac.
**Driver:** `exp_option_msdm.py --stage trainseeds --device auto` · same v2b reward, frozen TaskMonitor, ledgers.

## Verdict: **NOT_ROBUST** — the demo-mix improvement does not survive training-seed variance

The Mac single-training-seed **POSITIVE (mix_25, ft_dom 0.594)** is **superseded**. Across 8 training seeds the
demo-mix method's ft_dom is high-variance with a **median below baseline** and a **pooled ft_dom significantly
worse** than baseline on every fraction. No deployable checkpoint is saved (a non-robust method must not ship one).

Baseline (multi-seed, 4×48): ft_dom **0.5677 ± 0.063**, monitor_score 0.2521, sustained-PUSH 0.3802. Guards
**PASS/PASS**, provenance **PASS**, actor md5 `edf4fe81…`, v2b certified.

| frac | robust verdict | ft_dom median (IQR) [min,max] | pooled ft_dom tie-test vs base | verdict distribution (8 seeds) |
|---|---|---|---|---|
| 0.25 | **NOT_ROBUST** | 0.380 (IQR **0.233**) [0.167, 0.542] | **worse**, p<0.001 (0.374 vs 0.568) | 4 NEG · 3 PROMISING · 1 INC · **0 POS** |
| 0.50 | **NOT_ROBUST** | 0.445 (0.073) [0.370, 0.651] | **worse**, p=0.016 (0.476 vs 0.568) | 5 NEG · 2 POS · 1 PROMISING |
| 0.75 | **NOT_ROBUST** | 0.461 (0.148) [0.281, 0.615] | **worse**, p=0.001 (0.445 vs 0.568) | 4 NEG · 1 POS · 3 PROMISING |

### Per-training-seed ft_dom (the variance is the story)

- **frac=0.25:** [0.526, 0.167, 0.406, 0.542, 0.490, 0.276, 0.234, 0.354] — spans 0.17→0.54.
- **frac=0.50:** [0.651, 0.427, 0.562, 0.464, 0.479, 0.427, 0.370, 0.427].
- **frac=0.75:** [0.516, 0.422, 0.521, 0.615, 0.385, 0.318, 0.500, 0.281].

### monitor_score + sustained-PUSH are robustly UP (the contact gain is real; the delivery cost is the problem)

- monitor_score medians: 0.293 / 0.316 / 0.303 (all > baseline 0.252); per-seed consistently ≥ baseline.
- sustained-PUSH medians: 0.846 / 0.930 / 0.820 (all ≫ baseline 0.380); every seed well above baseline.
- **exploit / body-driven / arm-body: 0 across all 24 (frac × seed) cells** — the held-only demo pools are
  exploit-free by construction; no candidate's `no_bad` check failed. The trade-off is delivery, not exploit.

## Did train_seed=0 reproduce the Mac result? — Approximately; it was a favorable seed, not a bug

- Mac (CPU, torch 2.12): mix_25 / train_seed=0 → ft_dom **0.594**.
- kato15 (CUDA, torch 2.11): frac=0.25 / train_seed=0 → ft_dom **0.526**.

Both are **upper-band draws** (well above the 0.380 median), differing by 0.068. That gap is consistent with
BC-fine-tune non-determinism across **GPU vs CPU** reductions and **torch 2.11 vs 2.12** — the §3 RL/stochastic
carve-out (not bit-exact; rest claims on multi-seed median/IQR). The demo mix itself (numpy-seeded `mix_pools`) is
identical across platforms; only the BC optimisation trajectory differs. **Conclusion: no platform/config bug — the
Mac POSITIVE was a favorable training seed.** Both platforms' train_seed=0 land favorably; the *distribution* is
what's below baseline.

## Mechanism

Demo-mix imitation **robustly** injects sustained two-finger contact (monitor_score + sustained-PUSH up every seed)
but **robustly trades delivery** — ft_dom is high-variance with a median below baseline. Only occasional favorable
training seeds (Mac ts0, a couple of kato15 seeds) preserve ft_dom; the method is **not yet training-seed-stable**.
This is a `NEGATIVE_WITH_MECHANISM`-flavored NOT_ROBUST: the direction is behaviorally useful (contact), the
imitation *update* is unstable on the delivery metric.

## Required fields

- per-training-seed metrics, median/IQR, pooled tie-test, verdict distribution, monitor_score/sustained-PUSH
  distributions: **above**. exploit/body/arm-body: **0 everywhere**. tensor-contract **PASS**, policy-provenance
  **PASS**, actor md5 `edf4fe81…`. **No deployable checkpoint saved** (non-robust). Plot:
  `trainseed_robustness.png` (ft_dom min–max spread vs frac + monitor/sustained medians).

## Supersession

This report **supersedes** `reports/2026-07-08-option-msdm.md`'s POSITIVE headline. Corrected standing claim:
**the option-generated demo-mix imitation is NOT training-seed-robust on ft_dom** — it robustly improves contact
quality but at a high-variance delivery cost; the earlier POSITIVE was a favorable single training seed. The
deployable checkpoint `mix_25_msdm.pt` from the Mac run should **not** be treated as a robust artifact.

## Next (per the standing plan — not launched until this report is frozen)

`seed_stabilized_demo_mix_v2`: a variance-reduction ablation (control / gentler-LR / DAgger-anchored BC /
balanced-batch sampler / val-selected checkpointing / seed-selection diagnostic), ≥5 training seeds per recipe,
same eval protocol. Central question: **can favorable seeds be identified by a validation gate before test?** If
yes → checkpoint selection fixes the instability; if no → the method stays non-deployable and the next branch is
more/better demonstrations or a temporally-extended controller, **not** another imitation fine-tune. No option-RL
until a recipe is POSITIVE_ROBUST.

## Guards / discipline

No scalar TD3/SAC/CQL, no per-step residual, no reward change, no CORE edit, TaskMonitor external verifier. kato15
used a **separate directory** (existing checkout untouched), checkpoint md5-verified (`edf4fe81…`), only scipy +
matplotlib added to `.venv_stand`, smoke-gated before the run.
