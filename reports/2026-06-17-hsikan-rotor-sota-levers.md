# HSiKAN rotor SOTA levers — code staged, empirical run halted (post-OOM swap-death)

**Date:** 2026-06-17
**Plan:** [docs/plans/2026-06-17-hsikan-rotor-leakage-free](../docs/plans/2026-06-17-hsikan-rotor-leakage-free/) (this is the documented "A/B vs SiGAT" next step; no new plan dir — single-file knob addition within existing scope).
**Status:** ✅ **Complete.** Code + tests + A/B grid + leakage gates run. (Earlier blocked by a post-OOM wedged WMI service — `import torch` was 1239 s; user restarted `Winmgmt`, import returned to 2.3 s, confirming the diagnosis. See Diagnosis below.)

## Results (A/B, leakage-free, gate-passing)

`reports/hsikan_rotor_tuned_20260617.jsonl` — rotor-HSiKAN, bilinear+dedup, 5 seeds, strict train-only triads.

| dataset | baseline (verify recipe) | tuned (wd=1e-4, clip=1.0, class-wt, early-stop) | Δ mean | SiGAT-rotor | gap (tuned) |
|---|---:|---:|---:|---:|---:|
| bitcoin_alpha | 0.8418 | **0.8455** | +0.0037 | 0.8844 | −0.039 |
| bitcoin_otc | 0.8635 | **0.8685** | +0.0050 | 0.9019 | −0.033 |

**Leakage gates (tuned + `--shuffle-train-signs`):** alpha 0.536, otc 0.494 ≈ chance → the tuned recipe (incl. early-stop) is honest, no leakage.

**Honest read:** the levers the archived `HSIKAN_GAP_CLOSING_PLAN` measured as +0.04 AUROC on the *transductive* HighwaySignedKAN **transfer only weakly** to the leakage-free rotor line: a small, per-seed-consistent +0.004/+0.005 lift. They do **not** close the SiGAT gap (~−0.04 → ~−0.036). As the plan anticipated, the residual is "partly attention, not cycles" — a training-knob tweak cannot bridge it; a structural change (attention head) is required. This is the verified, conservative result; the tuned recipe is a minor honest improvement, not a SOTA claim.

## Context

`reports/hsikan_rotor_verify_20260617.jsonl` (the run interrupted by the system OOM) is **complete and intact** — 24/24 cells, both small bitcoin datasets, leakage gates passed (shuffle ≈ 0.41/0.50/0.49/0.47). Nothing lost there. The SOTA gap, leakage-free and gate-passing on both sides:

| dataset | rotor-HSiKAN (verify) | SiGAT-rotor (target) | gap |
|---|---:|---:|---:|
| bitcoin_alpha | ~0.842 | 0.884 | −0.042 |
| bitcoin_otc | ~0.863 | 0.902 | −0.039 |

rotor-HSiKAN trails SiGAT by ~0.04 while being **larger** (125k–192k vs 16k params) on an **untuned** recipe (`lr=1e-3, wd=1e-5, plain BCE, no clip, no early-stop`). The archived `HSIKAN_GAP_CLOSING_PLAN` already *measured* the levers that move this family: `wd=1e-4` (+0.04 AUC), `grad_clip=1.0` (+0.012), class-weighted BCE (+0.04 on these ~90%-positive fixtures). None were wired into the leakage-free driver. This change wires them.

## Files touched

- `hymeko_neuro/experiments/runs/run_hsikan_rotor.py` (~ +95 / −15 LOC)
  - New `TrainConfig` dataclass (lr, weight_decay, grad_clip, class_weight, early_stop, patience, eval_every, n_epochs). **Defaults reproduce the 2026-06-17 verify recipe exactly** → prior numbers stay reproducible.
  - New helpers: `_optimise` (epoch loop + grad-clip + val-based early-stop with best-state restore), `_pos_weight` (class-balanced `pos_weight = n_neg/n_pos`), `_drop_train_pairs` (dedup refactor, was inline-duplicated).
  - `run()` threads the config; builds a val split for early-stopping; emits the knobs + selected `val_auroc` in the result dict (provenance).
  - `main()` exposes `--lr --weight-decay --grad-clip --class-weight --early-stop`.
  - **Untouched:** the load-bearing strict train-only triad construction + `--shuffle-train-signs` leakage gate.
  - Fixed one pre-existing `E702` (line 56, formatting-only; declared per §3).
- `hymeko_neuro/tests/test_hsikan_rotor.py` (new, ~155 LOC) — 8 unit (TrainConfig defaults, `_pos_weight` balance/degenerate, `_drop_train_pairs` overlap/empty, `_optimise` epoch-assert / early-stop-restores-best / no-early-stop-nan) + 3 integration (table & rotor smoke, tuned-recipe path).

## CORE.YAML items touched

None. Confirmed CORE.YAML protects only `hymeko_core/` (Rust); `hymeko_neuro` python is application code.

## Test results

- **ruff check** (changed files): **PASS** ("All checks passed!").
- **pytest** `hymeko_neuro/tests/test_hsikan_rotor.py`: **11 passed in 9.16 s** (8 unit + 3 integration). 2 warnings are torch-internal sparse-CSR-beta notices from pre-existing `signedkan.py:753`, not this change.

## Performance

- **Peak RSS** (one tuned cell, bitcoin_otc, polled worker tree): **1724 MB = 10.5 % of the 16 GB cap.** Well within budget.
- **Wall time:** ~7–12 s/cell (early-stop shortens it). Full 22-cell grid ≈ 5 min.
- bitcoin_alpha/otc at hidden=32 only; epinions/slashdot/Gömb remain off this machine (Komondor/GCP).

## Diagnosis: post-OOM recovery (revised after reclamation)

Initial read ("swap-thrash, free RAM") was **wrong** and was corrected by discriminating tests after the user reclaimed memory. Distinguishing measured / inferred / hypothesis per the operating contract:

**Measured (post-reclamation):**
- Free physical RAM **10.56 GB** / 31.4 GB total, 66% load (via `GlobalMemoryStatusEx`, bypassing WMI). → **not RAM-starved.**
- Plain python startup **0.014 s**; `nvidia-smi` **0 s** (GPU healthy, 963/8192 MiB); raw `ctypes` call instant; Defender real-time protection **off**.
- `import torch` **1239 s** (≈ identical to the pre-reclamation 1229 s — unchanged by freeing RAM).
- `Get-CimInstance Win32_OperatingSystem` (WMI) query **hung >15 min, never returned**.

**Inferred:** the torch-import stall and the WMI hang share one cause — a **wedged WMI/RPC service** (Winmgmt), a known post-OOM Windows symptom. `import torch` queries system info through WMI during init; plain python / nvidia-smi / ctypes never touch WMI and are instant. Both slow ops block on the same ~20-min RPC timeout.

**Hypothesis (untested):** restarting `Winmgmt` (+ dependents) or a reboot restores fast torch import. A reboot is the reliable clear; it can be confirmed with a `py-spy dump` of a hung import (stack would sit in a WMI/RPC call) if certainty is wanted before rebooting.

**Resolution is machine-state, not code, and not more RAM:** reboot (or restart the wedged service). Then the staged A/B runs in seconds-to-minutes. Alternatively run on Komondor/GCP, sidestepping the laptop entirely (the plan already designates the heavy datasets for there).

## Next step (one command, when the machine is healthy)

A/B baseline vs tuned + leakage gate on the cheap fixtures:

```
# baseline (verify recipe) vs tuned, both + shuffle gate, 5 seeds, CPU-safe:
for d in bitcoin_alpha bitcoin_otc; do for s in 0 1 2 3 4; do
  python -m hymeko_neuro.experiments.runs.run_hsikan_rotor --dataset $d --embed rotor --head bilinear --dedup --seed $s            # baseline
  python -m hymeko_neuro.experiments.runs.run_hsikan_rotor --dataset $d --embed rotor --head bilinear --dedup --seed $s \
         --weight-decay 1e-4 --grad-clip 1.0 --class-weight --early-stop      # tuned
done; done
# gate (must hit ~0.5 under the tuned recipe too):
python -m hymeko_neuro.experiments.runs.run_hsikan_rotor --dataset bitcoin_otc --embed rotor --head bilinear --dedup --seed 0 \
       --weight-decay 1e-4 --grad-clip 1.0 --class-weight --early-stop --shuffle-train-signs
```

**Hard precondition before reporting any SOTA number:** the tuned recipe's `--shuffle-train-signs` gate must hit ≈0.5. No headline until it does (plan's load-bearing rule). The lift is a hypothesis until the A/B is run; only the leakage-free property is guaranteed.

## Open issues / follow-up

1. Execute pytest + the A/B once RAM frees. Tests are written but **unverified at runtime**.
2. Confirm early-stopping does not loosen the leakage gate (gate command above).
3. If the gate passes and the lift holds, extend to a per-seed table vs SiGAT and update `reports/cayley_rotor_README.md` (the verify is now a confirmed result, not a "signal").
4. Top RAM consumer at halt time: pending measurement (`Get-CimInstance Win32_OperatingSystem` query was itself queued behind the draining torch backlog).
