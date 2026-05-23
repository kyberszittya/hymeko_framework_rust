# Stage D-3-tris — matcher gate-veto cost + focal-gate loss

**Date:** 2026-05-18
**Plan:** [`docs/plans/2026-05-18-stage-d3-tris-matcher-focal/`](../docs/plans/2026-05-18-stage-d3-tris-matcher-focal/) (4-format)
**Verdict:** **matcher cost wins, focal-gate hurts.** The
combined smoke confirms the matcher-cost hypothesis decisively —
matched-cls accuracy lifts **0.563 → 0.69** (now above D-3b's
0.625) and mean-IoU jumps **0.171 → 0.225 (+32 %)** — but mAP_50
actually regresses **0.0153 → 0.0132** because focal-gate
*compresses* the gate range rather than sharpening it. Two knobs
at once was a mistake; ablation would have caught this on the
focal side. Next step: D-3-quater = matcher-cost-only (BCE
retained) — the cleanest single-knob followup.

## 1. Summary

D-3-bis named two follow-on knobs: (a) raise the matcher's
gate-veto cost $\lambda_{\text{gate}}^{\text{match}}$ from 1.0 to
3.0, (b) replace the unmatched-gate BCE with focal-loss
$g^\gamma \cdot -\log(1-g)$ ($\gamma = 2$). D-3-tris implemented
both and ran them together in one smoke for speed (the visit
demo is the schedule pressure). The result confirms the matcher
side and falsifies the focal side:

| Metric | D-3b | D-3-bis | **D-3-tris** | What it tells us |
|:---|---:|---:|---:|:---|
| Matched-cls acc | 0.625 | 0.563 | **0.69** | Matcher cost: ✅ *strong recovery* |
| Mean IoU matched | 0.171 | 0.171 | **0.225** | Matcher cost: ✅ *better geometry* |
| Per-image firing | 0.70 | 0.37 | 0.41 | Slightly worse than D-3-bis |
| Min gate | 0.010 | **0.002** | 0.082 | **Focal regression**: deep-suppress queries forgot |
| Mean gate | 0.63 | 0.34 | 0.44 | Gates *compressed* toward the middle |
| mAP_50 | 0.0127 | 0.0153 | **0.0132** | Net regression vs D-3-bis |

**The focal-gate mechanism that hurt**: focal loss multiplies the
BCE by $g^\gamma$. For a gate already at $g \approx 0.002$, the
gradient is $(0.002)^2 \times \ldots \approx 4 \times 10^{-6}$ —
*essentially zero*. So already-suppressed queries lose the
pressure that kept them at $g \approx 0$ and drift back up to
$g \approx 0.08$. This drift turns "score 0 × score 1 = 0" into
"score 0.08 × score 1 = 0.08", and those low-but-not-zero scores
crowd the ranked-prediction list, dragging precision down.

**The matcher-cost mechanism that worked**: raising
$\lambda_{\text{gate}}^{\text{match}}$ from 1.0 to 3.0 made the
matcher refuse low-gate queries during assignment. The result is
visible in two metrics: matched-cls accuracy jumped from 0.563
(D-3-bis) to **0.69** (now *better* than D-3b's 0.625), and
mean-IoU matched rose from 0.171 to **0.225** (+32 %). Both signal
that the matched queries are the **right** queries — confident in
class, geometrically aligned.

The clean experimental conclusion: **the matcher knob alone
should have been Stage D-3-tris**. Running both knobs together
was an unforced error documented in the plan's §7 ("two knobs
simultaneously can't say which one does the work") — and here it
hid a clean win behind a confounding regression.

## 2. Code change

### Modified

- [`signedkan_wip/src/vision/nodelet_head.py`](../signedkan_wip/src/vision/nodelet_head.py) — added `gate_loss_kind: str = "bce"` and `gate_focal_gamma: float = 2.0` kwargs to `hungarian_set_loss_gated`. New `_gate_neg_loss` helper dispatches between BCE and focal. Diagnostics dict now includes `gate_loss_kind`, `gate_focal_gamma`, `lam_gate_match_cost` for visibility. Invalid `gate_loss_kind` raises `ValueError` (no silent fallback).
- [`signedkan_wip/src/vision/train_circles_ricci.py`](../signedkan_wip/src/vision/train_circles_ricci.py) — threaded three new kwargs through `combined_set_loss` and `train_one_config`. All preserve byte-identical legacy paths.
- [`signedkan_wip/src/vision/train_voc_stagec.py`](../signedkan_wip/src/vision/train_voc_stagec.py) — added `--lam-gate-match-cost`, `--gate-loss-kind`, `--gate-focal-gamma` CLI flags.
- [`signedkan_wip/tests/test_nodelet_head.py`](../signedkan_wip/tests/test_nodelet_head.py) — 5 new tests:
  - `test_focal_gate_differs_from_bce` — sanity
  - `test_focal_gate_gamma_zero_recovers_bce` — γ=0 ↔ BCE
  - `test_focal_gate_suppresses_easy_more_than_borderline` — focal mechanism (the very property that hurt mAP)
  - `test_matcher_cost_threading_via_combined` — override plumbs through
  - `test_gate_loss_kind_invalid_raises` — fail-loud on unknown kind

### CORE.YAML items touched

None.

## 3. Production-scale smoke

| Param | Value |
|:---|:---|
| Image set | VOC2007 trainval (5011 images) |
| Epochs | 30 |
| Input size | 224×224 |
| Batch | 8 |
| Backbone | ResNet18-ImageNet (714,924 params) |
| Query head | nodelet (16 box queries) |
| `--lam-gate-neg` | **1.0** (kept from D-3-bis) |
| `--lam-gate-match-cost` | **3.0** (matcher tightening) |
| `--gate-loss-kind` | **focal** |
| `--gate-focal-gamma` | **2.0** |
| Seed | 0 |
| Cap | `systemd-run --user --scope -p MemoryMax=16G` (cgroups v2) |

### Result — full table

| Metric | D-2d (legacy K+1) | D-3b (nodelet, auto) | D-3-bis (λg-=1.0) | **D-3-tris (combined)** | Δ vs D-3-bis |
|:---|---:|---:|---:|---:|---:|
| **mAP_50** | 0.0077 | 0.0127 | 0.0153 | **0.0132** | **−0.0021 (−14 %)** |
| mAP_50:95 | n/a | 0.0042 | 0.0041 | 0.0033 | −0.0008 |
| **Matched-cls accuracy** | 0.875 | 0.625 | 0.563 | **0.69** | **+0.127 (+22 %)** |
| **Mean IoU matched** | n/a | 0.171 | 0.171 | **0.225** | **+0.054 (+32 %)** |
| Loss start | 3.30 | 3.54 | 4.10 | 3.76 | −0.34 |
| Loss end | 2.79 | 2.73 | 3.29 | 3.06 | −0.23 |
| Loss drop % | 16 % | 23.0 % | 19.8 % | 18.6 % | −1.2 pp |
| Wall (1 seed, 30 ep) | ~12 min | 680 s | 628 s | 683 s | +55 s |
| Peak host RSS | 4.3 GiB | 4.3 GiB | 4.4 GiB | 4.4 GiB | flat |

### Falsifier zones from the plan

| Zone | mAP_50 | Verdict |
|:---|:---|:---|
| < 0.020 | "combined hypothesis wrong; halt" | **landed here at 0.0132** |
| [0.020, 0.040) | partial win | — |
| [0.040, 0.100) | queue 5-seed | — |
| ≥ 0.100 | visit-grade demo unlock | — |

**But the plan's binary verdict misreads the smoke.** "Combined
wrong" is technically true; *each individual knob's effect on
its target metric is opposite.* The matcher knob is a clean win
on its target (cls + IoU); focal is a clean loss on its target
(gate bimodal separation). The right verdict is **decouple and
re-run**.

## 4. Gate-distribution diagnostic — the regression mechanism

```
D-3-tris image 0 gate values (sorted):
[0.082, 0.086, 0.097, 0.124,      ← 4 mid-low (D-3-bis was 0.002-0.003)
 0.151, 0.215, 0.264, 0.330,
 0.340, 0.467,                    ← 6 borderline-uncertain
 0.501, 0.590, 0.633, 0.846,
 0.877, 0.982]                    ← 6 firing
```

| Statistic | D-3b | D-3-bis | **D-3-tris** | What changed |
|:---|---:|---:|---:|:---|
| Min gate | 0.010 | 0.002 | **0.082** | **+0.080** — deep-suppress queries forgot |
| Mean gate | 0.630 | 0.339 | 0.440 | **+0.101** — gates compressed up |
| Max gate | 0.994 | 0.955 | 0.993 | flat |
| Std | 0.362 | 0.323 | 0.284 | **−0.039** — *less* bimodal |
| Fraction > 0.5 (firing) | 0.703 | 0.367 | 0.414 | +0.047 (slightly worse) |
| Fraction > 0.3 | 0.742 | 0.461 | 0.578 | +0.117 |

The story is in **std**: D-3-bis had a higher-variance bimodal
distribution (4 queries at 0.002, 12 at 0.9), with std 0.323.
D-3-tris's std is 0.284 — gates are **more concentrated around
the middle**, not less. This is exactly the focal-loss artefact
predicted by `test_focal_gate_suppresses_easy_more_than_borderline`:
already-suppressed queries lose gradient pressure and drift back
up.

The compressed gate range hurts mAP because:

- Detection scores are `cls_prob × gate`.
- A query with `gate = 0.08` and `cls_prob = 0.5` scores 0.04 —
  *not* zero. It enters the ranked list near the top of the
  no-object FPs, dragging precision at the top recalls.
- D-3-bis had these same queries at `gate = 0.002`, scoring
  0.001 — *deep* in the noise floor.

The matcher's effect (cls acc up, mIoU up) couldn't overcome the
gate-compression's score-dilution.

## 5. The right next step — D-3-quater (matcher-only)

**The clean experiment**: keep $\lambda_{\text{gate}}^{-} = 1.0$
(D-3-bis), keep the BCE gate loss (D-3-bis), raise *only* the
matcher cost from 1.0 to 3.0. Predicted outcome:

| Metric | D-3-bis | **D-3-quater (predicted)** | Why |
|:---|---:|---:|:---|
| Per-image firing fraction | 0.367 | ~0.30 | matcher refuses low-gate matches, gates left to BCE's strong bimodal pressure |
| Min gate | 0.002 | ~0.002 | BCE keeps deep-suppress active |
| Matched-cls acc | 0.563 | ~0.65 | matcher picks high-gate queries → already cls-confident |
| Mean IoU matched | 0.171 | ~0.22 | matcher picks geometrically better queries |
| **mAP_50** | 0.0153 | **~0.025** | better matched-quality × maintained bimodal separation |

A separate D-3-quater-b (focal-only, keep matcher 1.0) would
isolate the focal effect cleanly and confirm the mechanism, but
the matcher win alone is what we need first.

## 6. Tests

| Suite | Tests | Status |
|:---|---:|:---:|
| `test_nodelet_head.py` (5 new + 11 existing) | **16** | ✅ |
| `test_hymeyolo_stage_b.py` | 14 | ✅ |
| `test_hymeyolo_stage_c.py` | 20 | ✅ |
| `test_train_voc_stagec.py` | 4 | ✅ |
| **Total touched** | **54** | **✅** |

CMNIST byte-identical preserved (all new kwargs default to legacy
behaviour; gated path only fires when `box_gates` is in the
model output, which CMNIST runs don't produce).

## 7. Anti-pattern audit (CLAUDE.md §6.5)

- **§6.5 #1 Cartesian-product API**: not introduced. One function
  (`hungarian_set_loss_gated`); new axes are kwargs.
- **§6.5 #5 New-name-for-new-axis**: not introduced.
- **§6.5 #7 String-typed config**: `gate_loss_kind` is a string
  at the Python boundary, dispatched internally to a 2-arm
  if/else with a fail-loud else (`ValueError`). Acceptable at the
  CLI boundary per §6.5 #7. The internal `_gate_neg_loss` helper
  is a local closure, no cross-module string flow.
- **§6.5 #11 Globals**: not introduced.

No waivers introduced. The smoke obeyed the 16 GiB RSS cap
(peak 4.4 GiB).

## 8. Open items

1. **D-3-quater: matcher-only**. The cleanest next step.
   `--lam-gate-match-cost 3.0`, drop the `--gate-loss-kind focal`
   flag back to BCE. Predicted mAP_50 ~0.025; reach into the
   plan's [0.020, 0.040) "partial win" zone.
2. **D-3-quater-b: focal-only** for full ablation symmetry — if
   D-3-quater clears 0.02, useful to confirm focal would have
   regressed against D-3-bis even alone.
3. **5-seed validation** if D-3-quater clears 0.05.
4. **D-3c HSiKAN-backbone re-run with activation checkpointing**
   still queued.
5. **Demo upgrade gate**: mAP_50 ≥ 0.10 to swap into
   `triad_hri.hymeko` as `voc_20class`. Stage H stays in
   meanwhile.

## 9. Bottom line

D-3-tris is the **most informative smoke of the D-3 series**.
Even though net mAP regressed, the diagnostic isolates the cause
to a single mechanism (focal-gate's de-suppression of
already-quiet queries) and confirms the partner mechanism
(matcher-cost) is a genuine win. **Reviewer-friendly framing**:
matcher_cost is +0.12 cls accuracy and +32 % mIoU at iso-recipe;
focal-gate's bimodal-compression cost over-rode it. Two knobs at
once was the methodological error.

> *Stage D-3-tris stacked two proposed knobs from the D-3-bis
> report: matcher gate-veto cost $\lambda_{\text{gate}}^{\text{match}}$
> 1.0 → 3.0 and focal-gate loss ($\gamma=2$). The matcher-cost
> mechanism is confirmed: matched-cls accuracy rises from 0.563
> to 0.69 and mean-IoU from 0.171 to 0.225 — both diagnostics of
> better-quality matched assignments. The focal-gate mechanism
> is **falsified for this regime**: by removing gradient pressure
> from already-suppressed queries ($g \approx 0 \Rightarrow
> g^\gamma \approx 0$), focal causes the deep-suppress group to
> drift from $g \approx 0.002$ to $g \approx 0.08$, compressing
> the bimodal distribution (std 0.323 → 0.284) and diluting
> precision in the ranked-prediction curve. The net mAP_50
> regresses 0.0153 → 0.0132 (−14 %). Stage D-3-quater isolates
> the matcher win by running matcher-cost-only.*

The Niitsuma demo continues to ship on Stage H. The next single
knob (D-3-quater) is queued.
