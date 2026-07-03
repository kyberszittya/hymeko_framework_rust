# `compute_detection_metrics` — StopIteration on parameter-less model

**Date:** 2026-06-16 · **Slug:** compute-detection-metrics-device-fix · **Type:** pre-existing bug fix (single-file local)

## Summary

While verifying the DETR overfit-fix change (`reports/2026-06-16-detr-overfit-fix.md`)
against every caller of the shared `hungarian_set_loss`, a full caller-test sweep surfaced
**4 failing tests** in `test_hymeyolo_ricci_scale.py::test_compute_detection_metrics_*`.
Attribution check (`git stash` of the DETR change → the 4 tests fail identically on the
clean tree) confirmed this is a **pre-existing bug, not a regression from the DETR work**.

`compute_detection_metrics` ([train_circles_ricci.py:417](../hymeko_neuro/experiments/vision/train_circles_ricci.py#L417))
read the compute device with `next(model.parameters()).device`. For a **parameter-less
model** — a fixed/stub predictor such as the tests' `_FixedPredictionModel`, which returns
hard-coded predictions and registers no `nn.Parameter` — `next(...)` raises
`StopIteration`, crashing the metric.

**Fix:** fall back to the input tensor's own device when the model exposes no parameters.
`next(model.parameters(), None)` (sentinel default, no exception); if `None`, use
`X.device` — which makes the subsequent `X[s:e].to(device)` a no-op, the correct behaviour
for a stub model whose predictions already live on the input's device.

```python
first_param = next(model.parameters(), None)
device = first_param.device if first_param is not None else X.device
```

## Files touched

| File | Δ | What |
|---|---|---|
| [hymeko_neuro/experiments/vision/train_circles_ricci.py](../hymeko_neuro/experiments/vision/train_circles_ricci.py) | +4 / −1 | device lookup defaults to `X.device` for parameter-less models; contract noted in comment |

No test file changed — the **4 pre-existing tests are the regression suite** (they failed
before, pass after; each would have failed against the prior implementation, satisfying §3).

## CORE.YAML items touched

**None.** `train_circles_ricci.py` is `hymeko_neuro` application code.

## Test results

- Before: `test_hymeyolo_ricci_scale.py` → **4 failed, 9 passed**.
- After: `test_hymeyolo_ricci_scale.py` → **13 passed** (5.75 s).
- Full shared-loss caller sweep (nodelet, ricci_stim_train, ricci_scale, ricci_kcycle,
  entropy, voc_stagec, stage_a3, detr) → all green (see run log; was 63 passed / 4 failed,
  now all pass).
- Ruff: the edited region (lines 414–421) introduces no new errors.

## §6.5 anti-patterns

**None.** Two-line local fix; uses the `next(iter, default)` idiom rather than a `try/except`
(no silent failure — §6.4). Pre-existing lint debt elsewhere in the file is untouched and
out of scope.

## Provenance

- Git SHA `2b80cab` (working tree dirty — this file plus the DETR-fix files and the
  in-flight head-to-head artifacts).
- Env: torch 2.12.0+cu132, Python 3.12.13, Win32, RTX 3070.
- Discovered during the `are-you-making-bugs` verification pass for the DETR change;
  attribution proven by stash-and-rerun.
