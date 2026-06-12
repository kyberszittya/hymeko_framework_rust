# Seminar demo program — plan-item 4: `latency` demo (Demo 2 wrapper)

**Date:** 2026-06-12 · **Plan:** `docs/plans/2026-06-10-seminar-demos-remaining/` §Item 4

## Summary
Wired the already-measured forward-latency benchmark into the seminar CLI as a
`latency` `SeminarDemo`, completing plan-item 4. The measurement itself (Demo 2,
`run_inference_bench.py` + `bench_to_png.py`, ≥5 repeats after warm-up) was done
2026-06-11; this change makes it a one-command room demo
(`python -m signedkan_wip.demos.seminar latency`) that **reads** the committed
`inference_bench.json`, surfaces the within-family lean(h=4)→joint(h=16) width
ratio per (dataset, device), renders the slide-18 bars, and prints the honest
framing. Inference-only — no re-measurement in the room (that stays
`run_inference_bench`).

Pure reuse of `bench_to_png.{bars_from_rows,render}` — no measurement or
plotting logic re-implemented (§6.5 #2). Registered through the `DEMOS` dict, so
seed/device/peak-RSS/16 GB cap are inherited from `DemoRunner` by construction.

## Files touched
| LOC | File | |
|---:|---|---|
| 165 | `signedkan_wip/demos/seminar/demos/latency.py` | new — `LatencyDemo` + `width_ratios` |
| 118 | `signedkan_wip/tests/test_seminar_latency.py` | new — unit + integration |
| +2 | `signedkan_wip/demos/seminar/demos/__init__.py` | register `LatencyDemo` |

## CORE.YAML items touched
**None.** Additive demo layer; reads a committed result file and reuses
non-core `bench_to_png` helpers.

## Test results
Runner: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest -p no:randomly`.

| Layer | Tests | Result | Notes |
|---|---|---|---|
| Unit (`width_ratios`) | 3 | pass | ratio = joint/lean; skips cells missing a variant; skips non-positive lean |
| Data invariant (committed json, §3) | 1 | pass | every HSiKAN cell has `iqr ≥ 0` and `worst ≥ median` |
| Integration (`DemoRunner`) | 4 | pass | registration, ratio+provenance+honest-note, PNG render, `--no-figures` writes nothing |
| **Total** | **8** | **pass** | 10.8 s |
| Regression (`test_seminar_harness.py`) | 13 | pass | registry change did not break the harness/CLI |

Coverage (§3): every new function/method (`width_ratios`, `run`,
`_populate_metrics`, `_render`) is exercised; the missing-variant and
non-positive-lean failure branches each have a dedicated test.

## Performance results
End-to-end CPU, seed 0 (committed json, both devices rendered):

| Metric | Value | Budget | Source |
|---|---|---|---|
| Peak RSS | **637.7 MB** | < 3 GB ✓ (cap 16 GB ✓) | Win32 PeakWorkingSetSize |
| Wall | 1.49 s | minutes (plan) ✓ | end-to-end incl. matplotlib |

This demo reads a json and draws two bar charts — no torch import, no model
load — hence the sub-2 s wall (vs the `link` demo's cold-import cost). Reported
latency *numbers* are the committed measurement's; this run's wall is diagnostic
(§10), not a benchmark.

## Acceptance gate
Plan-item 4: json carries median/IQR/worst for the HSiKAN cells × dataset ×
device; PNG written; the honest line printed. **PASS.**
Measured headline (mean across datasets): **CPU 3.58× · CUDA 1.14×** lean→joint
— consistent with `reports/2026-06-11-latency-bench-extension.md` (~3.5× CPU,
≈1× CUDA). Per-cell: alpha/cpu 3.91×, otc/cpu 3.24×, alpha/cuda 0.98×,
otc/cuda 1.29×. Artifacts: `demo_out/latency/latency_{cpu,cuda}.png`.

## Honest framing (printed by the demo)
- the lean→joint width gap is the message: ~3.5× CPU, ~1× CUDA;
- the deck's "11×" was the optuna_best_otc-vs-joint result — OTC-specific and
  tuple-set-driven, **not** a general width claim;
- on absolute forward latency SGCN and MLP-blind are lightest; the defensible
  claims are accuracy-per-parameter and the within-family width gap — **not**
  "faster than SGCN".

## Static analysis
- `ruff check`: clean (new files + registry).
- `mypy --strict` (latency.py): `Success: no issues found`.
- `radon cc -a -nc`: no block ranked C or worse (`run` was extracted to
  `_populate_metrics` to drop from C(11) to below the warn threshold).
- No §6.5 anti-patterns: one `SeminarDemo` registered in the dict (#1/#13); no
  algorithm/plotting logic re-implemented (#2/#9); no globals (#11); read-only
  on a single committed artifact, no v-suffix files (#13).

## Provenance
- Git SHA `af803ee` (working tree dirty — pre-existing untracked seminar work +
  the three files above).
- Python 3.12.13; ruff/mypy/radon per `tools.yaml`. OS Windows 11 Pro 26200.
- Seed 0 (fixed); deterministic (reads a frozen json, no RNG in the path).
- Input: `signedkan_wip/experiments/results/inference_bench.json` (measured
  2026-06-11, commit context in `reports/2026-06-11-latency-bench-extension.md`).

## Open issues / follow-ups
- Remaining seminar build items: **balance** (item 5, opener), **mesh+Sinkhorn**
  (item 6, the only new algorithm), **bridge** (item 7, closer). Next in plan
  order is `balance`.
- `--rerun` (re-measure in place) intentionally not added — re-measurement is a
  heavy non-room operation and stays in `run_inference_bench`; the demo reads
  the committed artifact, per the plan's "(or reads the json)".
