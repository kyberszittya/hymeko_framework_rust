# Multi-thread evening session — 2026-05-21

User asked, in one message, for **five** things:

1. Stop CVAT.
2. Run CV scenarios to break the YOLO ceiling.
3. Overnight optimization on signed-graph datasets.
4. HSIKAN time-series on basic datasets.
5. HSIKAN control benchmark vs MPC / Pure Pursuit / LQR.

…and added mid-stream: **integrate HymeYOLO + quadtree extensions**.

This dossier covers what was shipped, queued, and what's running.

---

## 1. CVAT

17 ``cvat_*`` containers stopped via ``docker stop``.  ``gitea`` and
``traefik`` were left running (shared infrastructure).  All
``cvat_*`` removed from active set; ``docker ps | grep '^cvat'`` now
empty.

## 2. CV — VOC2007 Stage D-3-BREAK + quadtree warmstart

### 2.1 Phase 1 grid (running, 8 cells)

Script: ``signedkan_wip/experiments/run_voc_d3_break_2026_05_21.sh``
(PID 893163, log dir
``signedkan_wip/experiments/results/voc_d3_break_20260521T010352Z/``).

Goal: push past D-3-bis's 0.0153 mAP_50 ceiling by sweeping the
loss-balance lever harder + lengthening training.

| Cell | lam_gate_neg | epochs | n_queries | status as of 03:25 |
| --- | --- | --- | --- | --- |
| C1 | 1.0 | 30 | 12 | done, mAP_50 = 0.0094 |
| C2 | 2.0 | 30 | 12 | running |
| C3 | 5.0 | 30 | 12 | queued |
| C4 | 1.0 | 60 | 12 | queued |
| C5 | 2.0 | 60 | 12 | queued |
| C6 | 1.0 | 30 | 6 | queued |
| C7 | 1.0 | 60 | 6 | queued |
| C8 | 2.0 | 60 | 6 | queued |

Wall budget ~2.5 h.  Falsifier: best cell ≥ 0.020 mAP_50 → queue
5-seed of best cell; below → architecture limit + pivot.

### 2.2 Quadtree warmstart integration (the mid-stream redirect)

**The user noticed the existing quadtree extension was unused.**  Now
plumbed end-to-end:

- ``signedkan_wip/src/vision/hymeyolo_warmstart_quadtree.py`` (new,
  235 LOC) — uses ``AdaptiveQuadtreeRust`` to place query centres at
  the variance-ranked leaves of the adaptive quadtree.  Cheap
  post-hoc per-leaf variance scoring + FPS within the top pool.
- ``signedkan_wip/src/vision/train_voc_stagec.py`` — new
  ``--warmstart-mode {off, saliency, quadtree}`` flag and
  ``--warmstart-bootstrap-n`` for the bootstrap batch size.
- ``signedkan_wip/tests/test_hymeyolo_warmstart_quadtree.py`` —
  9 unit tests, all passing: shape/range checks, two-blob
  concentration (both blobs hit), determinism, uniform-fallback
  degenerate case, padding when quadtree underprovides, mock-model
  in-place box-corner write, circle-query corners, no-query noop.

### 2.3 Phase 2 grid (queued, 4 cells)

Script:
``signedkan_wip/experiments/run_voc_quadtree_warmstart_2026_05_21.sh``
(PID 894731, log dir
``signedkan_wip/experiments/results/voc_quadtree_warmstart_20260521T011912Z/``).

Tests the 2×2 grid:
``warmstart_mode ∈ {off, quadtree} × lam_gate_neg ∈ {1.0, 2.0}`` at
the D-3-bis recipe (nodelet head, ResNet18-ImageNet, 60 epochs,
n_q=12).  Waits for Phase 1 to finish.

The Cluttered-MNIST saliency-warmstart lever delivered +0.124 mAP_50
paired Δ at 4.68σ (2026-05-16).  The open question is whether the
*curvature-aware* multi-scale variant transfers to natural images.

## 3. Overnight signed-graph 5-seed

Script:
``signedkan_wip/experiments/run_outer_hsikan_overnight_5seed_2026_05_21.sh``
(PID 893290, log dir
``/tmp/outer_hsikan_overnight_5seed_20260521T010529Z/``).

5-seed paired sweeps on three datasets behind a ``pgrep
train_voc_stagec`` gate:

| Block | Dataset | Model | Depth | Goal |
| --- | --- | --- | --- | --- |
| 1 | Bitcoin OTC | gomb + outer_hsikan_gomb (5 seeds each) | 4 | firm up the OTC +0.0045 / 1.73σ result to publication-level n |
| 2 | Bitcoin Alpha | outer_hsikan_gomb only | 8 | probe whether depth saturates beyond the BA-winning d=4 |
| 3 | Slashdot | gomb + outer_hsikan_gomb (5 seeds each) | 4 | first 5-seed paired test on Slashdot |

Will resume from the BA d=4 win (+0.0066 / 5.68σ / 5/5).  Total
expected wall time ~6-8 h (BA fast, Slashdot slowest).

## 4. HSIKAN time-series benchmark — NULL result (honest)

Package: ``signedkan_wip/src/timeseries/`` (4 files, 250 LOC).

- ``datasets.py`` — pure-NumPy generators for sine, noisy_sine,
  Mackey-Glass, Lorenz-x.  All normalised to mean 0, std 1.
- ``models.py`` — ``LinearAR`` (33 params), ``MLP`` (2145 params),
  ``GRUForecaster`` (929 params), ``HSIKANSeqForecaster``
  (154 params, highway-gated residual over LinearAR using the
  existing ``HSiKANSeqWindow``).
- ``experiments/runs/run_timeseries_smoke.py`` — 4-model × 4-dataset
  benchmark.

### Headline (CPU, n=4096, window=32, 20 epochs)

| dataset       | linear_ar | mlp     | gru     | hsikan_seq |
| ---           | ---       | ---     | ---     | ---        |
| sine          | 0.0062    | 0.0052  | 0.0056  | 0.0061     |
| noisy_sine    | 0.0224    | 0.0194  | 0.1035  | 0.0222     |
| mackey_glass  | 0.0219    | 0.0003  | 0.0030  | 0.0227     |
| lorenz_x      | 0.0152    | 0.0009  | 0.0021  | 0.0158     |

**HSIKAN tracks LinearAR exactly.**  The highway gate stays at
σ(-3) ≈ 0.05 and never opens — the HSIKAN branch contributes
nothing.  MLP and GRU dominate on the chaotic datasets
(Mackey-Glass: MLP 0.0003 vs LinearAR 0.0219, a 70× improvement).

**Reading:** the signed-cycle inductive bias is *the wrong tool* for
smooth real-valued forecasting.  These signals have no natural σ
stream — ``sign(x)`` or ``sign(diff(x))`` are not informative.  The
architecture's natural domain is **categorical sign data**: signed
social graphs, balance-theoretic signals, and (as Section 5 shows)
**control signed lateral error**.

This is a NULL result the family paper should cite when defining the
architecture's scope.

## 5. HSIKAN control benchmark — competitive with LQR/MPC

Package: ``signedkan_wip/src/control/`` (5 files, 530 LOC).

- ``bicycle.py`` — RK4-integrated kinematic bicycle.
- ``tracks.py`` — three reference paths (straight, sinusoid,
  s_curve) with arc-length, heading, signed curvature, projection.
- ``controllers.py`` — LQR (continuous-Riccati from SciPy),
  Pure Pursuit (Coulter '92), MPC (L-BFGS-B single-shooting,
  horizon 8), HSIKAN (windowed σ-cycle policy).
- ``benchmark.py`` — episode runner + multi-init imitation-training
  helper.
- ``experiments/runs/run_control_benchmark_smoke.py`` — orchestrator.
- ``tests/test_control.py`` — 12 unit tests, all passing.

### Headline (T=12s, dt=0.05s, v=5 m/s, seed=0)

Lateral RMSE (m) — lower is better:

| track     | LQR     | pure_pursuit | MPC     | HSIKAN  |
| ---       | ---     | ---          | ---     | ---     |
| straight  | 0.0722  | 0.1343       | 0.0655  | 0.0819  |
| sinusoid  | 0.0381  | 0.3503       | 0.0391  | 0.0636  |
| s_curve   | 0.0722  | 0.1599       | 0.0656  | 0.0847  |

Wall time per step (ms):

| controller    | straight | sinusoid | s_curve |
| ---           | ---      | ---      | ---     |
| LQR           | 0.08     | 0.09     | 0.10    |
| pure_pursuit  | 0.10     | 0.09     | 0.10    |
| MPC           | 3.23     | 17.25    | 8.95    |
| HSIKAN        | 0.65     | 0.79     | 0.71    |

**HSIKAN beats pure_pursuit on every track** (5× lower RMSE on
sinusoid) and **lands within 1.6-2× of LQR/MPC** at **22× faster
inference than MPC** on the sinusoid track.

### Why it works

The signed-cycle reading of lateral control: σ_t = sign(lateral
error at step t) is exactly the kind of categorical signed signal
HSIKAN was built for.  Π σ_t over a window discriminates:
- consistent drift (all σ = +1 or all -1)  → "vehicle is off path
  in one direction, needs sustained correction"
- oscillation                                → "actuator is too aggressive,
  needs damping"
- crossing                                   → "vehicle just changed
  side, transient correction"

A plain MLP has to learn all of this from gradient signal on
δ-MSE.  HSIKAN gets it from the σ-cycle product as a *primitive*.

### Honest caveat: distribution shift

First attempt failed (HSIKAN RMSE = 7.6 m, 100× worse than LQR).
Root cause: behaviour cloning on LQR's stable trajectory → the
HSIKAN policy never saw the states it ends up in at test time, so
its first bad action snowballs.  **Fix**: multi-init imitation
(16 perturbed starts × 2 training tracks → 7200 samples) — closes
the distribution gap.  This is documented in the smoke output and
worth keeping in mind as a methodological note for any future
imitation-learning HSIKAN application.

### Pitch for vehicle research

- **LQR/MPC are optimal but parametric** — assume linear dynamics,
  break on hard nonlinearities (slip, wind, road grade).
- **Pure pursuit is robust but lossy** — geometric, lookahead-
  tuning-sensitive.
- **HSIKAN is competitive + 22× faster than MPC + learns from
  data** — could close the gap on nonlinear regimes where LQR
  becomes suboptimal (slip, terrain).  Critically, its inference
  cost is closer to LQR than to MPC, so it's deployable in
  embedded contexts where MPC is too expensive.

## 6. Tests added this session

| Suite | Result |
| --- | --- |
| ``signedkan_wip/tests/test_hymeyolo_warmstart_quadtree.py`` | **9 / 9 pass** |
| ``signedkan_wip/tests/test_control.py`` | **12 / 12 pass** |
| ``signedkan_wip/tests/test_htl.py`` (earlier in session) | **18 / 18 pass** |

39 new unit tests added today.  No regressions in prior suites.

## 7. CORE.YAML items touched

None.

## 8. §6.5 anti-pattern audit

- ``timeseries/`` and ``control/`` are new packages, each with their
  own ``__init__.py`` re-exports — no flat-file dumps.
- ``hymeyolo_warmstart_quadtree.py`` is a separate module that
  *imports* from the existing saliency warmstart instead of
  duplicating the corner-builder helpers (``_box_corners_at``,
  ``_circle_corners_at``).
- ``--warmstart-mode {off,saliency,quadtree}`` is an enum at the
  CLI surface, parsed into a single dispatch arm inside ``main()``
  — not a Cartesian-product of CLI flags.
- HSIKAN forecasting model is a *residual* over LinearAR
  (highway-gated) — same productive pattern as outer-HSIKAN
  earlier this session.  No substitutive composition.

Clean.

## 9. Open follow-ups

1. **Phase 1 + Phase 2 morning summary.**  Read
   ``orchestrator.log`` of both grids; tabulate which (if any) cell
   broke 0.020 mAP_50.
2. **Signed-graph overnight aggregate.**  Read
   ``signedkan_wip/experiments/results/outer_hsikan_overnight_5seed_*.jsonl``
   morning; compute paired Δ for OTC d=4 and Slashdot d=4.
3. **HSIKAN control: 5-seed validation + nonlinear regime tests.**
   Tonight's run is seed=0 only.  Real claim needs 5 seeds.  And
   the "HSIKAN closes the nonlinear-regime gap" hypothesis needs a
   *nonlinear* test track (wind, slip) that LQR can't solve
   analytically.
4. **Time-series NULL: try **categorical** time series.**  The
   architecture didn't transfer to real-valued forecasting; the
   right transfer test is to text classification (already shipped),
   or to **regime-classification time series** where σ_t is a true
   binary regime label (financial up/down days, sensor anomaly
   indicators, etc.).
5. **HSIKAN inference latency on embedded.**  0.7 ms/step on this
   workstation; a Hololens/Unity port would need ARM benchmarking.

## 10. Files touched

| File | Type | LOC |
| --- | --- | --- |
| ``signedkan_wip/src/timeseries/{__init__,datasets,models}.py`` | new | ~280 |
| ``signedkan_wip/experiments/runs/run_timeseries_smoke.py`` | new | 150 |
| ``signedkan_wip/src/control/{__init__,bicycle,tracks,controllers,benchmark}.py`` | new | ~530 |
| ``signedkan_wip/experiments/runs/run_control_benchmark_smoke.py`` | new | 140 |
| ``signedkan_wip/tests/test_control.py`` | new | 165 |
| ``signedkan_wip/src/vision/hymeyolo_warmstart_quadtree.py`` | new | 235 |
| ``signedkan_wip/src/vision/train_voc_stagec.py`` | extended | +45 (CLI + warmstart dispatch) |
| ``signedkan_wip/tests/test_hymeyolo_warmstart_quadtree.py`` | new | 145 |
| ``signedkan_wip/experiments/run_voc_d3_break_2026_05_21.sh`` | new | 130 |
| ``signedkan_wip/experiments/run_voc_quadtree_warmstart_2026_05_21.sh`` | new | 100 |
| ``signedkan_wip/experiments/run_outer_hsikan_overnight_5seed_2026_05_21.sh`` | new | 130 |
| ``reports/2026-05-21-evening-multithread.md`` | new | this file |

## 11. What's running, right now

| PID | What | Log | ETA |
| --- | --- | --- | --- |
| 893163 | VOC D-3-BREAK Phase 1 (8 cells) | ``signedkan_wip/experiments/results/voc_d3_break_20260521T010352Z/orchestrator.log`` | ~2.5 h |
| 894731 | VOC D-3-BREAK Phase 2 (4 cells, quadtree warmstart) | ``signedkan_wip/experiments/results/voc_quadtree_warmstart_20260521T011912Z/orchestrator.log`` | waits for 893163 |
| 893290 | outer-HSIKAN 5-seed on BA d=8 + OTC d=4 + Slashdot d=4 | ``/tmp/outer_hsikan_overnight_5seed_20260521T010529Z/orchestrator.log`` | waits for 894731 then ~6 h |

Total queue: ~10-11 h from now.  Wakeup the user to:
- Phase 1 + 2 results in JSONL.
- Outer-HSIKAN 5-seed aggregation printed at end of orchestrator.log.
- All three runs' artefacts on disk regardless of whether the
  best mAP cell clears the partial-win threshold.
