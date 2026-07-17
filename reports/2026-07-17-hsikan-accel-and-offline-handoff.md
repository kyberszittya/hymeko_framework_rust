---
title: hsikan acceleration + Aibo/humanoid bounce campaign — offline handoff
date: 2026-07-17
slug: hsikan-accel-and-offline-handoff
status: pilot RUNNING (kato61, autonomous); humanoid scale-up READY (one command); code pushed
core_yaml_touched: none
---

# Offline handoff — read this when you reconnect

Everything below survives a disconnect: runs are detached (`setsid`, ppid=1), resumable (JSONL), and the code +
launchers are pushed to `origin` and on the shared katolab NFS home (visible from kato61/15/14/85).

## Boxes (2026-07-17 ~21:2x JST)

| box | GPU | arch | state |
|---|---|---|---|
| **kato61** | RTX 4090 | sm_89 (Ada) | **free → running the Aibo pilot**. Same arch as the RTX 6000 Ada, so its numbers transfer. |
| kato15 | RTX 6000 Ada | sm_89 | busy (your `~/envs/hymeko` SHL/spec-reward job, 99%) |
| kato14 | RTX 6000 Ada | sm_89 | busy (same job, 99%) |
| kato85 | GeForce GT 1030 | **sm_61** | **incompatible** with the `.venv_stand` cu128 torch (kernels are sm_75+). Do not use for torch. |

## 1. Pilot — RUNNING on kato61 (autonomous)

- **What:** Aibo × {flat,structural} × {bounce 3,8} × 3 seeds × 800k, compiled (`reduce-overhead`), **flat-first**.
- **Where:** `experiments/2026_07_17_aibo_bounce_pilot/` — `pilot.log` (live), `cells.jsonl` (results as cells finish),
  `summary.json` (written at the end).
- **Speeds (kato61):** flat ~570 steps/s (~20 min/cell), structural ~84 steps/s (~2.6 h/cell). **Total ≈ 18–20 h.**
- **Check it:**
  ```bash
  ssh kato61 'bash -lc "cd ~/hymeko_framework_rust && tail -5 experiments/2026_07_17_aibo_bounce_pilot/pilot.log; wc -l experiments/2026_07_17_aibo_bounce_pilot/cells.jsonl"'
  ```
- **If it died / to resume:** `cells.jsonl` is the resume ledger (finished cells are skipped). Relaunch:
  `ssh kato61 'bash ~/hymeko_framework_rust/pilot_run.sh'`.
- **Reading order:** the 6 **flat** cells land first (~2 h) — they answer the load-bearing question *does the Aibo
  walk at 800k, and does anti-bounce=8 help* (dx + CIP propel-edge). The 6 structural cells (structural-vs-flat)
  are only meaningful **if the flat cells show the body walking** (dx > 0, propel > 0); if flat@800k is still ~0
  like 200k was, the structural comparison is premature and the finding is "Aibo needs >800k / a real gait teacher".
- **Watch:** the first flat cell's crit/act loss was climbing (0.88→28) — could be flat-MLP Q-drift at 800k. The
  `cip_diagnose` dx/propel in `cells.jsonl` is the arbiter (walk vs diverge); the metric is divergence-guarded.
- **First result (cell 1, `aibo/flat/bounce=3/seed0/800k`, wall 1554 s):** `dx=+0.103, propel_edge=0.0,
  bounce_edge=0.541`. It **moves forward a little** (+0.10 vs ~0 at 200k) but by **bouncing** (bounce-edge 0.54),
  **not propelling** (propel 0) — exactly the bounce-domination the CIP discovery predicted. So the body is on the
  edge of walking, and the bounce=8 cells (later in the run) are the live test of whether up-weighting the
  anti-bounce term converts that bounce into forward propulsion. Hypothesis intact.

## 2. Humanoid scale-up — READY, launch when kato15/kato14 free

The pilot covers the Aibo. The humanoid is the missing body. When your job frees a box:

```bash
# on kato15 (seeds 0-2) and kato14 (seeds 3-4) — split for speed; each writes its own per-box dir
ssh kato15 'HYMEKO_SEEDS=0,1,2 bash ~/hymeko_framework_rust/scripts/kato15/scaleup_run.sh'
ssh kato14 'HYMEKO_SEEDS=3,4   bash ~/hymeko_framework_rust/scripts/kato15/scaleup_run.sh'
```

Runs humanoid × {flat,structural} × {bounce 3,8} × the given seeds × 800k, **compiled with `max-autotune`**
(the +24% lever below), flat-first. Output: `experiments/2026_07_17_humanoid_scaleup_<host>/{scaleup.log,cells.jsonl}`.
(To also bring the Aibo to 5 seeds: same idea with `bodies=("aibo_goal",)`, seeds 3,4 — edit `scaleup_launch.py`.)

## 3. hsikan acceleration — findings (the "look for a way" task)

Profiled the hsikan update (`torch.profiler`, Aibo N=33, hidden 256) and tested each lever with a parity check:

| lever | parity | speedup | verdict |
|---|---|---|---|
| **torch.compile** (`reduce-overhead`) | same policy | **5.79× update / 3.2× end-to-end** | the win — deployed |
| **`max-autotune` mode** | finite | **+24%** (1.24× over reduce-overhead, kato61) | **wired into the scale-up** (long cells only — slow cold compile) |
| fuse 3 linears → 1 (+ einsum→matmul) | ✓ 2.7e-5 | 1.03× | negligible — FLOP-bound, no work removed |
| spline 4 gathers → 1 stacked | ✓ 0.0 | 0.48× (worse) | the original 4-gather is already optimal |

**Conclusion:** hsikan is already well-tuned. It's **FLOP-bound** on the Aibo net (real matmul + spline work, not
launch overhead); compile captured the launch-bound win; micro-op fusion removes no actual work. The pilot runs at
`reduce-overhead` (already going, not worth a restart); the scale-up gets `max-autotune`. **The only further
speed lever is a model-size cut (hidden 256→128 or grid 5→3) — but that's a §6.5 #19 model change needing an A/B
(does the smaller structural actor still beat flat?), not a free speedup.** Your call if you want that A/B.

## 4. Pushed commits (durable on GitHub)

Branch `origin/integration/hymeko-main`:
- `778de44` — compiled + rate-decoupled SAC update; CIP anti-bounce reward factories + bounce campaign axis.
- `f0d6d8f` — CUDA-graph aliasing fix (found by the kato61 GPU smoke); measured 5.79×.
- (+ this commit) — `compile_mode` option (max-autotune), fine `log_every` (§3), scale-up scripts, this handoff.

Plan bundle: `docs/plans/2026-07-17-sac-compiled-update/` (gitignored). No CORE.YAML touched throughout.

## 5. When you reconnect — checklist

1. `ssh kato61 ...` (cmd in §1) → how many pilot cells done, is it alive.
2. If flat cells are down: ask me (or run) the aggregation — `cells.jsonl` → structural-vs-flat × bounce
   median/IQR + the §9 plot, and a GIF of the best Aibo policy if it walks.
3. If kato15/kato14 freed: launch the humanoid scale-up (§2).
4. Decide on the model-size A/B (§3) if you want more hsikan speed.
