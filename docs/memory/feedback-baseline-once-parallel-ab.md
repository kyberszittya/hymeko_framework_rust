---
name: feedback-baseline-once-parallel-ab
description: "Hajdu 2026-07-04 + 2026-07-05 (STRONGER, verbatim: 'MAKE A NOTE TO NOT REMEASURE EVERYTHING FROM THE FUCKING START'): measurements are CACHED FACTS. Baseline once; NEVER re-measure the full grid; re-measure ONLY the cell whose artifact changed. reports/ + results.json ARE the cache — read them before running anything."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d8544f20-a9dc-4c9c-9180-6d1373e0ede0
---

Hajdu directive (2026-07-04): the reward-shape A/B harness (`exp_galambos_coord_ab.py`) currently runs
**baseline then treatment sequentially, every time** — wasteful (an 8h run where the baseline is re-confirmed
each A/B).

**Why:** the baseline (the farming reward, or any fixed reference) is a FIXED point — its delivery curve does not
change between A/Bs. Re-running it every experiment burns hours confirming what we already know (baseline farms →
delivery collapses to ~0, confirmed 2026-07-04 seed 0: 0.10→0.16→0.02→0.02→0.00).

**How to apply:**
1. **Compute the baseline ONCE**, multi-seed, and CACHE its curve to disk (e.g. `experiments/baseline_reference/…`
   or a `baseline_reference.json`). Future A/Bs load the cached baseline instead of re-training it.
2. **Run treatment arms (and seeds) in PARALLEL** — overlapping torch runs are now permitted (page-file resolved,
   [[project-collab-ctde-substrate-galambos]] §17); launch N background processes (one per seed/variant) rather
   than the sequential Campaign loop. Watch GPU/CPU contention.
3. Each A/B then = "1 treatment × N seeds (parallel) vs cached baseline" — minutes-to-~1h, not 8h.

This is the run-side sibling of the demo-cache already added (collect demos once, reuse across seeds). The
goal is fast iteration on the delivery metric, not re-confirming the baseline. Related: the ms-fast
`reward_oracle.certify` already screens reward shapes BEFORE any RL, so RL A/Bs should be the FEW oracle-passing
candidates, run in parallel against a cached baseline.

**GENERALIZED (Hajdu, 2026-07-05 04:07, angry and right):** this applies to ALL measurements, not just the
RL baseline. A number measured under an unchanged (code, config, seed-protocol) triple is a **cached fact**;
its cache is the disk (`reports/*.md`, `experiments/*/results.json`, run.logs). Re-measuring it burns wall
time and tokens to learn nothing. Rules:
1. **Before measuring anything, grep the caches** for an existing number under the same protocol; cite it
   instead of re-running.
2. **After a code change, re-measure ONLY the affected cell** — one seed-set spot check that must reproduce
   the cached value (identity check), NOT the full grid. On-record waste (2026-07-05): the 9-cell press
   sweep was re-run in full twice (once justified by a semantics change, once by accident via a module
   import with top-level code — never put runnable sweeps at module top level).
3. **Teacher/demonstrator anchors are measured once per (physics, controller) pair** and cited thereafter;
   the anchor is re-measured only when the env physics or the controller itself changed.
4. New experiments extend the grid; they do not re-walk it from the origin.
