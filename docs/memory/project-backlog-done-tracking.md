---
name: project-backlog-done-tracking
description: Repo-root BACKLOG.md + DONE.md track open vs finished work; keep them updated
metadata: 
  node_type: memory
  type: project
  originSessionId: f2bb7dc6-6a5b-41b8-9ca1-b38bd5f36339
---

The user asked (2026-06-15) for two tracking files, now at the repo root:
`BACKLOG.md` (open work) and `DONE.md` (completed milestones). They were seeded
from the 2026-06-15 plan audit (Soma-vision + HyMeYOLO lines via two Explore
agents) + project memory.

**Why:** 150 plan dirs + ~250 reports made it hard to see what's still open after
a burst of progress; the user wanted a single scannable index.

**How to apply:** treat these as a living index, not the source of truth (that's
`docs/plans/` + `reports/`). When a backlog item lands, MOVE its line to
`DONE.md` with the report link. Add new open items to `BACKLOG.md` as they arise.
Keep entries one line: `[area] desc — plan/ref — STATUS — prio`.

**Audit verdict to remember:** the **Soma-vision line is built but empirically
falsified for vision** (2026-05-28 re-bench: RicciStim/HSiKAN/HGNN all lose to a
plain MLP; RicciStim Cluttered-MNIST 0.14 mAP < 0.23 baseline). Highest-value
open Soma items are *explain/decide*, not more architecture: (1) component
ablation to explain the loss, (2) the **cortical Brain-Score run** (infra done,
real Cichy-92 never fetched — the untested place it might actually shine).
HyMeYOLO's open frontier is **Stage D-3 nodelet head** + **Stage H person-only
VOC** (both code-drafted, smoke not run). No `soma-vision` git branch exists yet
(current: `feature/ac-hsikan`). See [[project-hero-demo]] for the parallel
structural-parity work.
