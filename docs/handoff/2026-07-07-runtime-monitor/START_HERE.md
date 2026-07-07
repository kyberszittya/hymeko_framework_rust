# START HERE — Runtime Monitor Subsystem (session handoff)

**For:** the next agent/chat session picking up this work.
**Created:** 2026-07-07 20:07 +09:00, branch `hymeko-neuro-migration`.
**Read this first, then the two committed reports linked at the bottom.**

---

## 0. Register (non-negotiable)

Persona is **Aiko Seto** — Japanese-teacher register: restraint, precision, brevity. No therapy-speak, no
soothing closers, no unsolicited emotional framing. **Every reply starts with a real-clock timestamp**
`[YYYY-MM-DD HH:MM TZ]` (from `Get-Date`, not guessed). Execute concrete directives directly — no preamble,
no four-option questions after a clear instruction. User = Dr. Csaba Hajdu, 25-yr systems engineer; prefer
trait/generic/OO/dataflow over flat duplication.

## 1. What this line of work is

A **reward-independent runtime monitor** system: external verifiers that judge whether the *scenario* actually
happened (approach → contact → move-toward-target → success), **independent of the reward**. Governing principle:

> The reward may guide learning, but the runtime monitor judges the scenario.

Code lives in `hymeko_rl/eval/task_monitor/`. It is reward-independent, generated from the HyMeKo task contract,
and used by scripted/BC/DAgger/RL evaluation and (planned) a Code Agent + CIP diagnostic layer.

## 2. Current state (all committed + pushed: commit `a37a873`)

`hymeko_rl/eval/task_monitor/` package:

| File | Role |
| --- | --- |
| `contract.py`, `context.py`, `root.py`, `submonitors.py`, `consistency.py`, `pipeline.py`, `provenance.py` | coin-delivery hierarchical monitor (prior sessions; `provenance.py` NOT reviewed by me) |
| `submonitors.py` → `StagnationMonitor` / `StagnationVerdict` | no-progress detector (net window displacement); **opt-in**, not in `default_submonitors()` |
| `context.py` → `SupportsDistanceSeries` (Protocol: `n` + `dist`) | lets a distance-only monitor be task-agnostic |
| `cip_export.py` → `export_cip_variables(verdict) -> CipExport` | verdict → 8 named CIP scalars + `missing` set |
| `metaworld.py` → `CoffeePushMonitor`, `DialTurnMonitor`, `MetaWorldSubmonitor[_Ctx]` | distance + wrapped-angular task templates |

Tests: `hymeko_rl/tests/{test_task_monitor,test_cip_export,test_metaworld_monitors}.py` — **59 pass**
(28 coin unchanged, 9 CIP, 22 MetaWorld). ruff/mypy clean on changed code; radon max CC 8 (B). No `CORE.YAML`
touched, no new deps.

## 3. Resume commands

```bash
git checkout hymeko-neuro-migration && git pull
python -m pytest hymeko_rl/tests/test_task_monitor.py hymeko_rl/tests/test_cip_export.py \
    hymeko_rl/tests/test_metaworld_monitors.py -p no:randomly -q      # expect: 59 passed
```

## 4. What was built this session (audit build order)

1. `StagnationMonitor` (done) — net-window (not one-sided `toward`) so it catches oscillation; emits
   `stagnation_duration` / `stagnated`. Opt-in → coin verdicts byte-unchanged.
2. CIP-export bridge (done) — 8 vars: `success_monitor_pass`, `progress_score`, `stagnation_duration`,
   `stagnated`, `forbidden_contact_count`, `clearance_min`, `phase_transition_failure`,
   `reward_progress_disagreement`. Unavailable → deterministic default + listed in `missing` (never fabricated).
3. `CoffeePushMonitor` (done) — Euclidean distance-to-target; approach/progress/success semantics.
4. `DialTurnMonitor` (done) — wrapped angular error `|wrap(angle-target)|`, overshoot detection, correct ±π
   handling. Same `StagnationMonitor` composes on its angular error via `SupportsDistanceSeries`.

## 5. What remains (do next)

- **Build #5 — DAgger monitor-success hook** (audit gap C.5): wire `TaskMonitor.evaluate_policy`'s
  `monitor_pass_rate` as the DAgger selection / early-stop metric. **Blocked by a standing constraint**
  ("do not modify DAgger") — confirm with the user before starting.
- **Gap C.2 — violation submonitors:** `clearance_min`, `forbidden_contact_count`, `phase_transition_failure`,
  `workspace_violation`, `action_saturation`. The CIP bridge already reads these opportunistically (top-level or
  any sub-verdict field of that name), so they light up with no bridge change once built. `action_saturation`
  first needs the action added to `record_trajectory`'s trajectory tensor.
- Real MetaWorld env wrapper + register these monitors alongside `TaskSpec` in `hymeko_rl/eval/tasks.py`.

## 6. Landmines / decisions already made

- **Spec bundle is gitignored.** The 6 design docs in `docs/plans/2026-07-04-hymeko-code-agent/` live only on the
  Windows working tree (`docs/plans/` is in `.gitignore`). Their substance is in the committed reports (§7). To
  sync them: `git add -f docs/plans/2026-07-04-hymeko-code-agent/` or relocate under a tracked path — ask first.
- **Working tree is very dirty** with unrelated uncommitted work (`MUJOCO_LOG.TXT`, `CLAUDE.md`, many
  `hymeko_rl/env`+`train` edits, checkpoints). Only the monitor subsystem + reports were committed. Do NOT
  `git add -A` — stage explicit paths.
- `StagnationMonitor` is intentionally **not** in `default_submonitors()` (keeps existing verdicts unchanged);
  compose it explicitly when you want stagnation reported.
- Quotas from `CLAUDE.md` still bind: plan→test→report for substantive work; timestamp every reply; oracle-certify
  before any RL launch; Fable/July-5 quarantine on that code region.

## 7. The durable records (committed, under `reports/`)

- `reports/2026-07-07-runtime-monitor-session-handoff.md` — full handoff.
- `reports/2026-07-07-runtime-monitor-spec-to-implementation-audit.md` — the spec→impl audit with per-build
  "implemented" notes (build order, gaps, what exists vs missing).
