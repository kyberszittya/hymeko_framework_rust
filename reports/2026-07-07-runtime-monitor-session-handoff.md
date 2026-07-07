# Session Handoff — Runtime Monitor Subsystem

**Date:** 2026-07-07
**Branch:** `hymeko-neuro-migration`
**Author:** Aiko (agent), for Dr. Csaba Hajdu
**Purpose:** hand off this session's runtime-monitor work so it can be continued on another machine (Mac).

---

## 1. What this session did

Five deliverables, in order. Each is reward-independent: a monitor observes a rollout and judges the *scenario*,
never the reward.

1. **Spec bundle** — `docs/plans/2026-07-04-hymeko-code-agent/` (6 files: `runtime_success_monitor.md` (central),
   `hymeko_to_cip.md`, `metaworld_task_descriptions.md`, `dagger_for_hymeko_tasks.md`, `hymeko_code_agent.md`,
   `README.md`). Defines the *reward-independent runtime monitor* concept across the HyMeKo–CIP–MetaWorld–CodeAgent
   bundle. **Note:** `docs/plans/` is gitignored (see §5) — this bundle is local to the Windows working tree.
2. **Spec→implementation audit** — `reports/2026-07-07-runtime-monitor-spec-to-implementation-audit.md`. Maps the
   spec onto the existing `hymeko_rl/eval/task_monitor/` package; identifies what exists / is partial / is missing;
   proposes a minimal build order. Items #1–#4 below were then implemented and the audit updated with per-item notes.
3. **Build #1 — `StagnationMonitor`** (`submonitors.py`): reward-independent no-progress detector over the
   net window displacement of the distance-to-target series; opt-in (not in `default_submonitors()`), so existing
   coin verdicts are byte-unchanged. Emits `stagnation_duration` / `stagnated`.
4. **Build #2 — CIP-export bridge** (`cip_export.py`): `export_cip_variables(verdict) -> CipExport` converts a
   monitor verdict into 8 named CIP scalar variables with an explicit `missing` set (no fabricated measurements).
5. **Build #3/#4 — MetaWorld templates** (`metaworld.py`): `CoffeePushMonitor` (Euclidean distance) and
   `DialTurnMonitor` (wrapped angular error, with overshoot). Both reuse the same `SubVerdict` / generic Strategy /
   Composite style and feed `export_cip_variables` unchanged; the single `StagnationMonitor` composes on both via
   the `SupportsDistanceSeries` protocol. Proves the framework covers distance-based AND angular task progress.

## 2. State of the code

`hymeko_rl/eval/task_monitor/` package (all committed this session — it was untracked before):

| File | Role | Touched this session |
| --- | --- | --- |
| `contract.py`, `context.py`, `root.py`, `submonitors.py`, `consistency.py`, `pipeline.py`, `provenance.py` | coin-delivery hierarchical monitor (prior sessions) | `context.py` (+`SupportsDistanceSeries`), `submonitors.py` (+`StagnationMonitor`, widened to the protocol) |
| `cip_export.py` | verdict → CIP variables | **new** |
| `metaworld.py` | coffee-push + dial-turn templates, `MetaWorldSubmonitor[_Ctx]` generic base | **new** |
| `__init__.py` | exports | updated |

Tests (all committed): `hymeko_rl/tests/{test_task_monitor,test_cip_export,test_metaworld_monitors}.py`.

## 3. Test + gate status (2026-07-07)

- `pytest -p no:randomly test_task_monitor.py test_cip_export.py test_metaworld_monitors.py` → **59 passed**.
  - 28 coin monitor (unchanged across all four builds — no regression),
  - 9 CIP export, 22 MetaWorld (10 coffee-push + 12 dial-turn).
- `ruff check` clean; `mypy` clean on all changed code (the 8 reported errors are pre-existing in untouched
  files — `mujoco` stubs, `env/reward.py`); `radon` max cyclomatic complexity = 8 (B), under the gate.
- No `CORE.YAML` items touched. No new dependencies.

## 4. What remains (not done this session)

- **Build #5 — DAgger monitor-success hook** (audit gap C.5): wire `TaskMonitor.evaluate_policy`'s
  `monitor_pass_rate` as the DAgger selection / early-stop metric. **Deliberately not started** — the standing
  constraint this session was "do not modify DAgger."
- **Audit gap C.2 — violation submonitors**: `clearance_min`, `forbidden_contact_count`,
  `phase_transition_failure`, `workspace_violation`, `action_saturation`. The CIP bridge already reads these
  opportunistically (default+missing until they exist). `action_saturation` is blocked on adding the action to the
  monitor trajectory tensor (`record_trajectory`) — also deliberately deferred this session.
- Real MetaWorld env wrapper / registering these monitors alongside `TaskSpec` in `tasks.py` — future.

## 5. How to continue on the Mac

1. `git fetch && git checkout hymeko-neuro-migration && git pull` (this session's commit is the tip; the branch
   now tracks `origin/hymeko-neuro-migration`).
2. Sanity: `python -m pytest hymeko_rl/tests/test_task_monitor.py hymeko_rl/tests/test_cip_export.py
   hymeko_rl/tests/test_metaworld_monitors.py -p no:randomly -q` → expect 59 passed.
3. Next task is build #5 (DAgger monitor-success hook), if/when the DAgger-freeze is lifted.

**Caveat — the spec bundle is local-only.** `docs/plans/` is gitignored (`.gitignore:100`), matching the repo
convention that `reports/` is the durable record and `docs/plans/` is scratch. So the 6 spec-bundle markdown files
did **not** travel to the Mac in this commit. This handoff report + the audit report (both under `reports/`, both
committed) capture the substance. If the bundle itself should be synced, either `git add -f
docs/plans/2026-07-04-hymeko-code-agent/` (overrides the ignore) or relocate it under a tracked path — operator's
call; not done unilaterally.

**Caveat — the rest of the working tree.** Only the runtime-monitor subsystem + these reports were committed. The
Windows working tree has substantial unrelated uncommitted work (checkpoints, `MUJOCO_LOG.TXT`, modified `CLAUDE.md`
and many `hymeko_rl/env`/`train` files, etc.) that is **not** this session's and was **not** committed. It stays on
the Windows machine for the operator to handle.
