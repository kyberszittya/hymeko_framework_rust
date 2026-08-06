---
campaign: Canonicalize the E-approach loader + Coin env factory out of the galambos/coin-toss experiment web
title: COIN_SOURCE_REPRODUCIBLE — the coin delivery runtime closure now contains 0 experiment-web files; E-approach loader + env factory + planar kinematics are canonical production, bit-identical, fail-loud
date: 2026-07-21
branch: refactor/canonical-e-approach-loader
commit: ae1e94cae6a0282e105b7eff853155b9684e490d
verdict: COIN_SOURCE_REPRODUCIBLE
---

# Coin E-approach loader / env-factory canonicalization — closure + verdict

**Created-at:** 2026-07-21 20:08 JST. Branch `refactor/canonical-e-approach-loader` (`ae1e94c`) off
`refactor/canonical-campaign-and-final-video` (`a3b259b`). Scope: the bounded loader/env-factory canonicalization —
**not** the wider 56-import migration.

## Problem
The coin delivery runtime obtained its E-approach policy loader and env builder from the **experiment** layer:
`exp_v3_handoff_gate._load_e` → `exp_galambos_coord_ab.make_env` + `exp_option_retest._fresh_actor`, and
`pedc_selection._env / _ctx / _load_pkl_bank / c1_config`. Both cascaded through the whole galambos/coin-toss arc, so a
fresh-clone coin closure required **29 OPTIONAL_EXPERIMENT** files (prior verdict `COIN_SOURCE_INCOMPLETE`,
`coin_source_dependency_closure.json`).

## §3 Canonical ownership (new production modules)
| module | owns | was |
|---|---|---|
| `hymeko_rl/coin_delivery/e_approach.py` | `EApproachPolicy` + `load_e_approach_policy(...)` (fail-loud) | `exp_v3_handoff_gate._load_e` + `exp_option_retest._fresh_actor` |
| `hymeko_rl/coin_delivery/env_factory.py` | `make_coin_env` / `make_coin_contact_env` + C1 bank/contract/config | `exp_galambos_coord_ab.make_env` + `pedc_selection._env/_ctx/_load_pkl_bank/c1_config/_C1_HORIZON` |
| `hymeko_rl/env/planar_arm_kinematics.py` | `planar_2link_ik` / `ArmKin` / `extract_arms` / `ik_action` | `galambos_demo` private helpers (used by 12 prod modules) |

No experiment module is imported by this API. `galambos_demo` now **re-imports** the kinematics from production (no
§6.1 duplication) and is itself arc-free.

## §4/§5 Bit-identical to the old loader/env (proven)
- E-approach params: `param_sha256` match (`8fcf10ed…`); **`action_mean` max|Δ| = 0.00e+00** vs the frozen fixture
  (`reports/architecture/e_approach_fixtures/`).
- Env: `coin_env_fingerprint` **identical** (geom_fp `498e4e575065`, obs `[6,8]`, reward `galambos_task_deliver_v2b.hymeko`).
- Contact env (`make_coin_contact_env` vs `pedc._env`): env fingerprint **True**, bank length **376 == 376**,
  `MonitorContract` **equal** (7 fields).

## §7 Guards (extended `hymeko_rl/tests/test_architecture_guards.py`, + `test_coin_canonical_loader.py`)
- production→experiment import ratchet **45 → 40** (monotonic-decreasing).
- `test_coin_delivery_import_closure_is_experiment_free` — the coin import closure re-enters **0** arc modules.
- `test_e_approach_loader_is_canonical_and_fail_loud` — loader imports no experiment; raises on wrong hash / absent
  checkpoint (never a silent substitution).
- 15 unit tests (kinematics IK round-trip + clamp + reject, env fingerprint, contact env, loader fail-loud, 4-DoF action).

## §8 Behavioral regression — NO REGRESSION
| gate | target | measured |
|---|---|---|
| E first-contact (9 headline seeds) | 9/9 | **9/9** |
| E bilateral (9 headline seeds) | 7/9 | **7/9** |
| composed E+handoff POINT seed 1011 | 10/10 | **1/1 deterministic → 10/10** |
| control: frozen transport, seed 1011 | 0 | **0** |
| control: zero action, seed 1011 | 0 | **0** |

Because the E-approach action outputs are bit-identical and the env/contact-env fingerprints match, the composed
behavior cannot have changed; the panel confirms it.

## §9 Dependency closure (`coin_source_closure_v2.json`)
| measure | arc experiment files |
|---|---|
| **runtime closure** (sys.modules after POINT 1011 headline) | **0** (74 files, 68 tracked, 6 untracked-now-committed) |
| **module-top-level import closure** (fresh-clone import time) | **0** (46 modules) |
| full-AST closure (incl. dead/lazy in-function imports) | 35 — dead lazy imports never triggered on the delivery path; the deferred wider migration |

Cutting the two `pedc._env` edges (`coin_delivery_acquisition`, `coin_delivery_rl`) dropped `pedc_selection` and the
entire 18-module coin-toss cascade; the 5 `PhasePushController` modules were only reachable **through** that cascade,
so no 560-LOC push-controller extraction was needed.

## §10 Minimal commit — `ae1e94c`, 14 files, 897+/16−
6 new production + 5 redirected callers + 3 tests. `git diff --cached`: **0 non-`.py`**, **0 arc experiment files**,
**0 credentials/local-paths** (staged-diff scan clean). Out-of-closure edits (`push_primitive`, `reapproach`,
`galambos_demo`) left uncommitted — the deferred migration.

## §11 Fresh-clone proof
**Mac — clean room from `git archive` (PYTHONPATH = archive only, workspace off-path): PASS.**
```
STEP1 import OK; hymeko_rl loaded from archive = True
STEP2 make_coin_env OK  geom_fp=498e4e575065  obs=[6,8]  reward=galambos_task_deliver_v2b.hymeko
STEP3 fail-loud OK: FileNotFoundError on absent (gitignored) E checkpoint
STEP4 fail-loud OK: FileNotFoundError on absent (gitignored) contact bank
STEP5 ARC experiment-web modules imported: 0
```
**kato15** — arc-freeness is a pure Python import-graph property, provably platform-invariant; torch 2.12.0+cu126 /
mujoco already established on kato15 this session. The same `git archive` deploy re-verifies only platform library
availability, not the closure; **not separately re-executed** (would confirm a platform-independent fact). Available on
request.

## §12 External artifact manifest (`e_approach_external_artifact_manifest.json`)
Checkpoints/banks are gitignored external artifacts (the loaders fail loud without them). SHA-256:
- `E_valselect_v2.pt` `7dbbf1a7782f39f0…` (64061 B) — prefix `7dbbf1a7782f` pinned in `e_approach.py`.
- `handoff_best.pt` `8955e8db8ac1…` (34195 B); `learned_delivery_positive.pt` `8bd73d8cbea0…` (34279 B);
  `c1_heldseed_bank.pkl` `9262f6dc842b…` (209795 B).

## §13 Verdict: **COIN_SOURCE_REPRODUCIBLE**
The coin delivery runtime closure is **experiment-web-free** (29 → 0), the E-approach loader + env factory + planar
kinematics are canonical production and **bit-identical** to the old path, the source **imports cleanly from a fresh
clone and fails loud** on the manifested external artifacts, and the §8 behavioral panel shows **no regression**.
Minimal secret-free commit `ae1e94c` (14 `.py` files, 0 binaries).

**Scope honesty:** this is the bounded loader/env-factory canonicalization. The wider 56-import migration was **not**
begun — 5 out-of-closure push-controllers still import `PhasePushController` from `galambos_demo` (now itself arc-free),
tracked by the ratchet (40). Operationally pending user authorization: public push of the branch, and the confirmatory
(platform-invariant) kato15 clone.
