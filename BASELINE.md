# Canonical baseline — EXECUTABLE_HYMEKO_OPTION_RL_PIPELINE_V1

**Immutable code baseline commit:** `772a11a4` · **immutable tag:** `executable-hymeko-option-rl-v1`
**Documented baseline (this doc):** tag `executable-hymeko-option-rl-v1.1` (= `772a11a4` + this file only) · **branch:**
`baseline/executable-hymeko-option-rl-v1`
**Status:** `EXECUTABLE_HYMEKO_OPTION_RL_PIPELINE_V1_COMPLETE` — frozen canonical baseline.

The working coin carry-option semi-MDP RL system whose architecture, execution, online verification, skill routing, and
learning contract all derive from HyMeKo. All new work branches from here; a failed experiment returns here without
re-running the critic / BC / action-language / certifier audits — unless a frozen regression contract below fails.

## Included milestone commits
- `d795d7f1` framework option-RL engine migration (`hymeko_rl/option_rl/`); coin = first task adapter · `bde89fd9` report
- `eda15e4b` §1 — the carry-option phase automaton IS `coin_carry_option_v1.hymeko` (gated bit-identical)
- `11cfd7ea` §2A — delivery verdict = online trace-monitor sourced from the `.hymeko` `@certificate`
- `05bbd1c4` §3 — first-class trained/frozen skill routing from the `.hymeko`
- `2e2fa058` §4 — minimal option-RL metamodel; the run is described in HyMeKo
- `772a11a4` — coverage report (HEAD of the immutable baseline)
- `4077dd51` Stage-5b infra · `5085676d` Stage-5b results

## Active `.hymeko` descriptions
- `data/robotics/coin_carry_option_v1.hymeko` — phase automaton (§1) + `@certificate` monitor semantics (§2A) + per-phase
  skill routing (§3).
- `data/robotics/option_rl_meta.hymeko` — task-independent option-RL metamodel vocabulary (§4).
- `data/robotics/coin_carry_option_rl_run.hymeko` — the concrete Stage-5b run contract (§4).

## Checkpoint / artifact hashes
- Frozen `pi_0` (`experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt`): SHA-256
  `1902454ca7a74c27…` — **gitignored** (`frozen/.gitignore: *.pt`); see Runtime dependencies below.
- Teacher bank (`…/carry_option_teacher_bank_v1.npz`): SHA-256 `8a6d612d3fc515bd…` (committed).
- Proposal checkpoint (`…/carry_proposal_refined.pt`, the RL init named in `coin_carry_option_rl_run.hymeko`): committed.

## Runtime dependencies (NOT in git — must be provisioned per checkout)
1. **Frozen `pi_0` checkpoint** — gitignored by repo policy; not on the remote. Provision by copying it into
   `experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt` (SHA `1902454c…`). **This is the one
   load-bearing non-code artifact; keep an external backup** (a GitHub release asset on the tag is the least-invasive option).
2. **Rust `hymeko` CLI** — build artifact: `cargo build -p hymeko_cli` (produces `target/debug/hymeko`).

## Regression gate (frozen contracts)
```
cd <checkout>
PYTHONPATH=<checkout> .venv/bin/python -m pytest \
  hymeko_rl/tests/test_coin_carry_fsm.py hymeko_rl/tests/test_coin_carry_monitor.py \
  hymeko_rl/tests/test_coin_carry_skills.py hymeko_rl/tests/test_coin_option_rl_spec.py \
  hymeko_rl/tests/test_coin_carry_option_rl.py hymeko_rl/tests/test_coin_carry_option.py \
  hymeko_rl/tests/test_coin_carry_proposal.py hymeko_rl/tests/test_option_rl.py -p no:randomly -q
```
**Expected: 45 passed.** Contracts covered: automaton parity (§1); monitor full-trace parity + fail-closed shadow (§2A);
skill-route parity + frozen-skill optimizer exclusion (§3); HyMeKo option-RL query gate + γ^τ round-trip + Stage-5 checkpoint
compatibility + update-0 evaluation-contract reproduction (§4). A failure here is a **checkout/environment regression**
(most likely a missing runtime dependency above), not a reason to reinterpret prior research results.

## Stage-5b RL result (frozen)
`CONSISTENT_POSITIVE_SAC_LEAN` — 4/4 SAC seeds beat their own update-0 proposal at fixed b=8 (ΔK6 median **+0.042**,
IQR [+0.035, +0.051]); TD3 control negative. `RL_SUPERIORITY_NOT_YET_STATISTICALLY_ESTABLISHED` — per-seed bootstrap CIs
span 0.

## Exact claims / non-claims
**Claims:** the structured option/search pipeline physically delivers the coin (~0.167 held-out vs pi_0 0.0); genuine
reward-driven semi-MDP RL is implemented; selected RL policies do not destroy the initialized competence; architecture,
execution, monitoring, skill routing, and learning contracts derive from HyMeKo; every load-bearing run fact is queryable
from the parsed graph and fail-closed-validated. **Non-claims:** statistical RL superiority is not yet established; the Rust
`hymeko_monitor` does not yet online-verify the run (§2B); the env `_strict` counter still exists internally (shadow, gated);
reward weights and most optimizer hyperparameters remain Python backend config.

## Working with the baseline
Return to the frozen baseline:
```
git switch baseline/executable-hymeko-option-rl-v1        # or: git switch --detach executable-hymeko-option-rl-v1
```
Start new work FROM the immutable baseline (never modify the baseline branch directly):
```
git switch -c feat/<new-direction> executable-hymeko-option-rl-v1
```
Separate future branches (each isolated; a failed experiment is abandoned/fixed on its own branch, never contaminating the
baseline): object/geometry generalization (`feat/object-to-target-variants-v1`), Rust `hymeko_monitor` PyO3 backend (§2B),
larger statistical-significance campaign, additional HyMeKo-native hyperparameters.
