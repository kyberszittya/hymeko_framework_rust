# Asymmetric (MADDPG) privileged centralized critic — CTDE contract completion

**Date:** 2026-07-05 15:40 +0900
**Author:** Aiko (agent) for Dr. Csaba Hajdu
**Plan:** `docs/plans/2026-07-05-maddpg-privileged-critic/` (tex/pdf/tikz/mmd)
**Git SHA at start:** `4320202` (working tree already dirty at session start — pre-existing uncommitted work; see *Provenance*)

## Summary

The user handed a five-point acceptance contract for the collaborative coin-toss central critic:
`Q(global_coin_target_contact_state, a_left, a_right, phase)`. I first **audited the wired off-policy path**
(`exp_galambos_coord_ab → campaign → train_offpolicy`) against the five conditions, then closed the two that
failed.

**Audit verdict (before this change):**

| # | Condition | Before |
|---|-----------|--------|
| 1 | critic input contains both agents' actions | ✅ held (`QCritic` action_dim = joint width) |
| 2 | critic input contains global coin-target geometry | ✅ held (`node_features` coin→zone vector) |
| 3 | critic input contains contact state for both agents | ❌ **absent** from the critic obs |
| 4 | actor update uses the centralized critic gradient | ✅ held (`-critics[0](s, actor(s))`) |
| 5 | eval reports both_contact + target coin velocity + delivery | ⚠️ partial (no coin velocity) |

The `phase` term of the Q signature was also absent. Contact and phase are **not reconstructable** from the
geometry obs (contact needs contact forces; phase is the `_ever_grasped` history latch), so they cannot be
recovered post-hoc — they must be supplied and stored.

**Design chosen (user decision):** **asymmetric CTDE (MADDPG)** — the centralized critic reads a privileged
global state `z(s) = [left_contact, right_contact, phase-onehot(reach, carry, in_zone)]` (5-d) that the
**decentralized actors never see**. The actor is untouched (geometry obs only, `_FEAT=8`), so decentralized
execution needs no `z` and **existing collab actor checkpoints stay loadable** — only the critic (training-only,
transient) gains width.

**After this change:** conditions 1–4 hold; condition 5 complete (`coin_vel_to_zone` added). The centralized
critic now ingests `Q(s, [a_L|a_R], z)` with `z` carrying contact + phase.

## Files touched (my edits only)

Source (non-core — CORE.YAML grep for these paths returned nothing):

| File | Change |
|------|--------|
| `hymeko_rl/train/replay.py` | `ReplayBuffer(priv_dim=0)`; `add/add_batch` priv args; `sample_with_priv`; `sample` unchanged. |
| `hymeko_rl/train/ddpg.py` | `QCritic(priv_dim=0)` + `forward(obs,action,priv=None)`; `train_offpolicy` priv branch (buffer/rollout/loss/log), fail-loud guard; non-priv path byte-identical. |
| `hymeko_rl/env/planar_grasp_env.py` | `privileged_state()`, `privileged_dim=5`; `disk_vel` field in `PlanarGraspMetrics`; removed 2 now-unused `type: ignore`. |
| `hymeko_rl/agents/multichannel_ctde.py` | `build_collaborative_offpolicy(privileged=True)` sets critic `priv_dim` from `env.privileged_dim`. |
| `hymeko_rl/experiments/exp_galambos_coord_ab.py` | `_coordination_metrics` (both_contact + `coin_vel_to_zone` in one rollout); `_both_contact_rate` kept as wrapper. |
| `hymeko_rl/eval/critic_probe.py` | `privileged=False` at its two build sites (the probe studies the plain Q(s,a) clone-critic geometry). |

Tests (my edits): `test_replay.py` (new, 7 tests), `test_multichannel_ctde.py` (+4 priv tests, 1 updated),
`test_offpolicy_framework.py` (+2 priv tests), `test_planar_grasp_env.py` (+4 privileged/disk_vel tests, helper
fixed), `test_htl_reward.py` (helper fixed for the new metrics field).
*(The diffstat also shows `test_campaign.py`, `test_ddpg.py`, `test_galambos_demo.py`, and part of
`exp_galambos_coord_ab.py` as changed — those are **pre-existing uncommitted work**, not this task.)*

## CORE.YAML items touched

**None.** Grep of `CORE.YAML` for `planar_grasp`, `multichannel_ctde`, `ddpg`, `campaign`, `replay` — no matches.
No dependency added.

## Test results

All executed with `pytest -p no:randomly` (deterministic order).

| Layer | Coverage | Result |
|-------|----------|--------|
| Unit | `privileged_state` shape/one-hot-phase/contact/transitions; `disk_vel` consistency; `QCritic` 2-arg vs 3-arg (head +priv_dim wider); buffer priv round-trip; **non-priv `sample` bit-identity** vs plain buffer | pass |
| Integration | asymmetric-CTDE priv critic trains **finite** through `train_offpolicy` — single-env (warm-start) **and** vectorized (`n_envs=4`) paths on the real env | pass |
| Regression | `test_build_collaborative_offpolicy_critics` updated to the priv default; fail-loud guard (priv critic on non-priv env → `ValueError`); symmetric flag disables priv; non-collab off-policy / SAC / critic-probe / campaign / offpolicy-eval unchanged | pass |

Aggregate over the affected + regression suites: **all passing** (runs of 101, 61, 50, 40 across the touched
files; no failures after the two positional-`PlanarGraspMetrics` test helpers were updated for the new field).

## Static analysis (§6.3)

- **ruff:** clean on all changed files.
- **mypy --strict (isolated, HEAD-vs-current baseline compare):** `replay.py` clean; `ddpg.py` 7→7 (no net new —
  pre-existing torch-untyped-call / cudagraph noise); `multichannel_ctde.py` 0→0; `planar_grasp_env.py` 3→3
  (pre-existing mujoco-stub + `observation_space.shape` Optional); `exp_galambos_coord_ab.py` 2→**0** (improved).
  **Net new type errors introduced: zero.** No new `# type: ignore` added; two now-unused ones removed.

## Performance

- **Production-scale smoke** (`exp_galambos_coord_ab --smoke`, 1 seed × 3k steps, real priv critic, vec `n_envs=8`,
  device auto): **wall 40.9 s**, ran clean end-to-end. Oracle gate certified the training reward (run would raise
  otherwise). Curve: `bc_step0 → rl_refined → rl_refined`, losses finite. Artifacts in
  `experiments/2026_07_05_15_36_galambos_coord_ab_deliver/`.
- **`coin_vel_to_zone`** now emitted in every curve point and the peak (cond. 5).
- **Cost of `z`:** critic head input +5 floats; buffer priv arrays `2 × capacity × 5` float32 ≈ **4 MB** at
  capacity 1e5 — negligible vs the 16 GB cap. Actor (deploy path) untouched. Well under budget; no RSS/latency
  contract at risk.
- delivery = 0.0 at the 3k path-check budget is expected (path check, not a delivery run; prior evidence —
  `2026-07-05-qterm-collapse-rootcause` — is that model-free off-policy RL degrades this task past the 0.84
  teacher ceiling).

## New / removed dependencies

None.

## Provenance

- Working tree was **already dirty at session start** (pre-existing `M` on `exp_galambos_coord_ab.py`,
  `test_campaign.py`, `test_ddpg.py`, `test_galambos_demo.py`, several others; untracked checkpoints/experiments).
  This change layers the privileged-critic path on top; the audit and edits above are attributable to this task.
- Smoke seed: 0. Device: auto (CUDA when available; smoke wall measured on this host). MuJoCo physics env.
- Behavior change on the DEFAULT collab build: `build_collaborative_offpolicy(privileged=True)` now yields
  `priv_dim=5` critics on any env exposing `privileged_state` — i.e. `exp_galambos_coord_ab` and
  `galambos_plain_reward` train with the privileged critic by default. `critic_probe` opts out (`privileged=False`).

## Open issues / follow-up

- **Demonstration (out of scope here):** a multi-seed 200k A/B (privileged vs. symmetric critic on delivery) is
  the run that would *measure* whether the privileged critic helps. It is a separate launch, gated on a clean
  smoke (done) and an explicit go-ahead. This task makes the critic **contract-complete**; it does not claim a
  delivery gain — and prior evidence says not to expect one from model-free off-policy RL on this task.
- The 3-state phase is a coarse proxy; if the A/B shows it inert, a finer phase (or dropping it) is a cheap next
  iteration. It is critic-only, so it cannot corrupt delivery.
- Per the July-5 wording contract: `scripted_controller ~0.80–0.84`, `bc_clone ~0.44–0.52`, `rl_refined` worse
  than BC in measured runs, `framework_substrate` still to implement. This change is a **CTDE apparatus
  correction**, not a new delivery result.
