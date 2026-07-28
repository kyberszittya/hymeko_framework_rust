# R10.2 Stage 2 (Boundary 4) — conservative structured-option TD3: NO improvement within the frozen budget (honest negative)

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · commits `4697d263` (4a) → `360a36b9` / `7ee6373d` (4b) · dev s1 · downstream/transit/Stage-1B scaffold/physics/safety FROZEN · s4/s7 untouched · f1–f4 SEALED · no tag moved**

## Verdict

**`STRUCTURED_OPTION_TD3_NO_IMPROVEMENT_WITHIN_FROZEN_BUDGET`** (0/3 seeds improved). This is a *bounded* negative, **not** an RL-impossibility claim, and the conservative protocol behaved exactly as designed: the actor was never released onto an unrankable critic, so the deployed policy safely remains the proven scaffold.

## The learning result (3 seeds, ≤400 HOME-start episodes, eval every 25, frozen dev panel)

| seed | released | gate passed | Spearman(Q,r) | scaffold K6 | post-TD3 K6 | nominal K6 | safe | boundary |
|---|---|---|---|---|---|---|---|---|
| 0 | **No** | No | 0.025 | 1/16 | 1/16 | ✓ | 16/16 | 0 |
| 1 | **No** | No | 0.139 | 1/16 | 1/16 | ✓ | 16/16 | 0 |
| 2 | **No** | No | −0.021 | 1/16 | 1/16 | ✓ | 16/16 | 0 |

In all 3 seeds the actor stayed at the scaffold (θ=0): post-TD3 panel behaviour is identical to the scaffold, nominal HOME K6 is preserved, and there are **zero safety violations and zero boundary-route regressions**. `td3.json` + `td3_run.log` hold the full per-seed history.

## Why — the scaffold is a local optimum in structured-option space (the deep finding)

The **option-ranking gate** (`Q(corrective) > Q(medoid) > Q(destructive)` per state + a Spearman(Q, reward) ≥ 0.3 sanity) was checked 6× per seed (18 total, upd 124→374). It **never passed**, and the reason is informative rather than a training artifact:

- The critic consistently ranks the **medoid (θ=0, the scaffold) highest** — e.g. seed 2 final: `q_medoid 0.288 ≥ q_corrective 0.256 ≥ q_destructive 0.248`. So `Q(corrective) > Q(medoid)` fails **because the medoid is already the best option**.
- Spearman(Q, reward) hovers at ≈0 (−0.26 … +0.14) because, near the scaffold, exploration produces mostly interchangeable safe far-misses (the Boundary-3 landscape: 87/96 safe negatives at 25–50 mm) — the critic correctly learns "θ=0 is good" but there is no smooth, rankable *corrective* structure among the perturbations to learn.

In other words: within the frozen σ=0.05 exploration ball, **there is no local structured option that beats the scaffold**, so a reward-driven local policy correctly cannot improve on it. This is convergent with the R8 result (`BOUNDED_RESIDUAL_RL_GENERALISES_BUT_DOES_NOT_YET_DELIVER`) — the scaffold sits at the soft-frictional contact ceiling, and local corrections do not exceed it.

## What this does and does not claim

- **Does:** within this frozen budget (σ=0.05, D frozen, ≤400 ep/seed, this critic/actor), reward-driven TD3 does not beat the scaffold, and the scaffold is a local optimum in the structured-option coordinate.
- **Does NOT:** claim RL is impossible here, or that a larger budget / richer critic / wider exploration / a genuinely different (non-local) search could not improve — those are out of this boundary's frozen scope.
- The claim is **not** nominal K6 (the scaffold already has it); it rode entirely on paired panel improvement, which did not occur.

## Files touched

| file | role | Δ |
|---|---|---|
| `torque_path_frozen.py` | **new (4a)** — frozen `D` + `SIGMA=0.05` + honest `REVIEW_DECISION` (strict gate did NOT pass) | +50 |
| `torque_path_env.py` | **new (4a)** — HOME-start `StructuredOptionCaptureEnv` + K6-dominant reward (`reset≠1` = `boundary_route_variation`) | +95 |
| `torque_path_td3.py` | **new (4b)** — conservative TD3 engine: zero-init actor, immutable positive replay, critic warm-up, ranking gate, bounded eval-aligned gate checks | +255 |
| `coin_kinetic_structured_option_td3.py` | **new (4b)** — 3-seed driver + 3-way compare + verdict | +90 |
| `test_torque_path_env.py` (8) + `test_torque_path_td3.py` (5) | **new** — env/reward/frozen + TD3-engine tests (incl. a gate-reachability guard) | +140 |
| `exploration_freeze_decision.json`, `td3.json`, `td3_run.log` | **new** — the σ-freeze record, the machine-readable result, the run log | — |

**CORE.YAML items touched: none.** Frozen scaffold/downstream/transit/physics/safety untouched; s4/s7 untouched; f1–f4 sealed.

## Tests / static / performance

- **13 new tests pass** (8 env/reward/frozen + 5 TD3 engine), plus the earlier 31 (torque_path + conditioning + capture_rl) still green. `ruff` clean; `radon cc -a -nc` no block at C or worse (`train_seed` kept at B via `_learn_updates` / `_gate_due` / `_maybe_gate` helpers).
- A performance bug was found and fixed (`7ee6373d`): the ranking gate had been re-checked *every* episode (~90 s each); it is now checked only at eval points, bounded to `max_gate_checks=6`.
- 3-seed run: ~30 min wall (background), peak RSS < 1 GB (small torch actor/critic + mujoco, consistent with the 0.25 GB conditioning runs); hard cap 16 GB. Production scale: real mujoco 3.10, full downstream horizon 80, the 16-member frozen dev panel × 3 seeds × 400 episodes, training perturbations (seed 12345) disjoint from the eval panel (seed 90210).

## Provenance

- Commits `4697d263` → `360a36b9` → `7ee6373d`. Env: Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-arm64 (CPU). Seeds: TD3 {0,1,2}; train-perturbation 12345; eval-panel 90210. Deterministic.
- Eval on the **dev** panel (not a final held-out transfer claim; s4/s7 remain sealed validation-only).

## R10.2 arc close (Boundaries 1–4)

- **B1** plan (4-format, tectonic).
- **B2** `c8e90e11` — coordinate identity gates PASS: zero-θ ≡ medoid scaffold bit-exact → strict K6 2.79 mm.
- **B3** `be4cf935` — TERMINAL_OFFSET_TRACKING + LOCAL_THETA_SENSITIVITY PASS; σ=0.05 review-accepted (strict gate NOT passed; 0/96 safety).
- **B4** here — conservative TD3 → **NO_IMPROVEMENT_WITHIN_FROZEN_BUDGET**; the scaffold is a local optimum in structured-option space.

**Net:** a fully-proven, transparent HOME→strict-K6 scaffold; a correctly-conditioned, identity-frozen 15-D structured-option coordinate; and an honest, conservative reward-driven RL result that the scaffold is locally optimal and cannot be beaten by a local structured option within the frozen budget. The deployed policy is the scaffold. **STOP.**

## Candidate follow-ups (out of this boundary's scope, for a future decision)

Non-local / larger-budget search (CEM/population over θ, not a local critic), a richer or recurrent critic, a wider or annealed exploration schedule, or a genuinely different contact-mode scaffold (per the geometry-generalization roadmap G1–G4). None are claimed or started here.
