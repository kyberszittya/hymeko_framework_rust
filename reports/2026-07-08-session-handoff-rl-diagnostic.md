# Session handoff — coin-delivery off-policy RL diagnostic thread (2026-07-07/08)

**One-line state:** the deployable policy is **MLP+DAgger** (ft_dom **0.452** 3-seed val-selected / **0.75** on the
frozen checkpoint `mlp_s1_selected_d3`, md5 `edf4fe81…`). Every off-policy critic-gradient RL attempt failed; the
root cause is measured. A **full safety stack + monitor ecosystem** was built and committed. RL is frozen pending a
decision on the vector-critic branch (needs a fairer re-test) vs gradient-free/imitation.

## The four-attempt RL arc (each removed the prior failure, exposed the next)

1. **Baseline CTDE-TD3+BC** (guarded sanity) — failed by critic **mis-ranking** (Q(exploit) > Q(dagger)).
2. **CQL actor smoke** — CQL fixed + *held* the ranking; failed by **Q-scale runaway + whole-policy drift**.
3. **Residual + phase-gated** (frozen critic, bounded ε, contact-gated) — removed runaway + drift; still
   degraded, *on-manifold*.
4. **ε-sweep + gradient probe** — **root cause, measured directly:** the STRONG_PASS critic's local **∂Q/∂a is
   monitor-misaligned** (a+ε∇Q raises Q but cuts two-fingertip contact 0.045→0.010) even where its *ranking* is
   correct. No ε>0 improves; every ε>0 degrades monotonically.

**Reframe (user, 2026-07-08):** this means scalar critic-gradient RL is *mis-specified*, not that RL is impossible.
The task is a hierarchical multi-component contract → use a **vector-valued / constraint-projected** update.

5. **Vector-critic + projected-gradient (Steps 1–5, no actor trained)** — **INCONCLUSIVE.** Cosines showed the
   scalar gradient weakly conflicts with delivery/anti-exploit; but the component critics were poorly fit
   (negative Q on non-negative returns) and the branch metrics were noisy/flat. **Root cause of the
   inconclusiveness:** the replay is **near-deterministic DAgger — no action diversity**, so *no* critic (scalar or
   vector) can learn the Q-vs-action shape; ∇ₐQ is extrapolation. **Step 6 (vector actor smoke) NOT authorized.**

## What was built (committed) — the RL safety/diagnostic ecosystem

All non-core, tested, ruff-clean. Package `hymeko_rl/eval/task_monitor/` (already committed by a parallel agent)
holds the hierarchical monitor; this thread added:

- **`hymeko_rl/eval/task_monitor/{pipeline,provenance}.py`** — `PipelineSchemaLedger` (5-stage tensor field-order
  guard, wired into `train_offpolicy(verify_schema=…)`) + `PolicyProvenanceLedger` (checkpoint md5 / param hash /
  action checksum / anchor identity, wired into `train_offpolicy(provenance=…)`).
- **`hymeko_rl/eval/critic_benchmark.py`** — 5 diagnostics + `classify_critic` FAIL/WEAK_PASS/STRONG_PASS margin
  gate (A=WEAK, B=WEAK, **C=CQL STRONG_PASS**, E=FAIL).
- **`hymeko_rl/train/critic_repair.py`** — critic-only repair trainer (A/B/C/E loss Strategies), `cql_regularizer`
  (additive CQL penalty for `train_offpolicy(critic_regularizer=…)`), `train_residual` (frozen-critic residual).
- **`hymeko_rl/agents/residual_actor.py`** — `ResidualActor` `clip(π_DAgger + gate·ε·tanh(r_φ))`, zero-init,
  contact-gated.
- **`hymeko_rl/train/search_objective.py`** — `SearchObjective` (per-step monitor component signals; separate from
  the verifier). **`hymeko_rl/train/vector_critic.py`** — 6 component critics + PCGrad `projected_gradient`.
- **`hymeko_rl/train/{ddpg,replay}.py`** — the `verify_schema` / `provenance` / `critic_regularizer` hooks +
  `ReplayBuffer.column_schema()`.
- Tests: `test_{task_monitor,policy_provenance,critic_benchmark,residual_actor,vector_critic}.py` (~55 tests).
- Reports: `2026-07-07-{v2-task-monitor,pipeline-schema-guard,policy-provenance-guard,guarded-rl-sanity,
  critic-ranking-benchmark,cql-actor-smoke,residual-smoke,eps-sweep-gradient-probe}.md`,
  `2026-07-08-vector-critic-projected-gradient.md`. Figures under `reports/figures/{task_monitor,critic_benchmark,
  eps_sweep}/`. Result JSONs under `experiments/v2_*/results.json`.

## Binding rules established this thread

- **Minimum safety stack before ANY RL run:** PipelineSchemaLedger PASS · PolicyProvenanceLedger PASS · TaskMonitor
  active · reward-vs-monitor reported · critic-vs-monitor reported (if a critic). Every report carries the 14
  fields incl. checkpoint hashes + reward/env file.
- **Monitor stays the external verifier — NOT in the reward.** Any learning objective must be a *separate*
  `SearchObjective`, not the frozen `TaskMonitor`.
- No SAC, no plain actor-critic, no multi-seed until a critic clears the STRONG_PASS margin gate AND a
  vector-projected direction is shown promising.

## Decision pending (next session)

- **(A) Re-test the vector hypothesis fairly** (recommended): inject **action diversity** into the replay
  (exploration noise around DAgger / a perturbed-action dataset) so critic gradients are meaningful; fit component
  critics by **Monte-Carlo returns** (actor is frozen → exact) + an OOD term; use a **sensitive Step-5 metric**
  (larger ε, component-scored horizon, two-finger-state filter). Then re-run `scratchpad/v2_vector_critic_probe.py`
  logic and re-decide Step 6.
- **(B) Accept the upstream obstruction** — if near-deterministic-DAgger replay can't support reliable critic
  gradients, scalar + vector both inherit it → gradient-free/monitor-directed (CEM/ES on the ε-bounded residual
  scored by the monitor — crosses the "monitor external" line, needs sign-off) or **better imitation** (the lever
  past a BC/DAgger ceiling has always been imitation here).

## ⚠ Infra

**D: drive hit 100% full mid-session** (system-wide — mostly unrelated experiment GIFs + `.venv`, not this
thread's artifacts). Freed ~4 GB from regenerable caches to finish. **A real disk cleanup is required before the
next multi-run job.** User noted WMI resource spikes. Scratchpad harnesses live under the session temp dir
(`…/scratchpad/v2_*.py`) — not committed; the committed reports + result JSONs capture every number.

## Resume commands

```
# repro any verdict from the committed result JSONs:
cat experiments/v2_epsilon_sweep/results.json        # root-cause: gradient monitor-misalignment
cat experiments/v2_vector_critic/results.json        # inconclusive vector diagnostic
python -m pytest hymeko_rl/tests/test_{task_monitor,policy_provenance,critic_benchmark,residual_actor,vector_critic}.py
```
