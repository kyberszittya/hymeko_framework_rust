# Collaborative (cooperative CTDE) Galambos reframe — prototype (2026-06-25, overnight)

## Summary

Reframed Galambos from a single policy controlling both arms into a **cooperative multi-agent** scenario: each
arm is an agent, the two cooperate to deliver the coin, and they **share the team reward**. Realized with two
small, reuse-heavy pieces and **no environment duplication**:

- `CollaborativeGalambos` — a cooperative multi-agent *view* over `PlanarGraspEnv`. The underlying action vector
  is already arm-separable (`[j1_left, j2_left, j1_right, j2_right]`), so the view only splits per-agent actions
  and merges them back; obs is the shared hypergraph (CTDE), reward is the existing global reward shared by both
  agents.
- `CTDEActorCritic` — one **shared HSiKAN backbone** (the structural reasoning) feeding **per-arm action heads**
  (decentralized actors) + **one centralized critic**. CTDE: centralized training, decentralized execution.
  `action_mean(obs)` returns the concatenated full action, so it is drop-in for the existing BC / PPO / eval code.

A discovery pass (Explore agent) confirmed **zero prior multi-agent scaffolding** in the repo, so this is net-new
(no duplication). The reframe realizes `project-actor-critic-shared-reasoning` (share the HSiKAN reasoning, split
into per-agent heads) and is the prototype for Kato's collaborative-k-agent item in the dual-discriminator plan.

**Key result on trainability:** because `CTDEActorCritic` exposes the standard `act`/`value`/`evaluate(obs,
action)` interface and emits the concat action under the team reward, **the existing `train_ppo` already trains it
as correct cooperative CTDE** — centralized critic (the shared critic sees the whole observation) + decentralized
per-arm heads + parameter-shared reasoning, optimized by standard PPO on the team reward. No new trainer is needed
for the shared-reward case. The `CollaborativeGalambos` view becomes necessary only for the harder MARL settings:
independent learners (separate policies), per-agent (non-team) rewards, or decentralized (arm-local) observations.

## Files touched

- `hymeko_rl/collaborative.py` (+170, new) — `arm_action_partition`, `CollaborativeGalambos`, `CTDEActorCritic`,
  `build_collaborative`.
- `hymeko_rl/tests/test_collaborative.py` (+90, new) — 5 tests (below).
- `hymeko_rl/exp_collaborative.py` (+70, new) — BC functional-sanity harness (single HSiKAN vs collab CTDE).

CORE.YAML items touched: none.

## Test results

`pytest hymeko_rl/tests/test_collaborative.py`: **5 passed** (~7 s). ruff + mypy --strict: clean.

- CTDE policy shapes + gradient to the shared backbone, *both* per-arm heads, and the centralized critic.
- CTDE constructor validation (feat_dim/agent-dims).
- Action partition tiles the actuator vector contiguously with no overlap; `merge` is the partition's inverse;
  wrong agent count raises.
- Multi-agent env rollout returns one obs + one reward per agent, rewards equal (cooperative team reward),
  `info` carries `in_zone`.
- `build_collaborative` is BC/eval-compatible (`action_mean` → full `(B, n_actions)`).

## Queued (behind the running task-graph BC job — not stacked, per the page-memory constraint)

- `python -m hymeko_rl.exp_collaborative --seed {0,1,2}` — BC functional sanity: does collab CTDE deliver
  comparably to single-agent HSiKAN? (Expect a tie at the Galambos BC ceiling — this checks soundness, not
  benefit; benefit needs real CTDE training with per-agent exploration.)
- A short `train_ppo` smoke on a `CTDEActorCritic` to confirm cooperative-CTDE training runs end-to-end on the
  existing trainer.

## Performance

Unit tests CPU-only, < 8 s, far under the 16 GB cap (toy tensors + one small MuJoCo env). No OOM.

## §6.5 anti-patterns

None. No env clone (a *view* over `PlanarGraspEnv`); the partition is derived from actuator names, not hard-coded;
the policy reuses `hsikan_backbone`/`mlp_backbone` (no new backbone); `build_collaborative` is one entry with a
`kind` dispatch (no per-kind wrappers); CTDE reuses the existing `train_ppo` (no duplicate trainer).

## Provenance

Git: branch `fix-hsikan`, dirty. Tests seeded (`torch.manual_seed`, env `seed=0`). Discovery via Explore agent
(read-only). Host: Windows 11. Next: BC sanity + PPO smoke after `reports/2026-06-25-dual-rate-taskgraph.log`
completes.
