# RL campaign, the "no-success" investigation, and the delivery pivot (2026-06-25)

## Summary

Built a sequential multi-task RL campaign (`offpolicy_eval`: galambos, galambos_taskgraph, FANUC, quadruped ×
PPO/DDPG/TD3/SAC × HSiKAN/MLP), ran it from scratch, and found **all returns negative (no coin delivery)**. The
user flagged this as a likely RL-process bug. I investigated with discriminating tests rather than declaring, and
the verdict is: **not a bug — the env is sound and the learner works; the negatives are from-scratch
hard-exploration at a small step budget.** I then pivoted to a **BC-warm-start delivery campaign** (the validated
`galambos_bc` path), which delivers (first cell: 25%).

## The investigation (discriminating tests, not assertions)

**Hypothesis space for "all returns negative":** (a) broken/degenerate env, (b) reward/eval not wired, (c) the
learner not learning (process bug), (d) inherent hard-exploration + budget.

1. **Env is sound (rules out a, b).** A red herring first: the scripted demonstrator *crashed* on the campaign's
   `from_hymeko` env (`l1=l2=0`). Cause — the emitted arms put the link offset on the **x**-axis while the
   demonstrator's IK reads the **y**-axis; the arms articulate fine, the IK just can't read them. A
   forward-kinematics sweep confirmed **both** envs move identically (max body displacement 0.1534 at joints=1
   rad), and a reachability sweep showed the `from_hymeko` arm **reaches the coin** (min tip-to-coin **0.024 m**,
   vs the BC env's 0.019). Delivery is physically possible; the reward/success path is the same one BC scored.
2. **The learner learns (rules out c).** The training curves show **SAC climbing** — e.g. galambos sac mlp s0:
   −183 → −108 (Δ **+75**); hsikan s0: −154 → −126 (Δ +28). A broken reward/eval cannot produce a consistent
   +75 climb. **PPO is flat** (Δ +0…+6) — its known weakness on hard exploration, not a defect.
3. **Conclusion (d).** The cells train **from scratch, no BC warm-start, 30k steps**, on a hard two-arm grasp.
   The BC experiments delivered *because* they warm-started from the demonstrator; the campaign deliberately did
   not. SAC is climbing but hasn't crossed into delivery in 30k steps. That is budget + exploration, the
   documented wall (`project-galambos-hsikan-tie-rootcause`), not a new bug.

## The delivery pivot (BC warm-start)

Reused the validated `galambos_bc` path (collect demonstrator demos → `behaviour_clone` → PPO/off-policy refine,
on the hand-authored arms where the demonstrator works) across HSiKAN/MLP × PPO/DDPG/TD3 × 3 seeds, reporting
`bc_delivery` and `refine_delivery`.

**First result (hsikan, ppo, seed0):** `bc_delivery = 0.25`, `refine_delivery = 0.208` — **real delivery**
(25% of episodes), PPO-refine roughly holds. Full table pending (~18 cells, ~1 h). *Expected pattern* (stated up
front, per the FANUC memory): bc_delivery shows success; PPO-refine holds/improves; **DDPG/TD3-refine may regress**
toward 0 at 20k steps without a BC anchor (the proper fix is TD3+BC anchor + ≥1e5 steps — the queued FANUC "Both"
work).

## Other plan implementations completed this session

- **StructuralCritic** (`hymeko_rl/structural_critic.py`): value decomposed over the hypergraph's signed
  cycles+walks (enumerated once via the built `hymeko` PyO3 binding), full aggregation×pooling ablation grid,
  scalar TD. **11 tests pass, mypy --strict + ruff clean.** Not yet wired into `build_policy`.
- **Arc-weight rewrite**: `docs/manipulation_models/{meta_reward,galambos_task,pick_place_task}.hymeko` moved to
  arc weights (weights on the bundle→term incidence arcs, not term-node attributes); no attributes remain.
- **Task-graph A/B** (`reports/2026-06-25-dual-rate-taskgraph.md`): under BC, coin/zone-in-graph **helps MLP**
  (0.139→0.250) and **hurts HSiKAN** (0.111→0.042) — falsified the structural-leverage hypothesis as currently
  wired.

## Files touched (this report's scope)

- `hymeko_rl/offpolicy_eval.py` (+FANUC task), `hymeko_rl/structural_critic.py` (new),
  `hymeko_rl/tests/test_structural_critic.py` (new), `hymeko_rl/exp_dual_rate.py` (+`--task-graph`).
- `docs/manipulation_models/*.hymeko` (arc weights), `docs/plans/2026-06-25-{structural-critic,
  arc-weight-rl-rewrite,coin-toss-k-scaling}/` (plans).
- Reports: this file + `2026-06-25-dual-rate-{galambos,taskgraph}.md`, `2026-06-25-collaborative-galambos.md`.

## Resource / provenance

CPU-only (`CUDA_VISIBLE_DEVICES=-1`, `num_threads` 1–2); one heavy job at a time (WMI/page-memory safeguard);
per-cell subprocess isolation after a MuJoCo box-box `FatalError` (9>8 contacts) was found to abort in-process
sweeps. The `hymeko` binding built via `maturin develop` (maturin installed transiently via `uv pip`, non-core).
Git: branch `fix-hsikan`, dirty. Logs: `reports/2026-06-25-{campaign-full,campaign,galambos-delivery}.*`.
No OOM; 16 GB cap respected.

## Open / follow-up

1. Aggregate the delivery campaign (in flight, `b83xj6osi`) → delivery + HSiKAN-vs-MLP on a delivering task.
2. Resume the from-scratch campaign (`reports/2026-06-25-campaign-full.jsonl`) for quadruped/FANUC.
3. Off-policy delivery needs TD3+BC anchor + ≥1e5 steps (FANUC "Both"); 20k-refine collapse is expected, not a bug.
4. Wire StructuralCritic into `build_policy` as the critic head + run its ablation on galambos_taskgraph.
5. k-scaling env (k=1/3) — planned, deferred (risky env surgery, better with the user reachable).
