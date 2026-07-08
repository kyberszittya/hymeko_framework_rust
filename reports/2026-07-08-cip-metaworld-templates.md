# CIP over the MetaWorld task templates (coffee-push, dial-turn) — Ito+Kato items 2–4, template level

**Date:** 2026-07-08 15:47 JST
**Author:** Aiko
**Status:** built, tested, run. **Template-level method validation** (synthetic trajectories to the monitor
story). Real-env rollouts are a **gated** follow-up (require the `metaworld` dependency — §1).

---

## Context — where this sits in the Ito+Kato task

The CIP scenario is the **Ito + Kato** task, `docs/task/20260702_task_ito_kato`. Its items:

1. HyMeKo → CIP — built (bridge + DirectLiNGAM consumer).
2–4. **MetaWorld** description + test scenario + task description (**coffee-push, dial-turn**) — **this report**.
5. Runtime monitor on success — built (`task_monitor/`).
6. HyMeKo Code Agent — draft plan only.

**Two facts that set the scope (verified):** the `metaworld` package is **not installed** on this Mac (only
`gymnasium 1.3`, `mujoco 3.10`), and the MetaWorld monitor (`task_monitor/metaworld.py`) is **synthetic-driven by
design** — its own docstring puts a real-env wrapper out of scope. So the in-bounds step (no new dependency, monitor
used **read-only** as instructed) is to run the coin PoC's pipeline — **monitor → RolloutFrame → DirectLiNGAM →
`.hymeko` cross-view** — on synthetic trajectories generated to each task's monitor *story*.

## What was built (non-core, additive)

| File | LOC | Role |
| --- | --- | --- |
| `hymeko_rl/eval/cip/metaworld_cip.py` | 150 (new) | synthetic coffee-push/dial-turn generators (acyclic non-Gaussian SEM) + the CIP pipeline per task |
| `hymeko_rl/tests/test_metaworld_cip.py` | 60 (new) | 8 tests: no-metaworld-dep, success spread, end-to-end cross-view, causal-order sanity |
| `hymeko_rl/eval/cip/__init__.py` | +2 | export `run_metaworld_cip`, `TEMPLATES` |

**Design.** Each rollout is an **acyclic non-Gaussian SEM** (no hidden confounder — causal sufficiency): an
exogenous input (`approach_error` / `engage_error`) → the manipulation fraction (`contact` / `rotation`) → the
**monitor's** `progress_score` → a MetaWorld-style dense `reward_proxy` (reach + in-place + contact bonus) as the
downstream sink. The **monitor is unchanged and read-only**; DirectLiNGAM discovers the structure among the
controlled inputs + monitor outputs. (An earlier draft used a shared latent `skill` driving both approach and
contact — a hidden confounder that muddied the recovered order; removing it, per the causal-sufficiency
assumption the kato-lingam note flags, gave the clean recovery below.)

## Results (measured, N=120/task)

Run dir: `reports/figures/2026_07_08_15_45_cip_metaworld/`. Both DAGs cross-view-verified (engine backend,
canonical Blake3 hash, declared ≡ engine tensor view).

| Task | Recovered causal order | Strongest edges | Cross-view |
| --- | --- | --- | --- |
| **coffee_push** (pass 0.25) | `approach_error → contact_fraction → progress_score → reward_proxy` | `contact→progress +0.95`, `approach→contact −0.86`, `contact→reward +0.55` | ✅ `blake3:7a943e…` |
| **dial_turn** (pass 0.07) | `engage_error → rotation_fraction → progress_score → reward_proxy` | `rotation→progress +0.98`, `engage→rotation −0.90`, `rotation→reward +0.52` | ✅ `blake3:ec7e21…` |

Both recover the scenario **story** exactly, and — thematically consistent with the coin PoC — the
**`reward_proxy` is the downstream sink**, fed by contact/rotation and progress (never an exogenous root). The
signed structure survives the IR round-trip (e.g. `@c1{ (+approach_error, -contact_fraction); }` in
`causal_coffee_push.hymeko`).

## Honest scope

- This validates the **monitor + CIP + HyMeKo pipeline** on the coffee-push / dial-turn templates and
  machine-verifies the DAGs. It is **not** a claim about a real MetaWorld policy — the trajectories are synthetic,
  drawn from a known SEM (the MetaWorld analog of the coin **Phase-1** synthetic ground truth, not Phase-2 real
  rollouts).
- **Real-env upgrade (gated):** running CIP over real coffee-push/dial-turn rollouts requires installing
  `metaworld` (a new dependency → §1 approval; also a gymnasium-version compat check, since MetaWorld historically
  pins older gym). Not done unprompted. Once installed, the same pipeline points at real rollouts unchanged (the
  monitor already accepts the trajectory schema).
- Doctrine unchanged: the DAG is PROPOSED; controlled ablation decides. dial-turn's low pass-rate (0.07) is a tight
  success tolerance, not a pipeline issue — `progress_score` still carries the variance the recovery needs.

## Tests + static

- **30 passed** (`test_metaworld_cip.py` 8 + the read-only `test_metaworld_monitors.py` 22). Full CIP suite green.
- `ruff` clean · `radon cc` no block ≥ C (`run_metaworld_cip` split via `_fit_declare_render`) · `mypy --strict`
  clean on the new module (native `import hymeko` scope-ignored).
- **No §6.5 anti-patterns.** MetaWorld monitors used read-only (not modified); one module with a `--task` flag;
  shared `render_dag`/`cross_view_verify`/`DirectLiNGAM` reused; discovery pass confirmed no existing MetaWorld-CIP
  harness.

## Experiment provenance

- Git SHA at start: `a39acbb`. No training; no persistent state mutated. Deterministic (`np.random.default_rng`).
- Env: `metaworld` **absent** (by design); `gymnasium 1.3.0`, `mujoco 3.10.0` present but unused by this path.
- Artifacts: `reports/figures/2026_07_08_15_45_cip_metaworld/{coffee_push,dial_turn}_summary.json`,
  `dag_{coffee_push,dial_turn}.png`, `causal_{coffee_push,dial_turn}.hymeko`.

## Follow-ups

1. **Real-env rollouts (gated on `metaworld` install + §1 approval)** — the real-data analog of the coin Phase-2;
   would let CIP surface an *actual* reward-farming failure on coffee-push/dial-turn, not a template story.
2. **Reward ablation analog** — the coin Stage-A contact-reward ablation transfers directly once a real MetaWorld
   reward is in play (recompute reward variants offline, check the edge to `reward_proxy` collapses).

**Related:** `2026-07-08-cip-phase2-coin-poc.md`, `2026-07-08-cip-contact-reward-ablation-setup.md`;
task `docs/task/20260702_task_ito_kato`; memory `project-cip-lingam-rl-diagnostics`.
