# CIP over the MetaWorld task templates (coffee-push, dial-turn) — Ito+Kato items 2–4, template level

**Date:** 2026-07-08 15:47 JST
**Author:** Aiko
**Status:** built, tested, run. **Both** the template-level validation (synthetic) **and** the real-env upgrade
(real MetaWorld coffee-push rollouts) are done — the real-env upgrade was authorized after the compat check below.

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

## Real-env upgrade (authorized) — real MetaWorld coffee-push rollouts

Installed `metaworld==3.0.0` (the Phase-2 analog: real physics, real dense reward, the reward-independent monitor).
`hymeko_rl/eval/cip/metaworld_cip.py --source real` rolls the scripted `SawyerCoffeePushV3Policy` with **per-episode
action noise as the observed exogenous input** (no hidden confounder), maps each MetaWorld obs to the coffee-push
monitor schema (positions → distances; `near_object` as the contact/approach proxy — coffee-push is a *push*, so
grasp never fires), and runs the same monitor → frame → DirectLiNGAM → `.hymeko` cross-view pipeline. Variables:
`action_noise`, `near_fraction`, the monitor's `progress_score`, and MetaWorld's dense `total_reward`;
`mw_success` stratifies.

**Compat check (done before install, as promised).** `metaworld==3.1.1` hard-pins `mujoco==3.3.0` and would have
**downgraded** the shared `mujoco 3.10` the coin/pick-place envs use. Pinning `mujoco==3.10.0` resolves to
`metaworld==3.0.0` (looser bound) with **no downgrade** — `gymnasium 1.3` untouched, `PlanarGraspEnv` verified
still working. That is the version installed.

**Result (N=80, `reports/figures/2026_07_08_16_00_cip_metaworld_real/`):** the pipeline runs on real rollouts and
cross-view-verifies (`agree=True`, canonical hash). The **stable finding** is the strong edge
**`near_fraction → total_reward` (≈+0.96–0.97)** with `progress_score → total_reward` (≈+0.82): the real MetaWorld
coffee-push reward is **proximity/contact-shaped and downstream** of the task variables (structurally echoing the
coin finding, though coffee-push's reward also tracks progress — it is better-aligned than coin's).

**Honest instability (measured, important).** MetaWorld's env randomization is **not controlled by the seed I
pass** — two identical N=40 seed-0 runs gave different pass-rates (0.42 vs 0.50) and different full causal orders
(same noise sequence, different physics). So the `near_fraction → total_reward` edge is stable, but the **full DAG
order / the reward's exact rank is a point estimate** — a ranking claim needs **multi-seed median/IQR** (§3), or
fixing the MetaWorld goal. `action_noise` is often isolated (no detected edge): the scripted policy is robust to
noise, so the intervention's causal influence is weak — itself an honest finding.

## Dependency + one test change (from the metaworld install)

- **New dependency (user-approved):** `metaworld==3.0.0`, installed with `mujoco==3.10.0` pinned to prevent a
  downgrade. **Not yet declared in `pyproject.toml`** — its `ml` group is the right home, but the file currently
  carries a whole-file CRLF-normalization diff (pre-existing, not mine); adding the line there is left as a clean
  follow-up so this change doesn't entangle with that diff. The venv has it; record here for reproducibility.
- **One existing test made robust (not a monitor change):** `test_metaworld_monitors.py::test_no_metaworld_dependency`
  asserted `"metaworld" not in sys.modules`, which the real-env path now violates in-session. Its *intent* (the
  monitor module has no metaworld import) is preserved via a static source check; monitor code is untouched.

## Tests + static

- **86 passed** across the CIP + metaworld suite (`test_metaworld_cip.py` 10 incl. 2 real-env,
  `test_metaworld_monitors.py` 22, plus causal/emit/ablation/cip_export). The real-env test skips if `metaworld`
  is absent.
- `ruff` clean · `radon cc` no block ≥ C (`run_metaworld_cip` / `run_metaworld_cip_real` are A/B, split via
  `_fit_declare_render`) · `mypy --strict` clean on the module (native `import hymeko` + untyped `metaworld`
  attr scope-ignored).
- **No §6.5 anti-patterns.** MetaWorld monitors used read-only (not modified); one module with a `--task` flag;
  shared `render_dag`/`cross_view_verify`/`DirectLiNGAM` reused; discovery pass confirmed no existing MetaWorld-CIP
  harness.

## Experiment provenance

- Git SHA at start: `a39acbb`. No training; no persistent state mutated. Deterministic (`np.random.default_rng`).
- Env: `metaworld` **absent** (by design); `gymnasium 1.3.0`, `mujoco 3.10.0` present but unused by this path.
- Artifacts: `reports/figures/2026_07_08_15_45_cip_metaworld/{coffee_push,dial_turn}_summary.json`,
  `dag_{coffee_push,dial_turn}.png`, `causal_{coffee_push,dial_turn}.hymeko`.

## Follow-ups

1. **Multi-seed real-env aggregation** — MetaWorld's seed-uncontrolled randomization makes the single-run DAG order
   a point estimate; run k seeds and report edge-weight median/IQR (§3) for a stable ranking, or pin the goal.
2. **dial-turn real-env** — needs a dial-angle extraction from the MetaWorld obs (the handle position → angle);
   coffee-push maps cleanly, dial-turn does not yet. The synthetic template already covers dial-turn's story.
3. **Reward ablation analog on real MetaWorld** — the coin Stage-A contact-reward ablation transfers once a real
   MetaWorld reward decomposition is in play (recompute reward variants offline, check the `→ total_reward` edge).
4. **Declare `metaworld` in `pyproject.toml`** (`ml` group, with the `mujoco==3.10` pin) once the pre-existing
   whole-file CRLF diff on that file is resolved.

**Related:** `2026-07-08-cip-phase2-coin-poc.md`, `2026-07-08-cip-contact-reward-ablation-setup.md`;
task `docs/task/20260702_task_ito_kato`; memory `project-cip-lingam-rl-diagnostics`.
