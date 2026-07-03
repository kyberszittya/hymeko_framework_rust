# RL campaign framework — removing the accidental complexity

**Date:** 2026-07-04 00:21 (+09:00 JST) · **Author:** Aiko (Claude Code), for Dr. Cs. Hajdu
**Commits:** `138ee5c` (framework) → `16efe8b` (mypy clean) · **Branch:** `fix-hsikan`

## Summary

The BC→TD3+BC→best-checkpoint→artifacts loop had been copy-pasted into 5+ throwaway scripts this session
(`collab_coin_offpolicy`, `coin_confirm_200k`, `collab_coin_armcol_ab`, `karm_cointoss_stage3`,
`collab_box_fingertip`) — CLAUDE.md §6.5 #3 (per-experiment scaffold duplication). This lifts the loop into **one
deep module**, `hymeko_rl/campaign.py`. An experiment is now a frozen config plus four pure closures, not a
60-line copy.

**Complexity, correctly defined** (Ousterhout): the cost to understand and change a system, from *dependencies*
and *obscurity* — not lines of code. The reduction here: to change the loop there is now **one** place, not N; to
add an experiment you write a spec, not a copy; and the run **log lives with the artifacts** instead of vanishing
to an ephemeral scratchpad (the obscurity fix).

## Design (UML in `docs/plans/2026-07-04-rl-campaign-framework/`)

- **Strategy** (functional): `make_env`, `build` (architecture), `measure`, `demos` are injected callables. Swap
  architecture = swap `build`; swap task = swap `make_env`. No class hierarchy.
- **Template Method**: `Campaign.run` fixes the skeleton (dir + tee-log → per-seed BC→refine→best-checkpoint →
  summarise → GIF); the variable steps are the strategies.
- **Immutability / pure functions**: `CampaignConfig` is a frozen dataclass; `compare` maps the same inputs over
  architectures for an A/B.
- **Reuse, no new copies** (§6.1): `train_offpolicy`, `behaviour_clone`, `experiment_dir`, `render_actor_gif`
  used verbatim.

Class diagram: `plan.tikz`. Sequence of `run()`: `plan.mmd`. Full plan: `plan.tex`/`plan.pdf`.

## Before → after

```python
# before: ~60 lines per experiment, copy-pasted, each able to drift; the log lost to scratchpad.

# after: the whole k-arm coin-toss A/B, in the framework —
env = lambda: PlanarGraspEnv(robot=None, max_steps=300, difficulty=0.3, reward_spec=R, coin_shape="cylinder")
compare(
    CampaignConfig("karm_cointoss", select="delivery", total_steps=200_000),
    make_env=env,
    builds={"collab": lambda e: build_collaborative_offpolicy(e, kind="sa_hsikan", n_critics=2, critic_layernorm=True),
            "joint":  lambda e: build_offpolicy("sa_hsikan", obs_dim=..., n_critics=2, hg_state=e.hg, critic_layernorm=True)},
    measure=galambos_measure,          # {delivery, arm_crash} in one rollout
    demos=collect_galambos_demos,
)
```

## Tests (§3) — `hymeko_rl/tests/test_campaign.py`, 5 passed

1. `tee_stdout` writes every printed line to the file **and** restores `sys.stdout`.
2. `CampaignConfig` is frozen (immutability).
3. `Campaign.run` is self-contained: writes `results.json` + `run.log` + a best-checkpoint policy; the log
   contains the `CURVE` lines.
4. Best-checkpoint keeps the **peak** (0.9), never the lower endpoint (0.3) — the selection is correct.
5. `compare` runs one `Campaign` per architecture and returns a per-arch verdict.

`ruff check` clean; `mypy --strict hymeko_rl/campaign.py` clean (no new suppressions).

## Files touched (non-core; CORE.YAML: none)

- **New** `hymeko_rl/campaign.py` (~180 LOC), `hymeko_rl/tests/test_campaign.py` (5 tests).
- Plan `docs/plans/2026-07-04-rl-campaign-framework/` (4 artifacts, gitignored).

## Contract adherence (§0–§10, the point of this exercise)

Plan on disk and compiling **before** the code (§2); tests at the unit layer, all passing (§3); the change
*removes* duplication rather than adding it (§6.5 #3, §6.1); lints clean, warnings-as-errors (§6.3); this report
(§9). The framework is the mechanism that makes the next experiment cheap **and** contract-compliant by default.

## Open / next

- Migrate the surviving experiment (`run_galambos_bc`) onto `Campaign` when next touched (opportunistic, not a
  rewrite).
- Provide a `galambos_measure` helper (`{delivery, arm_crash}` in one rollout) next to the env so campaigns don't
  re-implement it.
