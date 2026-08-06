"""Fast architecture pre-screen — the toy lab as the front-end to the slow RL scenarios.

Every architecture verdict on pick-place / coin-toss costs ~1 h. This runs the *real* backbones through the
seconds-long toy bandit (:mod:`hymeko_rl.experiments.sanity_rl`) and returns a rank + a recommendation, so only survivors get
an hour-long scenario run ("screen then confirm"). The screen ranks **learning + B=1 deploy latency + params** —
the rollout-heavy launch-bound.

**Two things it deliberately does NOT screen** (stated so it cannot mislead):
  * **PPO-stability** — the 1-step bandit cannot see the warm-start collapse where MLP fails and HSiKAN holds.
    Confirm that on the multi-step collab/nav toy, or the scenario itself.
  * **holonomy advantage** — the toys are acyclic/linear; use :mod:`hymeko_rl.experiments.holonomy_probe` for that.
So the recommendation is "fastest learner", not "best policy for a warm-start scenario".
"""
from __future__ import annotations

from dataclasses import dataclass

from hymeko_rl.experiments.sanity_rl import BanditConfig, run_bandit_sanity


@dataclass(frozen=True)
class PrescreenResult:
    arms: dict[str, dict[str, float]]      # kind -> {reward, train_s, deploy_ms, params}
    recommend: str                         # fastest backbone that learns (latency criterion)
    criterion: str
    caveat: str


def prescreen(kinds: "tuple[str, ...]" = ("mlp", "hsikan", "sa_hsikan", "mixture"), *,
              steps: int = 300, learn_threshold: float = -0.3) -> PrescreenResult:
    """Pre-screen ``kinds`` on the toy bandit; rank by B=1 deploy latency among the backbones that learn.

    # Postconditions ``recommend`` is one of ``kinds``; ``arms[k]`` has reward / deploy_ms / params."""
    res = run_bandit_sanity(tuple(kinds), steps=steps, cfg=BanditConfig(target="flat"))
    learners = {k: m for k, m in res.items() if m["reward"] > learn_threshold}
    pool = learners or res                                       # if nothing learns, rank all by latency anyway
    recommend = min(pool, key=lambda k: pool[k]["deploy_ms"])
    return PrescreenResult(
        arms=res, recommend=recommend,
        criterion=f"fastest B=1 deploy among learners (reward > {learn_threshold})",
        caveat="latency + learning ONLY — does NOT screen PPO-stability (warm-start collapse) or holonomy; "
               "confirm stability on the multi-step toy / scenario before trusting the pick for a warm-start run.")


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kinds", nargs="+", default=["mlp", "hsikan", "sa_hsikan", "mixture"])
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args(argv)
    r = prescreen(tuple(a.kinds), steps=a.steps)
    print("RL pre-screen (toy bandit; pick the survivor before the hour-long scenario run):")
    for k, m in sorted(r.arms.items(), key=lambda kv: kv[1]["deploy_ms"]):
        print(f"  {k:12} reward={m['reward']:+.3f}  deploy={m['deploy_ms']:6.3f}ms  params={int(m['params'])}")
    print(f"=> recommend: {r.recommend}  ({r.criterion})")
    print(f"   caveat: {r.caveat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
