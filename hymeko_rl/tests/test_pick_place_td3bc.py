"""Plumbing test for the FANUC pick-place TD3+BC entrypoint (``hymeko_rl.experiments.pick_place_td3bc``).

Not a "solve the task" test — a tiny-scale integration check that the wrapper compiles, the env builds, demos
collect, BC runs, TD3+BC starts from the BC warm-start, best-checkpoint preserves the BC floor, and the
self-contained artifacts (results.json + run.log) are written. Requires the built ``hymeko`` CLI + MuJoCo
(run on kato15 with ``MUJOCO_GL=egl``).
"""
from __future__ import annotations

import json
from pathlib import Path

from hymeko_rl.experiments.pick_place_td3bc import PickConfig, run


def test_budget_resolves_smoke_and_full() -> None:
    """resolve() caps under smoke and expands the full defaults — the budget contract (pure, fast)."""
    full = PickConfig()
    seeds, steps, demos, epochs, n_eval, eval_every, n_envs = full.resolve()
    assert seeds == (0, 1, 2) and steps == 100_000 and demos == 18 and epochs == 80 and n_envs == 8
    smoke = PickConfig(smoke=True)
    s2, st2, d2, e2, ne2, _, ne_envs = smoke.resolve()
    assert s2 == (0,) and st2 == 4_000 and d2 == 6 and e2 == 5 and ne2 == 4 and ne_envs == 2
    # smoke must CAP a fully specified config, never expand it into a full launch
    big = PickConfig(smoke=True, total_steps=500_000, n_demos=999, bc_epochs=999)
    _, st3, d3, e3, *_ = big.resolve()
    assert st3 == 4_000 and d3 == 6 and e3 == 5


def test_pick_td3bc_runs_and_preserves_bc_floor(tmp_path) -> None:
    """Tiny end-to-end: env→demos→BC→TD3+BC→measure→artifacts. Asserts the plumbing and that the
    best-checkpoint (step-0 BC floor is evaluated and raced) yields a place metric in [0, 1]."""
    cfg = PickConfig(kind="mlp", smoke=True, total_steps=200, n_demos=2, bc_epochs=1, n_eval=1)
    summary = run(cfg, base=str(tmp_path))

    assert summary["kind"] == "mlp"
    assert summary["reward_certificate"].startswith("N/A")            # honest: no galambos oracle for pick
    assert "place_median" in summary and 0.0 <= summary["place_median"] <= 1.0

    exp = Path(summary["dir"])
    assert exp.exists() and (exp / "results.json").exists() and (exp / "run.log").exists()
    data = json.loads((exp / "results.json").read_text())
    assert data["select"] == "place" and len(data["seeds"]) == 1
    seed0 = data["seeds"][0]
    # the BC warm-start floor must be the first curve point (stage bc_step0) — the anti-collapse anchor
    assert seed0["curve"], "no eval points recorded"
    assert seed0["curve"][0]["stage"] == "bc_step0"
    assert "place" in seed0["peak"] and "lift" in seed0["peak"]
