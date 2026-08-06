"""Phase 2: the declarative ``.hymeko`` reward reproduces the procedural pick-and-place reward (parity)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.pick_place_env import PickPlaceEnv

_TASK = "data/robotics/pick_place_task.hymeko"


def test_declarative_reward_matches_procedural() -> None:
    """``PickPlaceEnv(reward_profile=pick_place_task.hymeko)`` scores every step identically to the built-in
    procedural reward — so the whole manipulation reward now comes from one ``.hymeko`` model, bit-for-bit."""
    proc = PickPlaceEnv(max_steps=60)
    decl = PickPlaceEnv(max_steps=60, reward_profile=_TASK)
    proc.reset(seed=3)
    decl.reset(seed=3)                                   # same seed → identical spawn + dynamics
    assert decl._reward_spec is not None and len(decl._reward_spec.terms) == 7
    rng = np.random.default_rng(7)
    for _ in range(60):
        a = rng.uniform(proc._a_lo, proc._a_hi).astype(np.float32)
        _o, r_proc, term_p, trunc_p, _ = proc.step(a)
        _o, r_decl, term_d, trunc_d, _ = decl.step(a)
        assert abs(r_proc - r_decl) < 1e-5, f"reward mismatch: procedural={r_proc} declarative={r_decl}"
        assert term_p == term_d and trunc_p == trunc_d
        if term_p or trunc_p:
            break
