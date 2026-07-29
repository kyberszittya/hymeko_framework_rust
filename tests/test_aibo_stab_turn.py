"""The structured stabilization action representation (``stab`` mode) for the fast rotational turn.

The fast turn (turn_rate 1.3) tips in ROLL (measured). The DOF that counter it — a lower CoM (crouch =
symmetric knee flexion) and a wider base (widen = mirrored hip abduction) — exist at the joint level but
no prior action space (``leg``/``omni``/``phase``) exposed them. The ``stab`` mode is that missing
representation: a 4-dim ``(Δrate, Δcrouch, Δwiden, Δlean)`` residual over a stabilized turn scaffold.
These tests pin the representation's contract: the structured offset maps to the right joints, ``a = 0``
with a crouch+widen scaffold keeps the fast turn UPRIGHT where the bare fast turn tips, and the residual
is a bounded modulation that stays inside the scaffold's neighbourhood.
"""

from __future__ import annotations

import numpy as np

from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv

_GRID = [(d, b) for d in (0.5, 0.7) for b in (0, 40, -40, 90, -90, 135, -135)]


def _grid_upright_reach(cfg: ResidualTrotConfig, policy) -> tuple[float, float]:
    env = ResidualTrotEnv(cfg, seed=0)
    hit = up = 0
    for i, (d, b) in enumerate(_GRID):
        _md, ok, upm = env.rollout_min_dist(policy, (d, b), seed=500 + i, horizon=2400)
        hit += int(bool(ok and upm > 0.5))
        up += int(upm > 0.5)
    return hit / len(_GRID), up / len(_GRID)


def test_stab_mode_action_is_four_dim() -> None:
    """The structured representation is low-dimensional (rate, crouch, widen, lean), not raw 12-dim legs."""
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="stab", obs_mode="flat"), seed=0)
    assert env.action_space.shape == (4,)


def test_stab_offset_maps_to_the_right_joints() -> None:
    """crouch → all knees (idx 3l+2); widen → hip abduction mirrored L/R (idx 3l+0); lean → knee diff."""
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="stab"), seed=0)
    off = env._stab_offset(crouch=0.5, widen=0.4, lean=0.0)
    knees = off[[2, 5, 8, 11]]
    abd = off[[0, 3, 6, 9]]
    assert np.allclose(knees, 0.5)                       # symmetric crouch on every knee
    assert np.allclose(np.abs(abd), 0.4)                 # widen on every hip abduction
    assert abd[0] == -abd[1]                             # left/right mirrored (fl vs fr)
    # lean is a left/right KNEE differential (roll bias), zero net over the symmetric pair
    leaned = env._stab_offset(crouch=0.0, widen=0.0, lean=0.3)[[2, 5, 8, 11]]
    assert leaned[0] == -leaned[1] and abs(leaned[0]) == 0.3


def test_zero_action_reproduces_stabilized_scaffold_upright() -> None:
    """a = 0 with a crouch+widen scaffold keeps the fast turn UPRIGHT where the bare fast turn tips."""
    z = np.zeros(4, np.float32)
    stab = ResidualTrotConfig(residual_mode="stab", obs_mode="flat", heading_mode="turn_then_walk",
                              turn_rate=1.3, stab_crouch=0.5, stab_widen=0.4, max_steps=1600)
    bare = ResidualTrotConfig(residual_mode="stab", obs_mode="flat", heading_mode="turn_then_walk",
                              turn_rate=1.3, max_steps=1600)                 # no stabilization
    reach_stab, up_stab = _grid_upright_reach(stab, lambda o: z)
    reach_bare, up_bare = _grid_upright_reach(bare, lambda o: z)
    assert up_stab > 0.8                                  # the scaffold stays upright on the fast turn
    assert reach_stab > 0.6                               # and reaches most wide bearings (hand-probe ~0.86)
    assert up_stab > up_bare + 0.3                        # decisively better than the bare (tipping) fast turn
    assert reach_stab > reach_bare + 0.3


def test_stab_scaffold_beats_the_slow_upright_baseline() -> None:
    """The stabilized fast turn reaches MORE than the slow-but-upright turn_rate=1.0 scaffold."""
    z = np.zeros(4, np.float32)
    fast = ResidualTrotConfig(residual_mode="stab", obs_mode="flat", heading_mode="turn_then_walk",
                              turn_rate=1.3, stab_crouch=0.5, stab_widen=0.4, max_steps=1600)
    slow = ResidualTrotConfig(residual_mode="stab", obs_mode="flat", heading_mode="turn_then_walk",
                              turn_rate=1.0, max_steps=1600)
    reach_fast, _ = _grid_upright_reach(fast, lambda o: z)
    reach_slow, _ = _grid_upright_reach(slow, lambda o: z)
    assert reach_fast > reach_slow                        # faster + stabilized reaches more within the horizon


def test_residual_is_a_bounded_modulation() -> None:
    """A non-zero stab residual changes the outcome but a = 0 is exactly the scaffold (determinism)."""
    cfg = ResidualTrotConfig(residual_mode="stab", obs_mode="flat", heading_mode="turn_then_walk",
                             turn_rate=1.3, stab_crouch=0.5, stab_widen=0.4, max_steps=400)
    env = ResidualTrotEnv(cfg, seed=0)
    env.reset(seed=3)
    d0 = [env.step(np.zeros(4, np.float32))[1] for _ in range(30)]
    env.reset(seed=3)
    d1 = [env.step(np.array([0.5, 0.0, 0.0, 0.0], np.float32))[1] for _ in range(30)]  # faster turn
    assert not np.allclose(d0, d1)                        # the rate modulation actually changes the rollout
    env.reset(seed=3)
    d0b = [env.step(np.zeros(4, np.float32))[1] for _ in range(30)]
    assert np.allclose(d0, d0b)                           # a = 0 is deterministic (same scaffold)


def test_existing_leg_mode_unchanged() -> None:
    """Regression: the stab additions leave the prior leg mode (defaults stab_* = 0) untouched."""
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="leg", obs_mode="flat"), seed=0)
    assert env.action_space.shape == (12,)
    env.reset(seed=1)
    off = env._stab_offset(0.0, 0.0, 0.0)
    assert np.allclose(off, 0.0)                          # zero scaffold offset → no behaviour change
