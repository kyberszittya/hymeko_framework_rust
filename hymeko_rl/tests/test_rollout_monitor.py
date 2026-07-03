"""Rollout health monitor + the 2026-06-30 physics-artifact fixes (blow-up guard, timestep, coin clearance)."""
import numpy as np

from hymeko_rl.experiments.gripper_pick_bc import _DIVERGE_QACC, eval_success
from hymeko_rl.eval.rollout_monitor import RolloutHealth, RolloutMonitor, actuated_dofs


class _FakeData:
    def __init__(self, ndof: int) -> None:
        self.qacc = np.zeros(ndof)
        self.qpos = np.zeros(ndof)


class _FakeEnv:
    """Minimal gym-like env that drives the monitor into a chosen failure mode deterministically (no MuJoCo)."""
    def __init__(self, mode: str, ndof: int = 2) -> None:
        self.mode, self.n_actions, self.data = mode, ndof, _FakeData(ndof)
        self._t = 0

    def reset(self, seed: int = 0) -> tuple[np.ndarray, dict[str, object]]:
        self._t = 0
        self.data.qacc[:] = 0.0
        self.data.qpos[:] = 0.0
        return np.zeros(self.n_actions), {}

    def step(self, _a: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self._t += 1
        if self.mode == "diverge" and self._t == 5:
            self.data.qacc[:] = 1e6
        elif self.mode == "move":
            self.data.qpos += 0.05                       # healthy: joints keep moving
        # mode == "stall": qpos stays put -> no motion
        return np.zeros(self.n_actions), 0.0, False, self._t >= 200, {}


def _noop(_env: object, _obs: np.ndarray) -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def test_monitor_healthy_when_moving() -> None:
    h = RolloutMonitor(stall_window=10).run(_FakeEnv("move"), _noop, max_steps=50, motion_dofs=[0, 1])
    assert h.healthy and not h.diverged and not h.stalled and h.moved > 0.0


def test_monitor_flags_divergence() -> None:
    h = RolloutMonitor().run(_FakeEnv("diverge"), _noop, max_steps=50, motion_dofs=[0, 1])
    assert h.diverged and h.event_step == 4 and not h.healthy and "DIVERGED" in h.summary()


def test_monitor_flags_stall() -> None:
    h = RolloutMonitor(stall_window=15).run(_FakeEnv("stall"), _noop, max_steps=80, motion_dofs=[0, 1])
    assert h.stalled and h.event_step == 14 and not h.healthy and "STALLED" in h.summary()


def test_monitor_rejects_bad_config() -> None:
    for bad in ({"stall_window": 0}, {"diverge_qacc": 0.0}, {"stall_eps": -1.0}):
        try:
            RolloutMonitor(**bad)                                   # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# --- eval_success blow-up guard (the artifact that inflated 0.125 -> 0.875) ---

class _DivAC:
    def action_mean(self, x: object) -> object:
        import torch
        return torch.zeros(1, 1)


class _DivEnv:
    """A 'lift' is reported ONLY after the sim diverges — i.e. the lift is a blow-up artifact, never real."""
    max_steps = 20

    def __init__(self) -> None:
        self.data = _FakeData(1)
        self._t = 0

    def reset(self, seed: int = 0) -> tuple[np.ndarray, dict[str, object]]:
        self._t = 0
        self.data.qacc[:] = 0.0
        return np.zeros(1), {}

    def step(self, _a: object) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self._t += 1
        diverged = self._t >= 3
        self.data.qacc[:] = 1e6 if diverged else 1.0
        return np.zeros(1), 0.0, False, self._t >= 12, {"lifted": 1.0 if diverged else 0.0, "reached": diverged}


def test_eval_success_guard_rejects_blowup_lift() -> None:
    # Without the guard this env reads lift=1.0/place=1.0 (the post-blow-up state); the guard must score it 0/0.
    lift, place = eval_success(_DivEnv(), _DivAC(), n_episodes=4, seed=0)   # type: ignore[arg-type]
    assert lift == 0.0 and place == 0.0 and _DIVERGE_QACC > 1e2


# --- the env physics fixes ---

def test_pick_env_stable_substep() -> None:
    from hymeko_rl.viz.render_pick_place import fanuc_pick_env
    e = fanuc_pick_env()
    assert e.model.opt.timestep <= 5e-4 + 1e-12                       # sub-step small enough to not detonate
    assert abs(e.model.opt.timestep * e.frame_skip - 0.005) < 1e-9    # control_dt preserved exactly
    assert actuated_dofs(e.model)                                     # actuated DOFs resolve (for stall checks)


def test_planar_coin_clears_arms() -> None:
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    env = PlanarGraspEnv(robot=None, max_steps=120, difficulty=0.3, task_graph=True)
    for s in range(30):
        env.reset(seed=s)
        cx = float(env.data.qpos[env._disk_x_adr])
        cy = float(env.data.qpos[env._disk_y_adr])
        assert env._clear_of_arms(cx, cy), f"coin spawned inside an arm at seed {s}"


def test_rollout_health_summary_shapes() -> None:
    assert "healthy" in RolloutHealth(10, False, False, None, 12.0, 3.0).summary()
