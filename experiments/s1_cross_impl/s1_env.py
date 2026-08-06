"""Self-contained S1 environment (fixed-mug, reach-only) — importable by BOTH the project venv and the SB3 venv.

No hymeko_rl dependency (only metaworld / numpy / gymnasium), so SB3 SAC and our train_sac run on the *byte-identical*
environment: same metaworld coffee-push task **frozen to one fixed mug+target** (explicit `_last_rand_vec`), same
reach-only reward `1 - tanh(4·d_eef_mug)`, same obs-normalisation (shared mean/std), same 5 cm success criterion,
same 500-step horizon. The frozen rand-vec + obs-norm are captured once and saved so both implementations load the
exact same setup.
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import gymnasium as gym

ENV_ID = "coffee-push-v3-goal-observable"
POLICY = "SawyerCoffeePushV3Policy"
REACH_SCALE = 4.0
CONTACT = 0.05  # 5 cm success criterion (predefined, stable)
HORIZON = 500
STD_FLOOR = 0.05


def _base():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]

        return ENVS[ENV_ID]()


def _fix_task(env, rand_vec):
    u = getattr(env, "unwrapped", env)
    u._freeze_rand_vec = True
    u._last_rand_vec = np.asarray(rand_vec, dtype=np.float64)
    return env


class S1ReachReward(gym.Wrapper):
    """Reach-only reward on an already-frozen task; publishes ``d_eef_mug`` + stage_success(<5cm) into info."""

    @staticmethod
    def _d(obs):
        return float(np.linalg.norm(np.asarray(obs)[0:3] - np.asarray(obs)[4:7]))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return obs, {**info, "d_eef_mug": self._d(obs), "stage_success": False}

    def step(self, action):
        obs, r_native, term, trunc, info = self.env.step(action)
        d = self._d(obs)
        r = float(1.0 - np.tanh(REACH_SCALE * d))
        info = {
            **info,
            "d_eef_mug": d,
            "r_reach": r,
            "r_native": float(r_native),
            "stage_success": bool(d < CONTACT),
        }
        return obs, r, term, trunc, info


class ObsNorm(gym.ObservationWrapper):
    def __init__(self, env, mean, std):
        super().__init__(env)
        self._m = np.asarray(mean, np.float32)
        self._s = np.maximum(np.asarray(std, np.float32), STD_FLOOR)

    def observation(self, obs):
        return ((np.asarray(obs, np.float32) - self._m) / self._s).astype(np.float32)


def make_s1_env(mean, std, rand_vec):
    """The exact S1 env: fixed task + reach reward + obs-norm. Identical for SB3 and our SAC."""
    return ObsNorm(S1ReachReward(_fix_task(_base(), rand_vec)), mean, std)


def save_setup(path: str, n_demo: int = 8) -> dict:
    """Capture ONE fixed task (rand_vec) + obs-norm mean/std from the scripted expert on that fixed task."""
    import metaworld.policies as mp

    env0 = _base()
    env0.reset(seed=0)
    rand_vec = np.asarray(
        getattr(env0, "unwrapped", env0)._last_rand_vec, dtype=np.float64
    )
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_demo):
            env = _fix_task(_base(), rand_vec)
            pol = getattr(mp, POLICY)()
            obs, _ = env.reset(seed=i)
            for _ in range(HORIZON):
                rows.append(np.asarray(obs, np.float32))
                a = np.clip(np.asarray(pol.get_action(obs), np.float32), -1.0, 1.0)
                obs, _r, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
    arr = np.asarray(rows, np.float32)
    mean, std = arr.mean(0), arr.std(0)
    np.savez(path, mean=mean, std=std, rand_vec=rand_vec)
    return {
        "path": path,
        "rand_vec": rand_vec.tolist(),
        "n_demo_steps": int(arr.shape[0]),
    }


def load_setup(path: str):
    d = np.load(path)
    return d["mean"], d["std"], d["rand_vec"]


if __name__ == "__main__":
    p = str(Path(__file__).resolve().parent / "s1_setup.npz")
    info = save_setup(p)
    m, s, rv = load_setup(p)
    # verify the mug is FIXED across resets of the built env
    env = make_s1_env(m, s, rv)
    mugs = []
    for i in range(5):
        env.reset(seed=100 + i)
        base = getattr(env.env.env, "unwrapped", None)
        o, _, _, _, inf = env.step(np.zeros(4, np.float32))
        mugs.append(inf["d_eef_mug"])
    print(f"saved {p} | rand_vec[:3]={info['rand_vec'][:3]}")
    print(
        f"d_eef_mug across 5 resets (should be ~constant if task fixed): {[round(x, 4) for x in mugs]}"
    )
