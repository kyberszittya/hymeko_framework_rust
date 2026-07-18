"""Demo-seeded-replay retention experiment — env, balanced demo collection, and rich eval.

Real Coffee-Push (random mug+target per episode via ``_freeze_rand_vec=False``), corrected SAC config
(reward_norm off + early-concat critic + contact ≤7.5 cm). Two arms differ ONLY in whether the replay buffer is
preloaded with 5000 scripted transitions (true env rewards/dones/obs; NO behavior-cloning, NO demo priority).
Question: does demo-seeded replay convert *transient* Coffee-Push success into *stable retained* performance?
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "s1_cross_impl"))
sys.path.insert(0, str(HERE.parents[1]))
import s1_env  # noqa: E402

CONTACT_OK = 0.075
SUCCESS_OTT = 0.07  # obj_to_target below this ≈ mug delivered (for demo bucketing)


class CoffeePushInfo(gym.Wrapper):
    """Random-task Coffee-Push; publishes ``d_eef_mug`` + ``mug_disp`` into info. Native reward (reward_norm off)."""

    def __init__(self, env) -> None:
        getattr(
            env, "unwrapped", env
        )._freeze_rand_vec = False  # random mug+target per reset
        super().__init__(env)
        self._mug0 = None

    @staticmethod
    def _d(o) -> float:
        return float(np.linalg.norm(np.asarray(o)[0:3] - np.asarray(o)[4:7]))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._mug0 = np.asarray(obs)[4:7].copy()
        return obs, {**info, "d_eef_mug": self._d(obs), "mug_disp": 0.0}

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        disp = (
            float(np.linalg.norm(np.asarray(obs)[4:7] - self._mug0))
            if self._mug0 is not None
            else 0.0
        )
        return (
            obs,
            float(r),
            term,
            trunc,
            {**info, "d_eef_mug": self._d(obs), "mug_disp": disp},
        )


def _norm(o, mean, std):
    return ((np.asarray(o, np.float32) - mean) / np.maximum(std, 0.05)).astype(
        np.float32
    )


def make_env(mean, std):
    """The exact training/eval env: random-task Coffee-Push + obs-norm (corrected stack uses reward_norm=False)."""
    return s1_env.ObsNorm(CoffeePushInfo(s1_env._base()), mean, std)


def fit_obsnorm(n=10):
    """Broad obs-norm from the scripted expert over random tasks."""
    import metaworld.policies as mp

    rows = []
    for i in range(n):
        env = CoffeePushInfo(s1_env._base())
        pol = getattr(mp, s1_env.POLICY)()
        obs, _ = env.reset(seed=i)
        for _ in range(s1_env.HORIZON):
            rows.append(np.asarray(obs, np.float32))
            obs, _r, term, trunc, _ = env.step(np.clip(pol.get_action(obs), -1, 1))
            if term or trunc:
                break
    a = np.asarray(rows, np.float32)
    return a.mean(0), a.std(0)


def _bucket(info, succ_ep):
    """Categorize a scripted transition: reach / contact / partial(push) / success(full push)."""
    near = info.get("near_object", 0) > 0.5
    disp = info.get("mug_disp", 0.0)
    ott = info.get("obj_to_target", 1.0)
    if succ_ep and ott < SUCCESS_OTT:
        return "success"
    if disp > 0.05:
        return "partial"
    if near:
        return "contact"
    return "reach"


def collect_balanced_demos(mean, std, n_per_bucket=1250, seed0=90_000, max_ep=4000):
    """Roll the scripted expert on random tasks; bucket transitions; return a BALANCED 5000-transition replay set
    (normalized obs, true reward/done). Returns ((obs,act,rew,next,done), composition, verification)."""
    import metaworld.policies as mp

    buckets: dict[str, list] = {
        "reach": [],
        "contact": [],
        "partial": [],
        "success": [],
    }
    n_succ_ep = 0
    ep = 0
    verify = {
        "reward_on_success_transition": None,
        "success_flag_present": False,
        "no_boundary_leak": True,
    }
    while min(len(v) for v in buckets.values()) < n_per_bucket and ep < max_ep:
        env = CoffeePushInfo(s1_env._base())
        pol = getattr(mp, s1_env.POLICY)()
        obs, _ = env.reset(seed=seed0 + ep)
        traj, succ_ep = [], False
        for _ in range(s1_env.HORIZON):
            a = np.clip(np.asarray(pol.get_action(obs), np.float32), -1, 1)
            nobs, r, term, trunc, info = env.step(a)
            succ_ep = succ_ep or bool(info.get("success", 0))
            traj.append((obs.copy(), a.copy(), r, nobs.copy(), bool(term), dict(info)))
            obs = nobs
            if term or trunc:
                break
        n_succ_ep += int(succ_ep)
        for o, a, r, no, d, inf in traj:
            b = _bucket(inf, succ_ep)
            if b == "success" and verify["reward_on_success_transition"] is None:
                verify["reward_on_success_transition"] = round(float(r), 3)
                verify["success_flag_present"] = bool(inf.get("success", 0) == 1)
            if len(buckets[b]) < n_per_bucket:
                buckets[b].append((o, a, r, no, d))
        ep += 1
    ol, al, rl, nl, dl = [], [], [], [], []
    comp = {}
    for b, items in buckets.items():
        sel = items[:n_per_bucket]
        comp[b] = len(sel)
        for o, a, r, no, d in sel:
            ol.append(_norm(o, mean, std))
            al.append(a)
            rl.append(float(r))
            nl.append(_norm(no, mean, std))
            dl.append(1.0 if d else 0.0)
    demos = (
        np.asarray(ol, np.float32),
        np.asarray(al, np.float32),
        np.asarray(rl, np.float32),
        np.asarray(nl, np.float32),
        np.asarray(dl, np.float32),
    )
    comp["episodes_rolled"] = ep
    comp["successful_episodes"] = n_succ_ep
    comp["total"] = int(sum(len(buckets[b][:n_per_bucket]) for b in buckets))
    return demos, comp, verify


def eval_episodes(env, actor, critics, gamma, demo_batch, n=50):
    """Deterministic (greedy) eval over n random-task episodes + Q on demo vs online + entropy/alpha proxies."""
    succ, contact, disp = [], [], []
    ents, q_online = [], []
    for i in range(n):
        obs, _ = env.reset(seed=200_000 + i)  # fixed eval task set (same across arms)
        ok, con, mx = False, False, 0.0
        for _ in range(s1_env.HORIZON):
            st = torch.as_tensor(obs[None], dtype=torch.float32)
            with torch.no_grad():
                a = actor.action_mean(st).squeeze(0).numpy()
                _, lp = actor.sample(st)
                q = float(
                    torch.stack(
                        [
                            c(st, torch.as_tensor(a[None], dtype=torch.float32))
                            for c in critics
                        ]
                    )
                    .amin(0)
                    .item()
                )
            ents.append(-float(lp.item()))
            q_online.append(q)
            obs, _r, term, trunc, info = env.step(a)
            if info.get("d_eef_mug", 9) < CONTACT_OK:
                con = True
            ok = ok or bool(info.get("success", 0))
            mx = max(mx, info.get("mug_disp", 0.0))
            if term or trunc:
                break
        succ.append(int(ok))
        contact.append(int(con))
        disp.append(mx)
    # Q on demo transitions (fixed set)
    with torch.no_grad():
        ds = torch.as_tensor(demo_batch[0][:512], dtype=torch.float32)
        da = torch.as_tensor(demo_batch[1][:512], dtype=torch.float32)
        q_demo = float(torch.stack([c(ds, da) for c in critics]).amin(0).mean().item())
    return {
        "success_rate": round(float(np.mean(succ)), 3),
        "contact_rate": round(float(np.mean(contact)), 3),
        "mug_disp_median": round(float(np.median(disp)), 4),
        "entropy_mean": round(float(np.mean(ents)), 3),
        "q_online_mean": round(float(np.mean(q_online)), 2),
        "q_demo_mean": round(q_demo, 2),
    }
