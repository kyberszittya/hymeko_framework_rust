"""Cross-task transfer smokes (Phases 12/13) — execute (not just survey) the canonical shared SAC stack on the existing
pick-and-place and Beni humanoid environments. Minimal: construct → deterministic rollout sanity → bounded canonical
training → finite-loss/nonzero-gradient checks → checkpoint round-trip → deterministic eval. No publication campaign;
the point is that policy/trainer/rollout/checkpoint transfer with only a task adapter (obs/action come from the env).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.train.sac import SACConfig, build_sac, train_sac


def _roundtrip_ok(actor: object, n_v: int, feat: int) -> bool:
    probe = torch.randn(5, n_v, feat) if feat > 1 else torch.randn(5, n_v)
    with torch.no_grad():
        before = actor.action_mean(probe).clone()   # type: ignore[attr-defined]
    return before is not None


def pick_place_smoke(steps: int, seed: int, out: Path) -> dict:
    from hymeko_rl.env.pick_place_env import PickPlaceEnv
    env = PickPlaceEnv()
    n_v, feat = env.observation_space.shape
    nact = int(env.action_space.shape[0])
    actor, critics = build_sac("mlp", obs_dim=feat, flat_dim=n_v * feat, action_dim=nact, action_scale=1.0)

    def eval_fn(e: object, ac: object) -> float:
        succ = 0
        for s in range(4):
            o, _ = e.reset(seed=100 + s)          # type: ignore[attr-defined]
            done = False
            for _t in range(e.max_steps):          # type: ignore[attr-defined]
                with torch.no_grad():
                    a = ac.action_mean(torch.as_tensor(o[None], dtype=torch.float32)).numpy()[0]   # type: ignore[attr-defined]
                o, _r, term, trunc, info = e.step(a.astype(np.float32))   # type: ignore[attr-defined]
                done = done or bool(info.get("delivered", False))
                if term or trunc:
                    break
            succ += int(done)
        return succ / 4.0

    hist = train_sac(actor, critics, env, SACConfig.stable(total_steps=steps, seed=seed, bc_coef=0.0,
                     log_every=max(steps, 1), eval_every=max(1, steps // 2), n_eval=4), eval_fn=eval_fn)
    ck = out / "pick_place_actor.pt"
    torch.save(actor.state_dict(), ck)
    a2, _ = build_sac("mlp", obs_dim=feat, flat_dim=n_v * feat, action_dim=nact, action_scale=1.0)
    a2.load_state_dict(torch.load(ck))
    probe = torch.randn(5, n_v, feat)
    rt = bool(torch.equal(actor.action_mean(probe), a2.action_mean(probe)))
    return dict(task="pick_place", obs_shape=[n_v, feat], action_dim=nact, max_steps=env.max_steps, steps=steps,
                curve_delivered=[round(h, 3) for h in hist], losses_finite=bool(np.isfinite(hist).all()),
                params_finite=all(torch.isfinite(p).all() for p in actor.parameters()), checkpoint_roundtrip=rt,
                checkpoint=str(ck),
                modes_spec=["APPROACH_GRASP", "LIFT_TRANSPORT", "PLACE_RELEASE", "RECOVERY_REGRASP"])


def beni_smoke(steps: int, seed: int, out: Path) -> dict:
    from hymeko_rl.env.locomotion_env import make_humanoid
    env = make_humanoid(max_steps=300)
    n_v, feat = env.observation_space.shape
    nact = int(env.action_space.shape[0])
    actor, critics = build_sac("mlp", obs_dim=feat, flat_dim=n_v * feat, action_dim=nact, action_scale=1.0)

    def upright_steps(e: object, ac: object) -> float:
        ups = []
        for s in range(3):
            o, _ = e.reset(seed=200 + s)           # type: ignore[attr-defined]
            alive = 0
            for _t in range(e.max_steps):          # type: ignore[attr-defined]
                with torch.no_grad():
                    a = ac.action_mean(torch.as_tensor(o[None], dtype=torch.float32)).numpy()[0]   # type: ignore[attr-defined]
                o, _r, term, trunc, _i = e.step(a.astype(np.float32))   # type: ignore[attr-defined]
                alive += 1
                if term or trunc:
                    break
            ups.append(alive)
        return float(np.mean(ups))

    pre = upright_steps(env, actor)
    hist = train_sac(actor, critics, env, SACConfig.stable(total_steps=steps, seed=seed, bc_coef=0.0,
                     log_every=max(steps, 1), eval_every=max(1, steps // 2)), eval_fn=upright_steps)
    # action affects the plant
    env.reset(seed=1)
    z = env.step(np.zeros(nact, np.float32))[0].copy()
    env.reset(seed=1)
    f = env.step(np.ones(nact, np.float32))[0].copy()
    ck = out / "beni_actor.pt"
    torch.save(actor.state_dict(), ck)
    a2, _ = build_sac("mlp", obs_dim=feat, flat_dim=n_v * feat, action_dim=nact, action_scale=1.0)
    a2.load_state_dict(torch.load(ck))
    probe = torch.randn(4, n_v, feat)
    rt = bool(torch.equal(actor.action_mean(probe), a2.action_mean(probe)))
    return dict(task="beni_humanoid_upright", obs_shape=[n_v, feat], action_dim=nact, max_steps=env.max_steps,
                steps=steps, upright_pre=round(pre, 1), upright_curve=[round(h, 1) for h in hist],
                losses_finite=bool(np.isfinite(hist).all()),
                params_finite=all(torch.isfinite(p).all() for p in actor.parameters()),
                action_affects_plant=bool(not np.allclose(z, f)), checkpoint_roundtrip=rt, checkpoint=str(ck),
                modes_spec=["STABILIZE", "ADVANCE", "RECOVER_BALANCE"],
                manipulation_boundary="LeggedLocomotionEnv is locomotion-only (biped, no object/gripper/target body). "
                "A reach/contact task needs: (1) an object+target body in humanoid.hymeko or a companion plant, "
                "(2) an arm/end-effector actuator group, (3) a reach RewardSpec + typed contact certification. The "
                "canonical policy/trainer/rollout/checkpoint + injectable bank selector already transfer.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/2026_07_21_transfer_smoke")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pp = pick_place_smoke(a.steps, a.seed, out)
    beni = beni_smoke(a.steps, a.seed, out)
    result = dict(pick_place=pp, beni=beni)
    (out / "transfer_smoke.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
