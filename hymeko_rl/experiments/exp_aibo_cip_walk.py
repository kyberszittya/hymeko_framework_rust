"""Cheap CIP-informed Aibo-walking loop — a local smoke wiring three research findings into the Aibo goal-reach
task, per the enhancement synthesis:

  * **CIP / LiNGAM** — a DirectLiNGAM diagnosis of the trot found the dominant causal edge ``leg_speed ⇒
    torso_height`` (bounce), not ``leg_speed ⇒ forward_vx`` (propel). We add the ``vertical_bounce`` reward term
    (``-v_z²``) so the reward gradient routes leg energy into forward motion, then **re-diagnose** the trained
    policy (the CIP verification loop).
  * **coin-toss** — the trot is the imitation anchor (DAgger/TD3+BC, not scratch off-policy; TD3+BC is used here
    only as the cheap smoke — the full campaign should prefer DAgger, which the Aibo *standing* showed is the
    lever). Multi-seed, oracle/trot ceiling reported.
  * **Peter-schema** — an **asymmetric CTDE critic** (privileged full state, via ``privileged_state``) to close
    covariate shift; a structural/relational actor is used when the off-policy builder supports it.

This is a *mechanism* smoke (short, CPU): it proves the loop runs and the CIP reward + asymmetric critic are
wired, NOT a converged walk (that needs the kato15 campaign). Reuses `eval.causal`, `train.ddpg`,
`QuadrupedTrotGait`; nothing re-implemented."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from hymeko_rl.env.locomotion_experts import QuadrupedTrotGait
from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
from hymeko_rl.env.reward import RewardSpec

# CIP-informed goal reward: up-weighted forward driver + the anti-bounce term DirectLiNGAM motivated.
# `bounce` is the anti-vertical-bounce weight (−v_z²). The 2026-07-17 teacher CIP discovery measured the Aibo
# trot as 3.7× bounce-dominated (bounce-edge 0.451 ≫ propel-edge 0.122), motivating an A/B of bounce ∈ {3, 8}.
def cip_goal_reward(bounce: float = 3.0) -> RewardSpec:
    """CIP-informed Aibo goal reward with a configurable anti-bounce weight.

    # Preconditions ``bounce >= 0``.  # Postconditions returns a :class:`RewardSpec` whose ``vertical_bounce``
    term carries ``bounce``; all other terms fixed. ``bounce=3.0`` reproduces the original ``CIP_GOAL_REWARD``."""
    return RewardSpec((
        ("goal_progress", 30.0),        # forward driver (CIP: forward is under-served vs bounce)
        ("success_bonus", 50.0),
        ("vertical_bounce", bounce),    # CIP: −v_z² — penalise the bounce channel (A/B axis)
        ("time_penalty", 0.1),
        ("joint_acceleration", 0.002),
    ))


CIP_GOAL_REWARD = cip_goal_reward()     # back-compat constant: the bounce=3.0 default
_CVARS = ("action_energy", "forward_vx", "uprightness", "torso_height", "leg_speed")


# Hamiltonian-momenta + Lyapunov reward (underactuated control, 2026-07-17): the principled generalisation of the
# CIP anti-bounce term. Rewards forward CENTROIDAL momentum + a control-Lyapunov certificate (energy-orbit +
# balance/CAM + capture-point) so the optimiser stays on the certified-stable manifold — and it supplies the
# forward-velocity reward the CIP-bounce spec was missing (the mined "reward pays for bounce-avoidance but not
# propulsion" gap). All terms are O(1)-normalised, so these weights set the balance.
def hamiltonian_goal_reward(target_speed: float = 0.6) -> RewardSpec:
    """Full-stack Hamiltonian-Lyapunov Aibo goal reward. ``target_speed`` (m/s) parameterises H_ref via the env."""
    return RewardSpec((
        ("goal_progress", 20.0),                    # keep the task objective (reach the goal)
        ("success_bonus", 50.0),
        ("forward_momentum", 8.0),                  # + p_com,x — propulsion as momentum (the missing forward reward)
        ("transverse_momentum", 2.0),               # − (p_y²+p_z²) — subsumes vertical_bounce + lateral
        ("centroidal_angular_momentum", 1.0),       # − ‖CAM_xy‖² — balance (tipping) certificate
        ("energy_regulation", 0.5),                 # − (H−H_ref)² — energy-shaping Lyapunov
        ("capture_point", 2.0),                     # − ξ_lat² — lateral DCM fall-predictor bound
        ("alive", 1.0),
        ("joint_acceleration", 0.002),
    ))


HAMILTONIAN_GOAL_REWARD = hamiltonian_goal_reward()


class CipAiboEnv(QuadrupedGoalEnv):
    """Aibo goal-reach env with the CIP-informed reward + a privileged state for the asymmetric CTDE critic.

    ``bounce`` sets the anti-vertical-bounce reward weight (default 3.0 reproduces prior runs). ``hamiltonian``
    swaps in the full-stack Hamiltonian-momenta + Lyapunov reward (underactuated control); ``target_speed`` (m/s)
    parameterises its energy reference H_ref (read by ``energy_regulation``/``capture_point``)."""

    def __init__(self, *, cip_reward: bool = True, bounce: float = 3.0, hamiltonian: bool = False,
                 target_speed: float = 0.6, **kw: Any) -> None:
        self.target_speed = float(target_speed)          # read by the Hamiltonian reward terms (H_ref, ω)
        if hamiltonian:
            spec: "RewardSpec | None" = hamiltonian_goal_reward(target_speed)
        elif cip_reward:
            spec = cip_goal_reward(bounce)
        else:
            spec = None
        super().__init__(base="free", task="goal", reward_spec=spec, **kw)

    def privileged_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)


def cip_diagnose(make_env: Callable[[], Any], action_fn: Callable[[Any], np.ndarray], *, seeds: int = 5,
                 steps: int = 400, noise: float = 0.15) -> dict:
    """Collect a rollout variable matrix and fit DirectLiNGAM. # Returns the two edges of interest — the propel
    edge ``leg_speed ⇒ forward_vx`` and the bounce edge ``leg_speed ⇒ torso_height`` — plus mean forward dx."""
    from hymeko_rl.eval.causal import DirectLiNGAM
    rng = np.random.default_rng(0)
    rows: dict[str, list[float]] = {c: [] for c in _CVARS}
    dxs = []
    for s in range(seeds):
        env = make_env()
        env.reset(seed=s)
        x0 = float(env.data.xpos[env.torso, 0])
        for _ in range(steps):
            a = np.clip(action_fn(env) + rng.normal(0, noise, env.n_actions), -1, 1).astype(np.float32)
            _, _, term, trunc, info = env.step(a)
            rows["action_energy"].append(float(a @ a))
            rows["forward_vx"].append(float(info["vx"]))
            rows["uprightness"].append(float(info["upright"]))
            rows["torso_height"].append(float(env.data.xpos[env.torso, 2]))
            rows["leg_speed"].append(float(np.abs(env.data.qvel[env._act_dofs]).mean()))
            if term or trunc:
                break
        dxs.append(float(env.data.xpos[env.torso, 0]) - x0)
    x = np.column_stack([rows[c] for c in _CVARS]).astype(float)
    b = DirectLiNGAM().fit(x, list(_CVARS)).adjacency
    i = {c: k for k, c in enumerate(_CVARS)}
    return {"propel_edge": float(b[i["forward_vx"], i["leg_speed"]]),
            "bounce_edge": float(b[i["torso_height"], i["leg_speed"]]),
            "mean_dx": float(np.mean(dxs)), "n": int(x.shape[0])}


def _greedy(actor: Any) -> Callable[[Any], np.ndarray]:
    import torch

    def fn(env: Any) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(env.node_features()[None], dtype=torch.float32)
            return np.asarray(actor.action_mean(o).squeeze(0).numpy(), dtype=np.float32)
    return fn


def run(smoke: bool = True) -> dict:
    """The cheap loop: (1) CIP-diagnose the trot; (2) train the enhanced stack short; (3) CIP re-diagnose."""
    import torch

    from hymeko_rl.train.ddpg import build_offpolicy, td3_bc_config, train_offpolicy
    out = Path("experiments/2026_07_16_aibo_cip_walk")
    out.mkdir(parents=True, exist_ok=True)
    gait = QuadrupedTrotGait()
    horizon = 300 if smoke else 400

    # (1) baseline: the trot demonstrator, CIP-diagnosed.
    trot_cip = cip_diagnose(lambda: QuadrupedGoalEnv(base="free", task="goal", goal_distance=4.0, max_steps=horizon),
                            lambda e: gait.action(e), seeds=4, steps=horizon)

    # trot demos (command anchor) for TD3+BC.
    obs_d, act_d = [], []
    for s in range(3):
        env = CipAiboEnv(goal_distance=4.0, max_steps=horizon)
        env.reset(seed=s)
        for _ in range(horizon):
            a = gait.action(env)
            obs_d.append(env.node_features())          # (N, 2) — matches the env observation_space
            act_d.append(a)
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    offline = (np.asarray(obs_d, np.float32), np.asarray(act_d, np.float32))

    # (2) enhanced stack: structural actor if the builder supports it, else mlp; asymmetric critic (priv), CIP reward.
    env = CipAiboEnv(goal_distance=4.0, max_steps=horizon)
    flat = int(np.prod(env.observation_space.shape))
    kind = "hsikan"
    try:                                     # Peter: relational/structural actor over the leg hypergraph
        actor, critics = build_offpolicy("hsikan", obs_dim=2, flat_dim=flat, action_dim=env.n_actions,
                                         action_scale=1.0, n_critics=2, hidden=128, device="cpu",
                                         hg_state=env.hg)
    except Exception as exc:                 # fall back to flat if the structural backbone won't build
        print(f"[cip_walk] structural actor unavailable ({type(exc).__name__}); falling back to mlp")
        kind = "mlp"
        actor, critics = build_offpolicy("mlp", obs_dim=flat, flat_dim=flat, action_dim=env.n_actions,
                                         action_scale=1.0, n_critics=2, hidden=128, device="cpu")
    steps = 4000 if smoke else 40_000
    cfg = td3_bc_config(total_steps=steps, start_steps=500, batch_size=128, eval_every=steps, n_eval=1,
                        log_every=1000, seed=0)
    train_offpolicy(actor, critics, env, cfg, offline_data=offline)

    # (3) CIP re-diagnose the trained policy.
    torch.manual_seed(0)
    learned_cip = cip_diagnose(lambda: QuadrupedGoalEnv(base="free", task="goal", goal_distance=4.0, max_steps=horizon),
                               _greedy(actor), seeds=4, steps=horizon)

    result = {"actor_kind": kind, "asymmetric_critic": True, "cip_reward": True, "train_steps": steps,
              "trot": trot_cip, "learned": learned_cip,
              "bounce_reduced": learned_cip["bounce_edge"] < trot_cip["bounce_edge"],
              "propel_improved": learned_cip["propel_edge"] > trot_cip["propel_edge"]}
    (out / ("smoke.json" if smoke else "full.json")).write_text(__import__("json").dumps(result, indent=2, default=float))
    return result


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(run(smoke="--full" not in sys.argv), indent=2, default=float))
