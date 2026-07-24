"""Stage 5 — search-in-the-loop semi-MDP option RL over the PROPOSAL CENTER.

Policy action = θ_center (the proposal). A FIXED environment wrapper samples b=8 structured candidates around θ_center,
selects the best by the frozen canonical local-search score, executes that committed push/brake/release option, and hands
off to frozen settling pi_0. The critic is Q(s_k, θ_center) because the stationary search wrapper is part of the environment
response to the proposal action. Semi-MDP target: R_option + γ^τ · Q_target(s_next, π_target(s_next)), no bootstrap after a
terminal (handoff→settled) option. The actor is NEVER trained against the search-selected θ (that stays diagnostic/provenance).

Reward: the raw per-step v3 reward is anti-aligned (audited 7dc46f24 — it penalises valid push-delivery), so the OPTION
reward is composed from the spec's certificate-aligned terms (eventual K6 ≻ robust handoff ≻ carry progress ≻ contact
retention ≻ −containment-exit ≻ −effort) and is CERTIFIED (structured expert ≫ pi_0) before any RL run. K6 and certificate
tolerances are unchanged; model selection is on held-out eventual K6, then containment safety.
"""
import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_carry_option import _safety_abort
from hymeko_rl.coin_delivery.coin_carry_proposal import search_select
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, DIM, PUSH_DTZ, T_MAX, T_MIN, _unpack
from hymeko_rl.coin_delivery.coin_markov_ablation_train import ACTION_SCALE, _aug, _det
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation

OBS = 48


# ------------------------------ action <-> θ (bounded) ------------------------------
def action_to_theta(a):
    """Normalised policy action in [-1,1]^15 → legal θ. Postcondition: amps∈±A_BOUND, durs∈[T_MIN,T_MAX]."""
    a = np.clip(np.asarray(a, np.float32), -1.0, 1.0)
    amp = a[..., :12] * A_BOUND
    dur = (a[..., 12:] + 1.0) * 0.5 * (T_MAX - T_MIN) + T_MIN
    return np.concatenate([amp, dur], -1)


def theta_to_action(theta):
    t = np.asarray(theta, np.float32)
    amp = np.clip(t[..., :12] / A_BOUND, -1.0, 1.0)
    dur = np.clip((t[..., 12:] - T_MIN) / (T_MAX - T_MIN) * 2.0 - 1.0, -1.0, 1.0)
    return np.concatenate([amp, dur], -1)


# ------------------------------ option reward (certificate-aligned, per-step + terminal bonus) ------------------------------
@dataclass
class OptionReward:
    """Per-step shaped reward + terminal certificate bonus, composed from the spec's listed terms (the raw v3 per-step reward
    is anti-aligned, audited 7dc46f24). R_option is the DISCOUNTED SUM of these per-step rewards through the full
    carry→frozen-settle→K6 continuation, so the K6/handoff signal is not lost at the handoff boundary."""

    w_k6: float = 10.0            # terminal, at the end of settling
    w_handoff: float = 2.0        # once, when strict first reaches ≥1
    w_progress: float = 8.0       # per-step, for reducing distance-to-target
    w_contact: float = 0.05       # per-step, small, for keeping carry contact
    w_exit: float = 1.0           # per-step, for a fresh containment exit
    w_effort: float = 0.002       # per-step, small


REWARD_SCALE = 5.0                # fixed critic reward-scale (algorithmic only; same across SAC/TD3; K6 still selects models)


# ------------------------------ single-option boundary executor (semi-MDP, discounted return) ------------------------------
def execute_one_option(rl, gate, theta, pi0, base, *, gamma, horizon, reward=None, max_macro=60):
    """Execute ONE committed push→brake→release macro; on a valid handoff (strict≥1) switch to FROZEN pi_0 and settle to a
    terminal K6 decision (the option carries the WHOLE carry→settle→K6 consequence); else return a recovery state for the
    next option. Returns R_option = Σ_{j<τ} γ^j r_{t+j} (discounted, so the semi-MDP target is exactly R_option + γ^τ Q'),
    τ, done, s_next, certificate. Deterministic given (rl,gate,theta,γ,reward)."""
    rw = reward or OptionReward()
    a_push, T_push, a_brake, T_brake, a_release, T_release = _unpack(theta)
    dtz_prev = rl._dtz(); dtz_start = dtz_prev; dtz_min = dtz_prev; touched0 = rl._touched
    contain_exit = 0; was_contained = dtz_prev <= CENTER_TOL; effort = 0.0; contact_steps = 0
    R = 0.0; gp = 1.0; handoff_paid = False
    phase, tph, t = "push", 0, 0
    handed = aborted = False

    def _pay(r):                                                          # accumulate one discounted per-step reward
        nonlocal R, gp
        R += gp * r; gp *= gamma

    while t < max_macro and t < horizon:
        s = int(rl._strict); o48 = rl.obs()
        if s >= 1:
            handed = True; break
        if not (gate.gate == 1.0):
            a = _det(pi0, o48)
        elif phase == "push":
            a = a_push; tph += 1
            if tph >= T_push or rl._dtz() <= PUSH_DTZ:
                phase, tph = "brake", 0
        elif phase == "brake":
            a = a_brake; tph += 1
            if tph >= T_brake or rl._dtz() <= CENTER_TOL or rl._speed() < 1.5 * SETTLE_VEL:
                phase, tph = "release", 0
        else:
            a = a_release; tph += 1
        tb = rl._touched
        ca = np.clip(np.asarray(a, np.float32), -ACTION_SCALE, ACTION_SCALE)
        estep = float((ca ** 2).sum()); effort += estep
        _r, term, trunc = step_ablation(rl, ca, "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        m = rl.inner._planar_metrics; contact = int(m.left_contact or m.right_contact); contact_steps += contact
        dtz = rl._dtz(); dtz_min = min(dtz_min, dtz); t += 1
        exited = was_contained and dtz > CENTER_TOL
        contain_exit += int(exited); was_contained = dtz <= CENTER_TOL
        r_t = rw.w_progress * (dtz_prev - dtz) + rw.w_contact * contact - rw.w_exit * int(exited) - rw.w_effort * estep
        if int(rl._strict) >= 1 and not handoff_paid:
            r_t += rw.w_handoff; handoff_paid = True
        _pay(r_t); dtz_prev = dtz
        if term or trunc:
            handed = handed or int(rl._strict) >= 1; break
        if _safety_abort(rl, tb):
            aborted = True; break
        if phase == "release" and tph >= T_release:
            break                                                        # macro completed without handoff → re-decide
        if int(rl._strict) >= 1:
            handed = True; break
    md = int(rl._strict); touched = touched0 or rl._touched
    if handed:                                                           # settle via FROZEN pi_0 to a terminal K6 decision
        ts = 0
        while t + ts < horizon:
            s = int(rl._strict); o48 = rl.obs()
            a = _det(base, _aug(o48, s)) if (gate.gate == 1.0 and s >= 1) else _det(pi0, o48)
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            md = max(md, int(rl._strict)); touched = touched or rl._touched; dtz = rl._dtz()
            exited = was_contained and dtz > CENTER_TOL; contain_exit += int(exited); was_contained = dtz <= CENTER_TOL
            dtz_min = min(dtz_min, dtz)
            _pay(rw.w_progress * (dtz_prev - dtz) - rw.w_exit * int(exited)); dtz_prev = dtz; ts += 1
            if term or trunc:
                break
        tau = max(1, t + ts)
    else:
        tau = max(1, t)
    k6 = int(md >= HELD_DWELL and touched)
    done = bool(handed or aborted)
    if done:
        R += gp * (rw.w_k6 * k6)                                          # terminal K6 bonus, discounted to the terminal step
    return {"R_option": float(R), "k6": k6, "reached_handoff": int(handed or md >= 1), "contain_exit_ct": contain_exit,
            "effort": effort, "dtz_start": dtz_start, "dtz_min": dtz_min, "contact_frac": contact_steps / max(1, t), "tau": tau,
            "done": done, "s_next": None if done else rl.obs().copy().astype(np.float32)}


# ------------------------------ the fixed search-wrapper environment ------------------------------
class SearchWrapperEnv:
    """Fixed b-search wrapper: reset picks an option-initiation state; step(θ_center) samples b candidates AROUND θ_center,
    selects the best by the frozen local-search score, then executes ONE option boundary. b/dist/executor/score/pi_0 fixed."""

    def __init__(self, templates, pi0, base, reward, *, gamma=0.99, b=8, horizon=160, max_options=4, seed=0):
        self.templates = templates; self.pi0 = pi0; self.base = base; self.reward = reward; self.gamma = gamma
        self.b = b; self.horizon = horizon; self.max_options = max_options
        self.rng = np.random.default_rng(seed); self._i = None; self._opt = 0; self._search_ctr = 0

    def reset(self, idx=None):
        self._i = int(self.rng.integers(0, len(self.templates))) if idx is None else idx
        self._rl = copy.deepcopy(self.templates[self._i][0]); self._gate = copy.deepcopy(self.templates[self._i][1])
        self._opt = 0
        return self._rl.obs().copy().astype(np.float32)

    def step(self, action, *, search_seed=None):
        """action = normalised θ_center in [-1,1]^15. Fixed b-search selects θ_selected by the frozen score; ONE option
        boundary is executed; returns (s_next, R_option (discounted), done, info). The Bellman action stored by the caller is
        θ_center (this action), NOT θ_selected — θ_selected is provenance only."""
        center = action_to_theta(action)
        ss = self._search_ctr if search_seed is None else search_seed; self._search_ctr += 1
        theta_sel, _out = search_select(self._rl, self._gate, center, self.pi0, self.base, np.random.default_rng(ss), b=self.b, horizon=self.horizon)
        o = execute_one_option(self._rl, self._gate, theta_sel, self.pi0, self.base, gamma=self.gamma, horizon=self.horizon, reward=self.reward)
        self._opt += 1
        done = o["done"] or self._opt >= self.max_options
        s_next = o["s_next"] if (o["s_next"] is not None and not done) else self._rl.obs().copy().astype(np.float32)
        info = {"tau": o["tau"], "k6": o["k6"], "reached_handoff": o["reached_handoff"], "contain_exit_ct": o["contain_exit_ct"],
                "theta_selected": theta_sel.astype(np.float32), "theta_center": center.astype(np.float32),
                "terminal": bool(o["done"]), "truncated": bool(not o["done"] and self._opt >= self.max_options)}
        return s_next, float(o["R_option"]), bool(done), info


# ------------------------------ networks ------------------------------
class QNet(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS + DIM, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class DetActor(nn.Module):                                               # TD3
    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, DIM), nn.Tanh())

    def forward(self, s):
        return self.net(s)


class GaussActor(nn.Module):                                            # SAC (squashed Gaussian)
    LOG_STD = (-5.0, 2.0)

    def __init__(self, h=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, DIM); self.log_std = nn.Linear(h, DIM)

    def forward(self, s):
        x = self.body(s); return self.mu(x), self.log_std(x).clamp(*self.LOG_STD)

    def sample(self, s):
        mu, log_std = self(s); std = log_std.exp()
        n = torch.randn_like(std); pre = mu + std * n; a = torch.tanh(pre)
        logp = (-0.5 * (n ** 2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1) - torch.log(1 - a ** 2 + 1e-6).sum(-1)
        return a, logp

    def mean_action(self, s):
        return torch.tanh(self(s)[0])


# ------------------------------ replay + semi-MDP target ------------------------------
@dataclass
class OptionReplay:
    cap: int = 20000
    s: list = field(default_factory=list); a: list = field(default_factory=list); r: list = field(default_factory=list)
    tau: list = field(default_factory=list); s2: list = field(default_factory=list); done: list = field(default_factory=list)
    prov: list = field(default_factory=list)

    def add(self, s, a, r, tau, s2, done, prov):
        for buf, v in ((self.s, s), (self.a, a), (self.r, r), (self.tau, tau), (self.s2, s2), (self.done, done), (self.prov, prov)):
            buf.append(v)
        if len(self.s) > self.cap:
            for buf in (self.s, self.a, self.r, self.tau, self.s2, self.done, self.prov):
                del buf[0]

    def __len__(self):
        return len(self.s)

    def sample(self, n, rng):
        idx = rng.integers(0, len(self.s), n)
        t = lambda buf, dt: torch.as_tensor(np.asarray([buf[i] for i in idx], dt))
        return (t(self.s, np.float32), t(self.a, np.float32), t(self.r, np.float32), t(self.tau, np.float32),
                t(self.s2, np.float32), t(self.done, np.float32))


def smdp_target(r, gamma, tau, done, q_next):
    """Semi-MDP Bellman target: R_option + γ^τ · Q_next for non-terminal options, R_option alone at terminal. This uses
    γ^τ (option duration), NOT one-step γ — the property test asserts exactly this."""
    return r + (1.0 - done) * (gamma ** tau) * q_next


# ------------------------------ eval + init distillation ------------------------------
def eval_policy(actor, templates, pi0, base, *, b, horizon=160, seed0=8000, search_seeds=1):
    """Deployed-controller eval: deterministic actor mean → θ_center → FIXED b-search → selected option outcome. K6/exit on
    the paired panel with FIXED per-state search seeds, averaged over ``search_seeds`` (Bellman-safe eval-time smoothing —
    lowers the SELECTION variance without merging transitions)."""
    k6, ex = [], []
    for i, (rl, gate) in enumerate(templates):
        center = _actor_center(actor, rl.obs())
        kk, ee = [], []
        for j in range(search_seeds):
            _th, out = search_select(rl, gate, center, pi0, base, np.random.default_rng(seed0 + i * 131 + j), b=b, horizon=horizon)
            kk.append(int(out["k6"])); ee.append(int(out["contain_exit_ct"] > 0))
        k6.append(float(np.mean(kk))); ex.append(float(np.mean(ee)))
    return round(float(np.mean(k6)), 3), round(float(np.mean(ex)), 3)


def _actor_center(actor, obs):
    with torch.no_grad():
        ot = torch.as_tensor(obs[None]).float()
        a = actor.mean_action(ot)[0].numpy() if hasattr(actor, "mean_action") else actor(ot)[0].numpy()
    return action_to_theta(a)


def eval_paired(actor, proposal, templates, pi0, base, *, b, search_seeds, horizon=160, seed0=8000):
    """Per-state PAIRED eval, averaged over ``search_seeds`` fixed seeds (Bellman-safe: each search seed is a real,
    independent wrapper response; we average the per-state K6, we do NOT merge different transitions). Returns per-state
    lists (rl_k6, upd0_k6, rl_exit) for a paired bootstrap of ΔK6 = RL − its own update-0 proposal."""
    rl_k6, up_k6, rl_ex = [], [], []
    for i, (rl, gate) in enumerate(templates):
        c_rl = _actor_center(actor, rl.obs()); c_up = proposal.theta(rl.obs())
        kk, uu, ee = [], [], []
        for j in range(search_seeds):
            sd = seed0 + i * 131 + j
            _t1, o1 = search_select(rl, gate, c_rl, pi0, base, np.random.default_rng(sd), b=b, horizon=horizon)
            _t2, o2 = search_select(rl, gate, c_up, pi0, base, np.random.default_rng(sd), b=b, horizon=horizon)
            kk.append(int(o1["k6"])); uu.append(int(o2["k6"])); ee.append(int(o1["contain_exit_ct"] > 0))
        rl_k6.append(float(np.mean(kk))); up_k6.append(float(np.mean(uu))); rl_ex.append(float(np.mean(ee)))
    return rl_k6, up_k6, rl_ex


def distill_actor(actor, proposal, obs, *, epochs, lr=1e-3, seed=0):
    """Identical RL init: distil the deterministic proposal (obs→θ_center) into the actor's MEAN map (MSE in action space).
    Single-valued target ⇒ MSE is correct here (unlike distilling the multimodal teacher)."""
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(obs, np.float32))
    y = torch.as_tensor(theta_to_action(proposal.theta(np.asarray(obs, np.float32))).astype(np.float32))
    opt = torch.optim.Adam(actor.parameters(), lr); loss = torch.tensor(0.0)
    for _ in range(epochs):
        pred = actor.mean_action(x) if hasattr(actor, "mean_action") else actor(x)
        loss = ((pred - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


# ------------------------------ SAC / TD3 (shared semi-MDP loop) ------------------------------
@dataclass
class RLConfig:
    gamma: float = 0.99
    tau_polyak: float = 0.01
    lr: float = 3e-4
    batch: int = 128
    warmup_options: int = 40
    total_options: int = 600
    updates_per_option: int = 1
    eval_every: int = 120
    b: int = 8
    horizon: int = 160
    max_options: int = 4
    policy_delay: int = 2         # TD3
    target_noise: float = 0.15
    noise_clip: float = 0.3
    expl_noise: float = 0.2       # TD3 exploration
    alpha: float = 0.1            # SAC entropy temperature (fixed)


def _polyak(tgt, src, tau):
    for tp, sp in zip(tgt.parameters(), src.parameters()):
        tp.data.mul_(1 - tau).add_(tau * sp.data)


def train_agent(algo, env, actor, dev_panel, pi0, base, cfg, log, *, seed=0):
    """Semi-MDP SAC ('sac') or TD3 ('td3') over the proposal center. Critic Q(s, θ_center); actor trained through the critic
    only (never through the black-box search). Model selection on dev K6. Returns (checkpoints, history)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    q1, q2 = QNet(), QNet(); q1t, q2t = QNet(), QNet(); q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    qopt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), cfg.lr)
    if algo == "td3":
        at = DetActor(); at.load_state_dict(actor.state_dict())
    aopt = torch.optim.Adam(actor.parameters(), cfg.lr)
    replay = OptionReplay(); history = []; ckpts = {}
    ckpts["update0"] = copy.deepcopy(actor.state_dict())
    best_val = -1.0; ckpts["best_val"] = copy.deepcopy(actor.state_dict()); upd = 0

    def act(o, explore):
        with torch.no_grad():
            ot = torch.as_tensor(o[None]).float()
            if algo == "sac":
                a = (actor.sample(ot)[0] if explore else actor.mean_action(ot))[0].numpy()
            else:
                a = actor(ot)[0].numpy()
                if explore:
                    a = np.clip(a + rng.normal(0, cfg.expl_noise, DIM).astype(np.float32), -1, 1)
        return a

    s = env.reset(); k6run = []
    for it in range(cfg.total_options):
        a = act(s, explore=(it >= cfg.warmup_options)) if it >= cfg.warmup_options else np.clip(rng.uniform(-1, 1, DIM).astype(np.float32), -1, 1)
        s2, r, done, info = env.step(a)
        # Bellman action = the actor's PROPOSAL-CENTER action a (θ_center); θ_selected is provenance only, never the stored action
        replay.add(s, np.asarray(a, np.float32), r / REWARD_SCALE, info["tau"], s2, float(info["terminal"]),
                   {"theta_center": info["theta_center"], "theta_selected": info["theta_selected"], "k6": info["k6"], "tau": info["tau"], "reached_handoff": info["reached_handoff"], "exit": info["contain_exit_ct"]})
        k6run.append(info["k6"]); s = env.reset() if done else s2
        if len(replay) >= cfg.batch and it >= cfg.warmup_options:
            for _ in range(cfg.updates_per_option):
                upd += 1
                bs, ba, br, bt, bs2, bd = replay.sample(cfg.batch, rng)
                with torch.no_grad():
                    if algo == "sac":
                        a2, logp2 = actor.sample(bs2); q_next = torch.min(q1t(bs2, a2), q2t(bs2, a2)) - cfg.alpha * logp2
                    else:
                        noise = (torch.randn_like(ba) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
                        a2 = (at(bs2) + noise).clamp(-1, 1); q_next = torch.min(q1t(bs2, a2), q2t(bs2, a2))
                    y = smdp_target(br, cfg.gamma, bt, bd, q_next)
                ql = ((q1(bs, ba) - y) ** 2).mean() + ((q2(bs, ba) - y) ** 2).mean()
                qopt.zero_grad(); ql.backward(); qopt.step()
                if algo == "sac":
                    ap, logp = actor.sample(bs); al = (cfg.alpha * logp - torch.min(q1(bs, ap), q2(bs, ap))).mean()
                    aopt.zero_grad(); al.backward(); aopt.step()
                    _polyak(q1t, q1, cfg.tau_polyak); _polyak(q2t, q2, cfg.tau_polyak)
                elif upd % cfg.policy_delay == 0:
                    al = -q1(bs, actor(bs)).mean(); aopt.zero_grad(); al.backward(); aopt.step()
                    _polyak(q1t, q1, cfg.tau_polyak); _polyak(q2t, q2, cfg.tau_polyak); _polyak(at, actor, cfg.tau_polyak)
        if (it + 1) % cfg.eval_every == 0 or it == cfg.total_options - 1:
            vk6, vex = eval_policy(actor, dev_panel, pi0, base, b=cfg.b, horizon=cfg.horizon)
            if len(replay) >= 8:
                with torch.no_grad():
                    _qs = replay.sample(min(64, len(replay)), rng); qm = float(q1(_qs[0], _qs[1]).mean())
            else:
                qm = 0.0
            recent = float(np.mean(k6run[-cfg.eval_every:])) if k6run else 0.0
            history.append({"it": it + 1, "dev_k6": vk6, "dev_exit": vex, "train_k6_recent": round(recent, 3), "q_mean": round(qm, 2)})
            log(f"    [{algo} it {it+1}/{cfg.total_options}] dev K6 {vk6} exit {vex} | train_k6 {round(recent,3)} | Q~{round(qm,2)} | replay {len(replay)}")
            if it + 1 >= cfg.total_options // 4 and "early" not in ckpts:
                ckpts["early"] = copy.deepcopy(actor.state_dict())
            if it + 1 >= cfg.total_options // 2 and "mid" not in ckpts:
                ckpts["mid"] = copy.deepcopy(actor.state_dict())
            if vk6 > best_val:
                best_val = vk6; ckpts["best_val"] = copy.deepcopy(actor.state_dict())
    ckpts["final"] = copy.deepcopy(actor.state_dict())
    ckpts.setdefault("early", ckpts["update0"]); ckpts.setdefault("mid", ckpts.get("best_val"))
    return ckpts, history
