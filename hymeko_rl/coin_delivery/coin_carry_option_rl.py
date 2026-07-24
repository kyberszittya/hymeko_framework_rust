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
from dataclasses import dataclass

import numpy as np
import torch

# The semi-MDP mechanism now lives in the framework engine (hymeko_rl.option_rl); this module is the COIN task adapter.
from hymeko_rl.option_rl import SemiMDPConfig, smdp_target, train_semi_mdp  # noqa: F401 (smdp_target re-exported for coin callers)
from hymeko_rl.option_rl import OptionReplayBuffer as OptionReplay  # noqa: F401 (coin-compat re-export)
from hymeko_rl.option_rl.agents import DetActor as _FDetActor, GaussActor as _FGaussActor, QNet as _FQNet

from hymeko_rl.coin_delivery.coin_carry_fsm import load_carry_automaton
from hymeko_rl.coin_delivery.coin_carry_monitor import TraceSample, load_carry_monitor_spec, make_monitor
from hymeko_rl.coin_delivery.coin_carry_option import _safety_abort
from hymeko_rl.coin_delivery.coin_carry_proposal import search_select
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, DIM, PUSH_DTZ, T_MAX, T_MIN, _unpack
from hymeko_rl.coin_delivery.coin_markov_ablation_train import ACTION_SCALE, _aug, _det
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, SETTLE_VEL  # HELD_DWELL now owned by the trace-monitor (§2A)
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
def _nxt(spec, phase, event, default):
    """Destination of ``phase`` on ``event`` — from the parsed .hymeko automaton when ``spec`` is given, else the hard-coded
    default. This is the ONLY thing the automaton drives; the guard predicates + their timing stay identical below."""
    if spec is None:
        return default
    dst, _ev = spec.step(phase, lambda e: e == event)
    return dst


def execute_one_option(rl, gate, theta, pi0, base, *, gamma, horizon, reward=None, max_macro=60, spec="auto", trace=None,
                       monitor=None, verify_shadow=False):
    """Execute ONE committed push→brake→release macro; on a valid handoff (strict≥1) switch to FROZEN pi_0 and settle to a
    terminal K6 decision (the option carries the WHOLE carry→settle→K6 consequence); else return a recovery state for the
    next option. Returns R_option = Σ_{j<τ} γ^j r_{t+j} (discounted, so the semi-MDP target is exactly R_option + γ^τ Q'),
    τ, done, s_next, certificate. Deterministic given (rl,gate,theta,γ,reward).

    §1: ``spec`` (a `ControllerSpec` from ``coin_carry_option_v1.hymeko``) sources the phase TRANSITION TOPOLOGY (``spec=None``
    = the gated hard-coded fallback). §2A: an online trace-``monitor`` (semantics from the same `.hymeko` `@certificate`)
    computes the SINGLE delivery verdict (strict/handoff/containment/K6) from the per-step physical trace — the env's
    ``rl._strict`` is a SHADOW, out of the decision path (``verify_shadow`` fail-closes if the monitor and the shadow diverge).
    ``trace`` records the per-step phase label + terminal marker."""
    if spec == "auto":                                                   # default: the .hymeko automaton is the runtime truth
        spec = load_carry_automaton()
    if monitor is None:                                                  # default: the .hymeko-sourced online certificate monitor
        monitor = make_monitor("python")
    rw = reward or OptionReward()
    a_push, T_push, a_brake, T_brake, a_release, T_release = _unpack(theta)
    dtz_prev = rl._dtz(); dtz_start = dtz_prev; dtz_min = dtz_prev
    monitor.reset(load_carry_monitor_spec(), dtz_start, bool(rl._touched))
    effort = 0.0; contact_steps = 0
    R = 0.0; gp = 1.0; handoff_paid = False
    phase, tph, t = "PUSH", 0, 0
    mstrict = 0                                                          # the MONITOR's strict (the certificate driver)
    handed = aborted = False

    def _pay(r):
        nonlocal R, gp
        R += gp * r; gp *= gamma

    def _mark(m):
        if trace is not None:
            trace.append(m)

    def _observe(term, trunc):                                          # feed one physical sample → monitor; shadow-check
        nonlocal mstrict
        m = rl.inner._planar_metrics; contact = int(m.left_contact or m.right_contact)
        res = monitor.observe(TraceSample(dtz=rl._dtz(), speed=rl._speed(), touched=bool(rl._touched),
                                          contact=bool(contact), terminated=bool(term or trunc)))
        mstrict = res["strict"]
        if verify_shadow and mstrict != int(rl._strict):
            raise AssertionError(f"MONITOR/shadow strict divergence at step {t}: monitor {mstrict} != env {int(rl._strict)}")
        return res, contact

    while t < max_macro and t < horizon:
        o48 = rl.obs()
        if mstrict >= 1:                                                 # handoff = monitor strict ≥ 1 (pre-action)
            _mark(_nxt(spec, phase, "handoff", "HANDOFF")); handed = True; break
        cur = phase
        if not (gate.gate == 1.0):
            a = _det(pi0, o48)
        elif phase == "PUSH":
            a = a_push; tph += 1
            ev = "push_reached" if rl._dtz() <= PUSH_DTZ else ("push_timeout" if tph >= T_push else None)
            if ev:
                phase, tph = _nxt(spec, "PUSH", ev, "BRAKE"), 0
        elif phase == "BRAKE":
            a = a_brake; tph += 1
            ev = ("brake_centered" if rl._dtz() <= CENTER_TOL else
                  ("brake_slow" if rl._speed() < 1.5 * SETTLE_VEL else ("brake_timeout" if tph >= T_brake else None)))
            if ev:
                phase, tph = _nxt(spec, "BRAKE", ev, "RELEASE"), 0
        else:
            a = a_release; tph += 1
        _mark(cur)
        tb = rl._touched
        ca = np.clip(np.asarray(a, np.float32), -ACTION_SCALE, ACTION_SCALE)
        estep = float((ca ** 2).sum()); effort += estep
        _r, term, trunc = step_ablation(rl, ca, "A")
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        dtz = rl._dtz(); dtz_min = min(dtz_min, dtz); t += 1
        res, contact = _observe(term, trunc); contact_steps += contact   # monitor computes strict/containment (single source)
        r_t = rw.w_progress * (dtz_prev - dtz) + rw.w_contact * contact - rw.w_exit * res["exited"] - rw.w_effort * estep
        if mstrict >= 1 and not handoff_paid:
            r_t += rw.w_handoff; handoff_paid = True
        _pay(r_t); dtz_prev = dtz
        if term or trunc:
            handed = handed or mstrict >= 1; _mark("HANDOFF" if handed else "SETTLED"); break
        if _safety_abort(rl, tb):
            _mark(_nxt(spec, cur, "abort", "ABORTED")); aborted = True; break
        if cur == "RELEASE" and tph >= T_release:
            _mark(_nxt(spec, "RELEASE", "release_done", "COMPLETED")); break   # macro completed without handoff → re-decide
        if mstrict >= 1:
            _mark(_nxt(spec, cur, "handoff", "HANDOFF")); handed = True; break
    if handed:                                                          # settle via FROZEN pi_0 (HANDOFF phase law) to terminal K6
        ts = 0
        while t + ts < horizon:
            o48 = rl.obs()
            a = _det(base, _aug(o48, mstrict)) if (gate.gate == 1.0 and mstrict >= 1) else _det(pi0, o48)
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            dtz = rl._dtz(); dtz_min = min(dtz_min, dtz)
            res, _c = _observe(term, trunc)
            _pay(rw.w_progress * (dtz_prev - dtz) - rw.w_exit * res["exited"]); dtz_prev = dtz; ts += 1; _mark("HANDOFF")
            if term or trunc:
                break
        tau = max(1, t + ts)
    else:
        tau = max(1, t)
        if trace is not None and trace and trace[-1] in ("PUSH", "BRAKE", "RELEASE"):
            trace.append("COMPLETED")                                    # macro exhausted its step budget w/o handoff → re-decide
    v = monitor.verdict()                                               # THE single certificate: k6 / handoff / containment-exit
    if handed:
        _mark(_nxt(spec, "HANDOFF", "delivered", "DELIVERED") if v["k6"] else _nxt(spec, "HANDOFF", "settle_horizon", "SETTLED"))
    done = bool(handed or aborted)
    if done:
        R += gp * (rw.w_k6 * v["k6"])                                    # terminal K6 bonus, discounted to the terminal step
    return {"R_option": float(R), "k6": v["k6"], "reached_handoff": v["reached_handoff"], "contain_exit_ct": v["contain_exit_ct"],
            "effort": effort, "dtz_start": dtz_start, "dtz_min": dtz_min, "contact_frac": contact_steps / max(1, t), "tau": tau,
            "done": done, "s_next": None if done else rl.obs().copy().astype(np.float32), "monitor_events": v["events"]}


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


# ------------------------------ networks (coin-bound framework nets: obs48 → θ-center 15-d) ------------------------------
# The architectures live in hymeko_rl.option_rl.agents; these thin subclasses bind the coin dims (OBS/DIM) so `QNet()`/
# `GaussActor()`/`DetActor()` keep their no-arg coin API AND identical state_dict keys (prior coin checkpoints load unchanged).
class QNet(_FQNet):
    def __init__(self, h=256):
        super().__init__(OBS, DIM, h)


class DetActor(_FDetActor):
    def __init__(self, h=256):
        super().__init__(OBS, DIM, h)


class GaussActor(_FGaussActor):
    def __init__(self, h=256):
        super().__init__(OBS, DIM, h)


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


# ------------------------------ coin RL config + train adapter over the framework engine ------------------------------
@dataclass
class RLConfig:
    """Coin RL config = the framework semi-MDP hyperparameters + the coin env params (b / horizon / max_options)."""

    gamma: float = 0.99
    tau_polyak: float = 0.01
    lr: float = 3e-4
    batch: int = 128
    warmup_options: int = 40
    total_options: int = 600
    updates_per_option: int = 1
    eval_every: int = 120
    b: int = 8                    # coin: fixed search budget
    horizon: int = 160            # coin: option/settle horizon
    max_options: int = 4          # coin: options per episode
    policy_delay: int = 2
    target_noise: float = 0.15
    noise_clip: float = 0.3
    expl_noise: float = 0.2
    alpha: float = 0.1

    def to_semi_mdp(self):
        return SemiMDPConfig(gamma=self.gamma, tau_polyak=self.tau_polyak, lr=self.lr, batch=self.batch,
                             warmup_options=self.warmup_options, total_options=self.total_options,
                             updates_per_option=self.updates_per_option, eval_every=self.eval_every, reward_scale=REWARD_SCALE,
                             policy_delay=self.policy_delay, target_noise=self.target_noise, noise_clip=self.noise_clip,
                             expl_noise=self.expl_noise, alpha=self.alpha)


def train_agent(algo, env, actor, dev_panel, pi0, base, cfg, log, *, seed=0):
    """Coin adapter over the framework `train_semi_mdp`: supplies the coin dev-eval (K6 on the held-out panel) and maps the
    coin RLConfig → SemiMDPConfig. The engine owns the semi-MDP mechanism; this only wires the coin task in. History keys are
    kept backward-compatible (``dev_k6``/``dev_exit``/``train_k6_recent``)."""
    def dev_eval_fn(a):
        k6, ex = eval_policy(a, dev_panel, pi0, base, b=cfg.b, horizon=cfg.horizon)
        return k6, {"exit": ex}

    ckpts, hist = train_semi_mdp(algo, env, actor, dev_eval_fn, cfg.to_semi_mdp(), obs_dim=OBS, act_dim=DIM, log=log, seed=seed)
    for h in hist:                                                        # framework keys → coin-compat keys
        h["dev_k6"] = h.pop("dev_score"); h["dev_exit"] = h.pop("aux", {}).get("exit"); h["train_k6_recent"] = h.pop("train_recent")
    return ckpts, hist
