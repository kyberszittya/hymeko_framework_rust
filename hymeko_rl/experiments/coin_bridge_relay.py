"""Coin bridge-relay solution — solve the measured distribution-connectivity problem the F11/F21 campaign exposed.

F21 showed the TRANSPORT actor was STARVED (1.6% occupancy): the controller almost never enters a transport-ready
state, so a dedicated transport policy never gets to run. The bridge-relay decouples the two responsibilities on the
SAME task: a **frozen TRANSPORT_POLICY** (the verified +0.0253 clear-start policy, 10/10 strict) owns stable
transport → certified delivery; a learned **BRIDGE_POLICY** owns approach / contact acquisition / bilateral bracketing
→ *entry into an empirically verified TRANSPORT_READY state*. A rule-based **relay** starts in the bridge and hands off
to the frozen transport policy once readiness holds for a short hysteresis window (falls back on loss). The readiness
basin is measured (not guessed): a state is TRANSPORT_READY only when the frozen transport policy independently finishes
from it. The delivery-v2b reward, strict predicate, env, and frozen transport policy are never changed; bridge shaping
is a SEPARATE training signal and the final evaluation is always the canonical strict certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash
from hymeko_rl.env.planar_snapshot import PlanarSnapshot, snapshot_planar
from hymeko_rl.eval.team_tensor import field_index
from hymeko_rl.experiments.coin_generator_exp import _restore_generated
from hymeko_rl.experiments.coin_two_arm_sac import policy_strict
from hymeko_rl.train.coin_delivery_actor import _attribution_from_trace, rollout
from hymeko_rl.train.sac import build_sac

TRANSPORT_CKPT = Path("experiments/2026_07_21_coin_clearance_curriculum/run_s0/actor_best.pt")
_ACT = 6

# Named public fields for the readiness detector (NO raw indices in production logic — resolved once, by name).
_FEAT_NAMES = ("l_to_coin_x", "l_to_coin_y", "r_to_coin_x", "r_to_coin_y", "left_contact", "right_contact",
               "both_contact", "arm_body_contact", "aperture", "coin_vx", "coin_vy",
               "coin_to_target_x", "coin_to_target_y", "mid_to_coin_x", "mid_to_coin_y")
_FEAT_IDX = [field_index(n) for n in _FEAT_NAMES]
_I_BOTH, _I_BODY = field_index("both_contact"), field_index("arm_body_contact")
_I_LEFT, _I_RIGHT = field_index("left_contact"), field_index("right_contact")


def ready_features(obs: np.ndarray) -> np.ndarray:
    """The named-field feature vector the readiness detector operates on (geometry + contact + velocity, §2.3).
    Sanitised: the env can emit a transient ``inf`` velocity on a ``dt=0`` step (restore boundary); left unbounded it
    poisons the kNN distance → NaN reward. Clip to a finite bound (the detector is scale-standardised anyway)."""
    f = np.asarray(obs, np.float32)[..., _FEAT_IDX]
    return np.nan_to_num(f, nan=0.0, posinf=10.0, neginf=-10.0)


def load_transport_policy(ckpt: Path = TRANSPORT_CKPT) -> Any:
    actor, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=_ACT, action_scale=1.0)
    actor.load_state_dict(torch.load(ckpt, map_location="cpu"))
    actor.eval()
    return actor


def greedy_fn(actor: Any) -> Callable[[Any, int, np.ndarray], np.ndarray]:
    def g(_inner: Any, _t: int, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    return g


def stochastic_fn(actor: Any) -> Callable[[Any, int, np.ndarray], np.ndarray]:
    def g(_inner: Any, _t: int, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return actor.sample(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32))[0].numpy()[0]
    return g


# ── Phase 2: the empirical TRANSPORT_READY basin ──────────────────────────────────────────────────────────────────
@dataclass
class ReadyLabel:
    """One labelled candidate state: its snapshot, named features, hash, label, and the frozen-policy evidence."""
    snapshot: PlanarSnapshot
    features: np.ndarray
    state_hash: str
    label: str                 # TRANSPORT_READY / LOOSE_READY / CONTACT_ONLY / NOT_READY / INVALID
    greedy_strict: bool        # the deploy-matched signal: frozen GREEDY transport certifies from here
    robustness: int            # of n low-noise stochastic rollouts, how many also certify (basin solidity)
    progress: float
    clean: bool
    body: bool


def label_readiness(env: Any, transport: Any, snap: PlanarSnapshot, *, n: int = 10) -> ReadyLabel:
    """Restore the exact state and run the FROZEN transport policy. The label is DEPLOY-MATCHED: the relay deploys the
    GREEDY transport policy, so TRANSPORT_READY requires the deterministic greedy policy to certify (progress>0 ∧ clean
    ∧ ¬body); the ``n`` seeded stochastic rollouts give a *robustness* count (how solidly in the basin), not the label
    (the stochastic policy is noticeably weaker — it degrades strict→loose — so it would mislabel solid states)."""
    obs0 = _restore_generated(env, snap)
    feats = ready_features(obs0)
    sh = snapshot_hash(snap)
    tr = rollout(env, greedy_fn(transport), max_steps=60)           # deterministic deploy signal
    att = _attribution_from_trace(tr)
    ff = att.fingertip_fraction
    clean = (min(att.alpha_L, att.alpha_R) / (ff + 1e-9)) >= 0.15
    greedy_strict = bool(policy_strict(tr))
    body_bad = att.alpha_body > 0.30
    robust = 0
    for i in range(n):
        _restore_generated(env, snap)
        torch.manual_seed(9_000 + i)
        robust += int(bool(policy_strict(rollout(env, stochastic_fn(transport), max_steps=60))))
    if greedy_strict and tr.progress > 0 and clean and not body_bad:
        label = "TRANSPORT_READY"
    elif tr.loose:
        label = "LOOSE_READY"
    elif bool(obs0[_I_LEFT] > 0.5 or obs0[_I_RIGHT] > 0.5):
        label = "CONTACT_ONLY"
    else:
        label = "NOT_READY"
    return ReadyLabel(snap, feats, sh, label, greedy_strict, robust, round(float(tr.progress), 4), clean, body_bad)


def collect_trajectory_states(env: Any, actor: Any, snap: PlanarSnapshot, *, stride: int = 3,
                              max_steps: int = 60) -> list[PlanarSnapshot]:
    """Roll out ``actor`` from ``snap`` and snapshot the physical state every ``stride`` steps (candidate collection)."""
    _restore_generated(env, snap)
    obs = env._last_obs
    out: list[PlanarSnapshot] = []
    g = greedy_fn(actor)
    for t in range(max_steps):
        if t % stride == 0:
            out.append(snapshot_planar(env.inner))
        a = np.clip(g(env.inner, t, obs), -1, 1).astype(np.float32)
        obs, _r, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return out


# ── Phase 2.3: the readiness detector (nearest-ready-state, named features, frozen threshold + hysteresis) ──────────
class ReadinessDetector:
    """Deterministic nearest-TRANSPORT_READY-state detector over named features (§2.3): a state is *ready* when its
    scaled distance to the nearest fitted READY state is ≤ ``enter_thresh``; separate ``exit_thresh`` gives hysteresis.
    No reward critic — a pure handoff mechanism. Distances are per-feature standardised on the READY bank."""

    def __init__(self, ready_feats: np.ndarray, *, enter_thresh: float, exit_thresh: float, k: int = 1) -> None:
        if ready_feats.ndim != 2 or len(ready_feats) == 0:
            raise ValueError("ReadinessDetector needs a non-empty (N, F) READY feature bank")
        self._mu = ready_feats.mean(0)
        self._sd = ready_feats.std(0) + 1e-6
        self._bank = (ready_feats - self._mu) / self._sd
        self.enter_thresh, self.exit_thresh, self.k = float(enter_thresh), float(exit_thresh), int(k)

    def distance(self, obs: np.ndarray) -> float:
        return self.distance_from_features(ready_features(obs))

    def distance_from_features(self, feats: np.ndarray) -> float:
        z = (np.asarray(feats, np.float32) - self._mu) / self._sd
        d = np.linalg.norm(self._bank - z[None], axis=1)
        return float(np.mean(np.sort(d)[: self.k]))

    def is_ready(self, obs: np.ndarray, *, currently_transport: bool) -> bool:
        thr = self.exit_thresh if currently_transport else self.enter_thresh
        return self.distance(obs) <= thr


def calibrate_thresholds(detector_feats: np.ndarray, labels: list[str], feats: np.ndarray) -> tuple[float, float]:
    """Pick enter/exit thresholds from the fitted bank. enter = a high quantile of each READY state's distance to its
    nearest OTHER READY state (self excluded — a state's nearest neighbour is itself at 0, which would collapse the
    threshold), i.e. the basin's internal spread; exit = 2× (hysteresis). Reported with confusion counts by the caller."""
    mu, sd = detector_feats.mean(0), detector_feats.std(0) + 1e-6
    bank = (detector_feats - mu) / sd
    ready_nn = []
    for f, lab in zip(feats, labels):
        if lab != "TRANSPORT_READY":
            continue
        z = (f - mu) / sd
        d = np.sort(np.linalg.norm(bank - z[None], axis=1))
        ready_nn.append(float(d[1] if len(d) > 1 else d[0]))        # skip self (distance 0)
    enter = float(np.quantile(ready_nn, 0.75)) if ready_nn else 1.0
    enter = max(enter, 1e-3)
    return enter, enter * 2.0


# ── Phase 5.3: the relay controller (bridge → frozen transport, handoff on readiness hysteresis) ───────────────────
@dataclass
class RelayLog:
    bridge_steps: int = 0
    transport_steps: int = 0
    handoffs: int = 0
    fallbacks: int = 0
    handoff_step: int = -1
    first_contact_step: int = -1
    bilateral_step: int = -1
    ready_step: int = -1
    mode_trace: list[str] = field(default_factory=list)


class RelayController:
    """Start in BRIDGE_POLICY; switch to the frozen TRANSPORT_POLICY once the detector reports ready for
    ``hysteresis`` consecutive steps; fall back to BRIDGE on readiness loss / bilateral loss / body contact / stall."""

    def __init__(self, bridge: Any, transport: Any, detector: ReadinessDetector, *, hysteresis: int = 3,
                 stall_window: int = 8) -> None:
        self._bridge_g, self._transport_g = greedy_fn(bridge), greedy_fn(transport)
        self._det = detector
        self._hyst, self._stall_window = int(hysteresis), int(stall_window)

    def act_fn(self, log: RelayLog) -> Callable[[Any, int, np.ndarray], np.ndarray]:
        state = {"mode": "BRIDGE", "ready_run": 0, "no_prog": 0, "last_dtz": None}

        def act(inner: Any, t: int, obs: np.ndarray) -> np.ndarray:
            o = np.asarray(obs)
            in_transport = state["mode"] == "TRANSPORT"
            if o[_I_LEFT] > 0.5 or o[_I_RIGHT] > 0.5:
                if log.first_contact_step < 0:
                    log.first_contact_step = t
            if o[_I_BOTH] > 0.5 and log.bilateral_step < 0:
                log.bilateral_step = t
            ready = self._det.is_ready(o, currently_transport=in_transport)
            if ready and log.ready_step < 0:
                log.ready_step = t
            if not in_transport:
                state["ready_run"] = state["ready_run"] + 1 if ready else 0
                if state["ready_run"] >= self._hyst:                      # valid handoff
                    state["mode"] = "TRANSPORT"
                    log.handoffs += 1
                    log.handoff_step = t
            else:
                body = o[_I_BODY] > 0.5
                lost = o[_I_BOTH] < 0.5
                if not ready or body or lost or state["no_prog"] >= self._stall_window:
                    state["mode"] = "BRIDGE"                              # fall back
                    log.fallbacks += 1
                    state["ready_run"] = 0
            in_transport = state["mode"] == "TRANSPORT"
            log.mode_trace.append("T" if in_transport else "B")
            (log.__setattr__("transport_steps", log.transport_steps + 1) if in_transport
             else log.__setattr__("bridge_steps", log.bridge_steps + 1))
            return (self._transport_g if in_transport else self._bridge_g)(inner, t, o)
        return act


def relay_rollout(env: Any, bridge: Any, transport: Any, detector: ReadinessDetector, snap: PlanarSnapshot, *,
                  hysteresis: int = 3, max_steps: int = 60) -> "tuple[Any, RelayLog]":
    """One composite relay episode from ``snap``; returns (canonical RolloutTrace, RelayLog). The trace is scored by
    the canonical strict predicate — the relay only chooses which frozen/learned action to apply each step."""
    _restore_generated(env, snap)
    log = RelayLog()
    ctrl = RelayController(bridge, transport, detector, hysteresis=hysteresis)
    trace = rollout(env, ctrl.act_fn(log), max_steps=max_steps)
    return trace, log


# ── Phase 5.2: the bridge-reward training env (SEPARATE from delivery-v2b; terminates on entering the basin) ────────
@dataclass
class BridgeReward:
    """Pre-registered bridge-only shaping weights (§5.2). The terminal READY bonus dominates the local shaping."""
    w_potential: float = 12.0      # potential-based: decrease in scaled distance to the nearest READY state
    w_first_contact: float = 1.0   # one-time: first fingertip contact
    w_bilateral: float = 2.0       # one-time: one-sided → bilateral transition
    w_hold: float = 0.1            # per-step: persistent valid bilateral (both ∧ ¬body)
    w_body: float = 2.0            # penalty: arm-body shove
    w_action: float = 0.01         # penalty: excessive action energy
    r_ready: float = 20.0          # terminal: entered a verified TRANSPORT_READY state (dominant)


class BridgeRewardEnv:
    """Wraps the coin ``direct_env`` with the bridge reward: shape toward the nearest verified READY state and
    TERMINATE with a dominant bonus on entering the basin. delivery-v2b reward / strict predicate / the env physics are
    untouched — this reward is a SEPARATE training signal, never reused for the final evaluation (which is the strict
    certificate on the real env). gym-shaped so ``train_sac`` drives it unchanged."""

    def __init__(self, inner_env: Any, detector: ReadinessDetector, reset_pool: list[PlanarSnapshot],
                 rng: np.random.Generator, reward: BridgeReward | None = None) -> None:
        if not reset_pool:
            raise ValueError("BridgeRewardEnv needs a non-empty reset pool of clear-start training states")
        self.env = inner_env
        self._det = detector
        self._rw = reward or BridgeReward()
        self._pool = reset_pool
        self._rng = rng
        self.observation_space = inner_env.observation_space
        self.action_space = inner_env.action_space
        self.max_steps = getattr(inner_env, "max_steps", 60)
        self._prev_dist = 0.0
        self._had_contact = False
        self._had_bilateral = False

    def reset_to(self, snap: PlanarSnapshot) -> np.ndarray:
        obs = _restore_generated(self.env, snap)
        self._prev_dist = self._det.distance(obs)
        self._had_contact = bool(obs[_I_LEFT] > 0.5 or obs[_I_RIGHT] > 0.5)
        self._had_bilateral = bool(obs[_I_BOTH] > 0.5)
        return obs

    def reset(self, *, seed: int | None = None):                    # restore a FRESH random clear-start state each episode
        return self.reset_to(self._pool[self._rng.integers(len(self._pool))]), {}

    def step(self, action: np.ndarray) -> "tuple[np.ndarray, float, bool, bool, dict]":
        obs, _r_v2b, _term, trunc, info = self.env.step(action)
        rw = self._rw
        dist = self._det.distance(obs)
        r = rw.w_potential * (self._prev_dist - dist)               # potential-based shaping (toward the basin)
        self._prev_dist = dist
        left, right = obs[_I_LEFT] > 0.5, obs[_I_RIGHT] > 0.5
        both, body = obs[_I_BOTH] > 0.5, obs[_I_BODY] > 0.5
        if (left or right) and not self._had_contact:
            r += rw.w_first_contact
            self._had_contact = True
        if both and not self._had_bilateral:
            r += rw.w_bilateral
            self._had_bilateral = True
        if both and not body:
            r += rw.w_hold
        if body:
            r -= rw.w_body
        r -= rw.w_action * float(np.mean(np.square(action)))
        ready = self._det.is_ready(obs, currently_transport=False)
        terminated = bool(ready)
        if terminated:
            r += rw.r_ready                                          # dominant terminal bonus for entering the basin
        return obs, float(r), terminated, bool(trunc), info


# ── Phase 2/3 orchestration: build the basin + detector, then train & evaluate the relay ───────────────────────────
def build_basin(env: Any, transport: Any, seed_snaps: list[PlanarSnapshot], *, stride: int = 2,
                n_robust: int = 10) -> "tuple[list[ReadyLabel], ReadinessDetector]":
    """Collect candidate states (successful greedy trajectories from the seed states + the seed states themselves),
    label readiness with the FROZEN transport policy, and fit the nearest-ready detector on the TRANSPORT_READY bank."""
    cands: list[PlanarSnapshot] = []
    for s in seed_snaps:
        cands.extend(collect_trajectory_states(env, transport, s, stride=stride))
        cands.append(s)
    labels = [label_readiness(env, transport, s, n=n_robust) for s in cands]
    ready = [lab for lab in labels if lab.label == "TRANSPORT_READY"]
    if not ready:
        raise RuntimeError("no TRANSPORT_READY states found — cannot build a basin")
    feats = np.stack([lab.features for lab in ready])
    enter, exit_ = calibrate_thresholds(feats, [lab.label for lab in labels], np.stack([lab.features for lab in labels]))
    return labels, ReadinessDetector(feats, enter_thresh=enter, exit_thresh=exit_)


def make_reverse_curriculum(labels: list[ReadyLabel], detector: ReadinessDetector,
                            clear_start: list[PlanarSnapshot]) -> "dict[str, list[PlanarSnapshot]]":
    """Reverse curriculum (§3): the non-ready collected states are bucketed by their detector distance to the READY
    basin (near → far = easier → harder approach), and the frozen clear-start corpora form the hardest bands. Training
    walks near→far so the bridge learns short recoveries before long clear-start approaches."""
    non_ready = [(detector.distance_from_features(lab.features), lab.snapshot)
                 for lab in labels if lab.label != "TRANSPORT_READY"]
    non_ready.sort(key=lambda t: t[0])
    ready = [lab.snapshot for lab in labels if lab.label == "TRANSPORT_READY"]
    n = len(non_ready)
    thirds = [non_ready[: n // 3], non_ready[n // 3: 2 * n // 3], non_ready[2 * n // 3:]]
    return {
        "B0_ready": ready,                                          # trivial: already ready → learn hold/handoff
        "B1_near": [s for _d, s in thirds[0]],                      # short recovery
        "B2_mid": [s for _d, s in thirds[1]],
        "B3_far": [s for _d, s in thirds[2]],
        "B4_clear_start": clear_start,                              # frozen clear-start corpus (hardest)
    }


def train_bridge(detector: ReadinessDetector, transport: Any, train_snaps: list[PlanarSnapshot], *,
                 steps: int, seed: int, warm_from: Path = TRANSPORT_CKPT, init_actor: Any = None,
                 log_every: int = 2500) -> Any:
    """Train a BRIDGE_POLICY (SAC) on the bridge-reward env over ``train_snaps`` — reach a verified READY state. Warm-
    started from the transport policy (a sensible approach init) unless ``init_actor`` continues a previous band."""
    from hymeko_rl.experiments.coin_generator_exp import direct_env
    from hymeko_rl.train.sac import SACConfig, train_sac
    inner = direct_env()
    inner._base_override = lambda _i, _t: np.zeros(_ACT, np.float32)
    inner._delta_override = 1.0
    rng = np.random.default_rng(seed)
    benv = BridgeRewardEnv(inner, detector, train_snaps, rng)       # env owns its clear-start reset pool
    if init_actor is not None:
        bridge, critics = init_actor
    else:
        bridge, critics = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=_ACT, action_scale=1.0)
        bridge.load_state_dict(torch.load(warm_from, map_location="cpu"))
    cfg = SACConfig.stable(total_steps=steps, seed=seed, bc_coef=0.0, log_every=log_every, eval_every=max(steps, 1) + 1)
    train_sac(bridge, critics, benv, cfg)
    return bridge, critics


def eval_relay(env: Any, bridge: Any, transport: Any, detector: ReadinessDetector,
               held_snaps: list[PlanarSnapshot], *, hysteresis: int = 3) -> dict[str, Any]:
    """Evaluate the composite relay (bridge → frozen transport) on held states with the CANONICAL strict certificate."""
    strict = loose = ready_entry = handoff = 0
    prog_sum = 0.0
    for snap in held_snaps:
        tr, log = relay_rollout(env, bridge, transport, detector, snap, hysteresis=hysteresis)
        strict += int(bool(policy_strict(tr)))
        loose += int(bool(tr.loose))
        ready_entry += int(log.ready_step >= 0)
        handoff += int(log.handoffs > 0)
        prog_sum += float(tr.progress)
    n = max(1, len(held_snaps))
    return dict(n=len(held_snaps), strict=strict, loose=loose, ready_entry=ready_entry, handoff=handoff,
                coverage=strict, mean_progress=round(prog_sum / n, 4),
                ready_entry_rate=round(ready_entry / n, 3), handoff_rate=round(handoff / n, 3))
