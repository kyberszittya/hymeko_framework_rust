"""Bounded residual over the trot-gait scaffold — learn to reach MULTIPLE goal positions.

The clock-driven ``SteeredTrotGait`` walks the AIBO but reaches only straight, near goals (measured
~1/6 across a distance×bearing grid: it steers poorly off-axis and closes distance slowly). This env
makes ``a = 0`` the pure scaffold (gait + heading pursuit, under the motion contract) and trains a
**bounded residual** (coin-R8 regime: ``final = clip(base + scale·a, ±1)``) to improve goal-reaching
across a *distribution* of goal positions — evaluated on **held-out** positions.

Reward = distance progress + heading alignment + a reach bonus − control cost; a fall terminates. The
residual is bounded so the safe scaffold (never falls, upright ≈ 1.0) is preserved — the coin-R8
prerequisite. Gym-like interface (``reset``/``step``, ``observation_space``/``action_space``) so the
repo SAC (`build_sac`/`train_sac`) drives it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv

from .locomotion_gait import GAIT_PHASES, RotationalTurnGait, SteeredTrotGait, heading_error
from .motion_contract import JointVelocityGovernor

LEGS = ("fl", "fr", "bl", "br")  # leg order (matches _DIAG_PHASE and the hip_abduct_{leg} joints)
_LEFT = (0, 2)                    # fl, bl are the LEFT legs (fr, br are right) — the crab symmetry axis


def minimal_leg_hypergraph(symmetric: bool = False):
    """A 5-vertex leg hypergraph (torso + 4 legs) — the crab-relevant structure only, ~6× cheaper than
    the full 33-vertex body hg. ``symmetric=False``: plain kinematic arcs torso↔leg (down +1 / up −1).
    ``symmetric=True``: the torso↔leg SIGNS encode the LEFT/RIGHT symmetry axis (left legs fl,bl +1;
    right legs fr,br −1), so the signed propagation routes the torso's lateral goal-demand
    DIFFERENTIALLY to the two sides — a symmetric crab the flat MLP cannot represent."""
    from hymeko_rl.agents.hypergraph_state import HypergraphState
    labels = ("torso", "fl", "fr", "bl", "br")
    edges = np.array([(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0)], dtype=np.int64)
    if symmetric:                                        # left legs (1,3) +1, right legs (2,4) −1
        signs = np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.int64)
        tag = "aibo_leg_min_sym_v1"
    else:
        signs = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64)
        tag = "aibo_leg_min_v1"
    return HypergraphState(labels, edges, signs, topo_hash=tag)


@dataclass(frozen=True)
class ResidualTrotConfig:
    """Task distribution + reward shaping for the residual-over-trot goal-reaching problem."""

    dist_lo: float = 0.5
    dist_hi: float = 0.75
    bearing_deg: float = 40.0            # goals sampled in bearing ∈ [−bearing_deg, +bearing_deg]
    residual_mode: str = "leg"           # "leg" = 12-dim raw-target residual | "steer" = 2-dim (Δyaw, Δdrive) gait-param residual | "phase" = 12-dim residual PHASE-GATED per leg | "omni" = 4-dim per-leg ABDUCTION amplitude (phase-locked lateral crab over the forward trot — the RICHER action space, adds lateral DOF the trot leaves unused)
    abd_scale: float = 0.5               # omni mode: bound on the learned per-leg abduction (lateral) amplitude
    obs_mode: str = "flat"               # "flat" = 9-D vector (MLP) | "hypergraph" = (n_vertices, 4) per-vertex on the body's kinematic hypergraph (for signedkan/hsikan structure propagation)
    leg_hg_symmetric: bool = False       # leg_hypergraph mode: encode the LEFT/RIGHT symmetry axis in the hg signs
    gait_phase: str = "diag"             # base-gait phase pattern (GAIT_PHASES): "diag" (trot, asymmetric — default) | "bound" (front/back, instantaneously LEFT-RIGHT SYMMETRIC — the symmetric-scaffold test) | "pace" | "pronk"
    swing_lift: float = 0.0              # >0: swing-gated knee lift = a REAL step (paw clears ~13 cm) instead of the ~2 cm sinusoidal shuffle. 0.40 + gait_freq 1.4 = visible stepping, upright, forward
    gait_freq: float = 1.2               # gait clock frequency; 1.4 pairs with swing_lift for clean stepping
    # --- LOW-DRIFT in-place turn (the rotational couple DRAGS its swing feet → ~17 cm lateral drift / 90°,
    # which spirals the body around wide goals so it enters them BACKWARDS). Lifting the swing feet at the
    # right phase turns nearly IN PLACE (measured ~2 cm/90° at turn_swing_lift 0.35, lift_off 2.9, freq 1.6)
    # → the AIBO can turn to FACE the goal then walk straight in (arrives ~5° for bearings ≤90°). Defaults
    # 0/gait_freq reproduce the prior drifting turn exactly.
    turn_swing_lift: float = 0.0         # >0: lift the turn's swing feet (0.35 ≈ 8× less drift, still upright)
    turn_lift_off: float = 1.5708        # turn swing-lift phase (2.9 minimises drift for the diagonal couple)
    turn_freq: float = 0.0               # turn clock (0 = use gait_freq; 1.6 pairs with the low-drift lift)
    mirror_augment: bool = False         # omni/flat: randomly present the LEFT-RIGHT-MIRRORED task each episode → a symmetry-preserved policy that reaches BOTH crab sides (breaks the symmetry-breaking one-sided optimum)
    residual_scale: float = 0.25         # bounded residual (coin-R8): a small correction over the gait
    yaw_res_scale: float = 0.5           # steer mode: bound on the learned steering correction (rad)
    drive_res_scale: float = 0.5         # steer mode: bound on the learned speed correction
    reach_radius: float = 0.12
    require_facing_deg: float = 0.0      # 0 = position-only reach (back-compat). >0 = success needs |heading err| ≤ this AT reach (must FACE the goal, not drift into it backwards). The rotational-couple turn drifts around wide goals, so position-only "reach" let it enter goals facing ~180° (caught 2026-07-30) — this makes arriving ALIGNED the objective.
    turn_first_deg: float = 0.0          # if |heading error| exceeds this, cut forward drive to TURN IN PLACE toward the goal first (0 = off: walk-and-arc, which never faces wide-bearing goals)
    turn_drive: float = 0.15             # forward drive while turning in place (turn_first_deg > 0)
    heading_mode: str = "arc"            # "arc" (default: skid-steer walk-and-arc — weak turning) | "turn_then_walk" (rotational-couple turn to face the goal, THEN walk — 5x the goal-reach)
    turn_rate: float = 1.0               # rotational-couple turn magnitude (turn_then_walk); ~47 deg/1000 steps, upright at 1.0
    turn_align_deg: float = 20.0         # turn_then_walk: turn until |heading error| within this, then walk
    # --- structured stabilization representation (the DOF the leg/omni/phase modes lack) ---------------
    # The fast rotational turn tips in ROLL (measured: roll diverges −48° while pitch stays small). The
    # physical counters are a LOWER CoM (crouch = symmetric knee flexion) and a WIDER support polygon
    # (widen = mirrored hip abduction) — DOF that exist at the joint level but no prior action space
    # exposed. With a constant crouch+widen the turn stays upright at turn_rate 1.3 and reach jumps
    # 0.14 → 0.86 (probed). These make the STABILIZED-turn scaffold (a=0); the "stab" residual mode then
    # learns a STATE-DEPENDENT modulation (turn faster where stable, brace harder where tipping).
    stab_crouch: float = 0.0             # constant symmetric knee flexion added while turning (lower CoM); ~0.5 stabilizes turn_rate 1.3
    stab_widen: float = 0.0              # constant mirrored hip abduction while turning (wider stance); ~0.4 stabilizes; >0.5 over-widens
    stab_lean: float = 0.0               # constant left/right knee-differential roll bias (lean into the turn); usually 0 at a=0
    rate_res_scale: float = 0.4          # stab mode: bound on the learned turn-rate modulation (fraction of turn_rate)
    crouch_res_scale: float = 0.3        # stab mode: bound on the learned crouch modulation
    widen_res_scale: float = 0.2         # stab mode: bound on the learned stance-width modulation
    lean_res_scale: float = 0.3          # stab mode: bound on the learned roll/lean modulation
    # On the SWING-LIFT scaffold crouch+widen are dead DOF (it is upright without them); the live lever that
    # faces wide bearings is the turn-vs-walk ALIGN threshold (a per-bearing-align oracle lifts held-out reach
    # 0.75→0.93). >0 REPURPOSES the residual's crouch slot (r[1]) to modulate the align threshold instead —
    # the RL sees the bearing (cos/sin herr in obs) and can learn a bearing-conditioned align.
    align_res_scale: float = 0.0         # degrees of learned align modulation (0 = off; r[1]=crouch as usual)
    balance_w: float = 0.0               # reward weight on the foot-support balance entropy H_bal (0=off); dense stay-upright signal for FAST turning (H_bal≈1 weight spread over 4 feet, →0 tipping) — the movement/balance-grounded entropy
    stability_w: float = 0.0             # reward weight on the DYNAMIC stability margin (0=off): a ZMP-family PREDICTIVE signal — capture-point-in-support (translational) + low torso tilt-rate (rotational, the one that fires for spin-tipping); +1 stable, −1 about to tip (fires ~40 steps BEFORE the fall, unlike the reactive H_bal)
    max_steps: int = 800
    v_max: float = 8.0                   # motion-contract joint-speed cap
    progress_w: float = 12.0
    heading_w: float = 0.5
    reach_bonus: float = 8.0
    ctrl_w: float = 0.004
    fall_upright: float = 0.4


class _Box:
    """Minimal Box space (shape + uniform sample) — enough for the repo SAC driver."""

    def __init__(self, dim, low: float = -1.0, high: float = 1.0, seed: int = 0) -> None:
        self.shape = dim if isinstance(dim, tuple) else (dim,)
        self._lo, self._hi = low, high
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.uniform(self._lo, self._hi, size=self.shape).astype(np.float32)


@dataclass
class ResidualTrotEnv:
    """Trot-scaffold + bounded residual, goal-reaching over a distribution of goal positions.

    # Preconditions
    ``a`` (action) is a 12-vector in [−1, 1] (a bounded residual over the gait). # Invariants: the
    scaffold action (``a = 0``) is applied under the motion contract and never overridden by more than
    ``residual_scale``; a fall terminates the episode.
    """

    cfg: ResidualTrotConfig = field(default_factory=ResidualTrotConfig)
    seed: int = 0
    _env: QuadrupedGoalEnv = field(init=False, repr=False)
    _gait: SteeredTrotGait = field(init=False, repr=False)
    _gov: JointVelocityGovernor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=self.cfg.dist_hi,
                                     reach_radius=self.cfg.reach_radius, max_steps=self.cfg.max_steps)
        if self.cfg.gait_phase not in GAIT_PHASES:
            raise ValueError(f"gait_phase must be one of {tuple(GAIT_PHASES)}; got {self.cfg.gait_phase!r}")
        self._phase_pat = GAIT_PHASES[self.cfg.gait_phase]        # per-leg gait phase (diag=asymmetric, bound=symmetric)
        self._gait = SteeredTrotGait(phase=self._phase_pat, swing_lift=self.cfg.swing_lift,
                                     freq=self.cfg.gait_freq)
        self._turn_gait = RotationalTurnGait(                     # rotational-couple turn; low-drift when turn_swing_lift>0
            swing_lift=self.cfg.turn_swing_lift or self.cfg.swing_lift,
            lift_off=self.cfg.turn_lift_off,
            freq=self.cfg.turn_freq or self.cfg.gait_freq)
        self._paw_bodies = [int(mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, f"paw_{lg}"))
                            for lg in LEGS]                      # feet, for the balance (support) entropy
        self._gov = JointVelocityGovernor(v_max=self.cfg.v_max)
        self._rng = np.random.default_rng(self.seed)
        self._prev_dist = 0.0
        self._step_i = 0
        _dims = {"leg": 12, "phase": 12, "steer": 2, "omni": 4, "stab": 4}
        act_dim = _dims[self.cfg.residual_mode]
        self.action_space = _Box(act_dim, seed=self.seed)
        if self.cfg.obs_mode == "hypergraph":
            self._abd_vtx = self._abduction_vertices()
            self.hg = self._env.hg                                # the body's full kinematic hypergraph
            self._n_vtx = int(self._env.hg.n_vertices)
            self.observation_space = _Box((self._n_vtx, 4), low=-5.0, high=5.0)  # (N vertices, feat)
        elif self.cfg.obs_mode == "leg_hypergraph":
            self.hg = minimal_leg_hypergraph(symmetric=self.cfg.leg_hg_symmetric)  # 5-vertex leg-only hg
            self._abd_vtx = [1, 2, 3, 4]                          # the 4 leg vertices (torso is vertex 0)
            self._n_vtx = 5
            self._abd_dof = [int(self._env.model.jnt_dofadr[mujoco.mj_name2id(
                self._env.model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_abduct_{leg}")]) for leg in LEGS]
            self.observation_space = _Box((5, 4), low=-5.0, high=5.0)
        else:
            self.observation_space = _Box(9, low=-5.0, high=5.0)
        self.max_steps = self.cfg.max_steps
        self.model = self._env.model

    # -- goal sampling + observation ------------------------------------------
    def _sample_goal(self) -> None:
        d = float(self._rng.uniform(self.cfg.dist_lo, self.cfg.dist_hi))
        b = float(self._rng.uniform(-self.cfg.bearing_deg, self.cfg.bearing_deg)) * np.pi / 180.0
        tx = float(self._env.data.xpos[self._env.torso, 0])
        ty = float(self._env.data.xpos[self._env.torso, 1])
        self._env.goal = np.array([tx + d * np.cos(b), ty + d * np.sin(b)], np.float32)
        self._env._prev_dist = self._env.dist_to_goal()

    def _phase(self) -> float:
        t = int(getattr(self._env, "_step", 0)) * int(self._env.frame_skip) * float(
            self._env.model.opt.timestep)
        return 2.0 * np.pi * self._gait.freq * t

    def _abduction_vertices(self) -> list[int]:
        """Hypergraph vertices of the 4 hip-abduction actuators (child body b → vertex b-1)."""
        return [int(self._env.model.jnt_bodyid[
            mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_abduct_{leg}")]) - 1
            for leg in LEGS]

    def _obs(self) -> np.ndarray:
        if self.cfg.obs_mode == "hypergraph":
            return self._obs_hypergraph()
        if self.cfg.obs_mode == "leg_hypergraph":
            return self._obs_leg_hypergraph()
        env = self._env
        dist = float(env.dist_to_goal())
        herr = float(heading_error(env))
        vx = float(env.data.cvel[env.torso, 3])
        vy = float(env.data.cvel[env.torso, 4])
        wz = float(env.data.cvel[env.torso, 2])
        ph = self._phase()
        obs = np.array([dist, np.cos(herr), np.sin(herr), vx, vy, wz,
                        np.sin(ph), np.cos(ph), float(env.data.xmat[env.torso].reshape(3, 3)[2, 2])],
                       dtype=np.float32)
        return self.mirror_obs(obs) if getattr(self, "_mirror", False) else obs

    def _obs_hypergraph(self) -> np.ndarray:
        """Per-vertex ``(n_vertices, 4)`` obs on the body's kinematic hypergraph, for structure
        propagation (signedkan): native ``[qpos, qvel]`` + per-leg gait phase + a GLOBAL lateral
        goal-demand. The signed hyperedges route the lateral demand to the per-leg abduction with the
        structure's signs — so a symmetric crab is representable via weight-sharing, unlike a flat MLP.
        """
        env = self._env
        nf = np.asarray(env.node_features(), np.float32)          # (N, 2): torso [dx, fwd_vel]; leg [qpos, qvel]
        out = np.zeros((nf.shape[0], 4), np.float32)
        out[:, :2] = nf
        ph = self._phase()
        for leg, vtx in enumerate(self._abd_vtx):                 # per-leg gait phase on the abduction vertices
            out[vtx, 2] = float(np.sin(ph + self._phase_pat[leg]))
        herr, dist = float(heading_error(env)), float(env.dist_to_goal())
        out[:, 3] = float(np.clip(np.sin(herr) * dist, -1.0, 1.0))  # GLOBAL signed lateral goal-demand
        return out

    def _obs_leg_hypergraph(self) -> np.ndarray:
        """Minimal ``(5, 4)`` per-vertex obs on the leg-only hypergraph: torso vertex carries the goal
        (forward + lateral) + body velocity; each leg vertex carries its abduction state + gait phase +
        the shared lateral demand. The signed hyperedges route the torso's lateral goal to the legs;
        the per-node weight-sharing makes the crab symmetric across the left/right legs."""
        env = self._env
        herr, dist = float(heading_error(env)), float(env.dist_to_goal())
        lat = float(np.clip(np.sin(herr) * dist, -1.0, 1.0))
        fwd = float(np.cos(herr) * dist)
        out = np.zeros((5, 4), np.float32)
        out[0] = [fwd, lat, float(env.data.cvel[env.torso, 3]), float(env.data.cvel[env.torso, 2])]
        ph = self._phase()
        for leg in range(4):
            out[leg + 1] = [float(env.data.qpos[self._env._leg_qadr[3 * leg]]),
                            float(env.data.qvel[self._abd_dof[leg]]),
                            float(np.sin(ph + self._phase_pat[leg])), lat]
        return out

    # -- gym-like API ----------------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._env.reset(seed=seed if seed is not None else int(self._rng.integers(1 << 30)))
        self._sample_goal()
        self._prev_dist = float(self._env.dist_to_goal())
        self._step_i = 0
        self._mirror = bool(self.cfg.mirror_augment and self._rng.random() < 0.5)  # mirrored episode?
        return self._obs(), {}

    @staticmethod
    def mirror_obs(o: np.ndarray) -> np.ndarray:
        """Left-right mirror of the flat obs [dist, cos(herr), sin(herr), vx, vy, wz, sin(ph), cos(ph), up]:
        flip the lateral goal (sin herr), lateral+yaw velocity, AND the gait phase by π (sin/cos ph) — the
        π-shift the diagonal trot needs to be left-right symmetric."""
        m = np.asarray(o, np.float32).copy()
        m[[2, 4, 5, 6, 7]] *= -1.0
        return m

    @staticmethod
    def mirror_act(a: np.ndarray) -> np.ndarray:
        """Left-right mirror of the 4-D omni abduction [fl,fr,bl,br] → [−fr,−fl,−br,−bl] (swap sides + sign)."""
        a = np.asarray(a, np.float64)
        return -a[[1, 0, 3, 2]]

    def blend_action(self, base: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """Bounded residual over the scaffold: ``clip(base + scale·clip(residual, ±1), ±1)``.

        Postcondition: the applied action never departs from ``base`` by more than ``residual_scale``
        per component (before the outer clip) — the safe scaffold is preserved (coin-R8).
        """
        r = np.clip(np.asarray(residual, dtype=np.float64), -1.0, 1.0)
        return np.clip(base + self.cfg.residual_scale * r, -1.0, 1.0)

    def phase_gates(self) -> np.ndarray:
        """Per-leg phase gate ``g_l = ½(1 + sin(ph + DIAG_PHASE_l)) ∈ [0, 1]`` — synced to each leg's
        trot stride, so a gated residual pulses in phase with the gait (preserving the limit cycle)."""
        ph = self._phase()
        return np.array([0.5 * (1.0 + np.sin(ph + self._phase_pat[leg])) for leg in range(4)],
                        dtype=np.float64)

    def foot_support_entropy(self) -> float:
        """Balance entropy H_bal: normalized Shannon entropy of the per-foot ground-support distribution
        (support ∝ how low each paw is). 1.0 = weight spread evenly over all 4 feet (balanced), →0 = weight
        on one foot (tipping). The movement/balance-grounded entropy — cleanly flags tipping (measured
        H≈1.0 upright, →0 when it falls). # Postconditions returns a float in ``[0, 1]``."""
        h = np.array([float(self._env.data.xpos[b, 2]) for b in self._paw_bodies])
        stance = np.maximum(0.0, 0.06 - h)                       # planted weight ~ how far below 6cm the foot is
        total = float(stance.sum())
        if total < 1e-8:
            return 0.0                                           # all feet airborne → no support (degenerate)
        p = stance / total
        return float(-(p * np.log(p + 1e-8)).sum() / np.log(len(self._paw_bodies)))

    def dynamic_stability(self) -> float:
        """A ZMP-family PREDICTIVE stability margin in ``[-1, 1]`` (+1 stable, −1 about to tip), combining:
        (a) **capture-point in support** (translational): the CoM's capture point ``CoM_xy + v·√(z/g)`` vs the
        stance-weighted support region; (b) **low torso tilt-rate** (rotational): the roll+pitch angular
        speed — the signal that actually fires for the AIBO's spin-tipping, ~40 steps BEFORE the fall (the
        reactive foot-support H_bal only flags it after). Both are physically grounded and anticipatory.
        # Postconditions returns a float in ``[-1, 1]``."""
        r = self._env
        com = np.asarray(r.data.subtree_com[0])
        g = float(-r.model.opt.gravity[2]) or 9.81
        vel = np.asarray(r.data.subtree_linvel[0])
        cp = com[:2] + vel[:2] * float(np.sqrt(max(com[2], 1e-3) / g))
        feet = np.array([r.data.xpos[b][:2] for b in self._paw_bodies])
        w = np.maximum(0.0, 0.06 - np.array([r.data.xpos[b][2] for b in self._paw_bodies]))
        if w.sum() < 1e-6:
            cp_margin = -1.0
        else:
            c = (feet * w[:, None]).sum(0) / w.sum()
            rad = float(np.sqrt(((feet - c) ** 2).sum(1) * w).sum() / w.sum())
            cp_margin = float(np.tanh(3.0 * (rad - float(np.linalg.norm(cp - c)))))
        tilt_rate = float(np.linalg.norm(np.asarray(r.data.cvel[r.torso])[0:2]))   # roll+pitch angular speed
        tilt_margin = float(np.tanh(1.5 * (1.5 - tilt_rate)))                       # +1 low rate (stable), −1 high
        return 0.5 * (cp_margin + tilt_margin)                                      # both must hold to be stable

    #: per-leg (fl, fr, bl, br) left/right mirror sign — widen/lean map to opposite abduction/knee per side
    _SIDE_SIGN = np.array([+1.0, -1.0, +1.0, -1.0])

    def _stab_offset(self, crouch: float, widen: float, lean: float) -> np.ndarray:
        """Structured stabilization offset (12-dim, [abduct, flex, knee] per leg) for the fast turn.

        The measured tip is a ROLL divergence; the physical counters are a lower CoM and a wider base:
        - ``crouch`` → symmetric knee flexion (all legs) — lowers the CoM, the dominant stabilizer;
        - ``widen``  → mirrored hip abduction (left/right opposite) — widens the support polygon;
        - ``lean``   → left/right knee-differential — a roll bias to lean into the turn.

        This is the low-dim, physically-meaningful action representation the ``leg``/``omni``/``phase``
        modes lack (raw targets, sinusoidal-only abduction). # Postconditions: 12-vector, before the
        outer ``clip(base + off, ±1)``; zero offset when crouch = widen = lean = 0 (no scaffold change)."""
        off = np.zeros(12)
        for leg in range(4):
            off[3 * leg + 2] += crouch + lean * self._SIDE_SIGN[leg]   # knee: crouch (sym) + roll bias (diff)
            off[3 * leg + 0] += widen * self._SIDE_SIGN[leg]           # abduct: widen stance (mirror per side)
        return off

    def _base_gait_action(self, herr: float, pursuit: float, base_drive: float, *,
                          crouch: float | None = None, widen: float | None = None,
                          lean: float | None = None, rate_mult: float = 1.0,
                          align_deg: float | None = None) -> np.ndarray:
        """The scaffold's base action before the residual: the rotational-couple TURN toward the goal when
        ``heading_mode="turn_then_walk"`` and the heading error is still wide, else the forward trot. This
        is the goal-reaching turning fix (0.11 → 0.56 reach); ``"arc"`` (default) keeps the prior weak
        skid-steer walk-and-arc.

        The turn carries the structured stabilization offset (``crouch``/``widen``/``lean``, defaulting to
        the ``stab_*`` config) so ``a = 0`` is the STABILIZED fast turn; ``rate_mult`` scales the turn rate
        (the ``stab`` residual modulates both). Defaults reproduce the prior behaviour exactly (cfg stab_*
        default 0, rate_mult 1). # Postconditions returns a governed ``(n_actions,)`` action in ``[-1, 1]``."""
        env = self._env
        crouch = self.cfg.stab_crouch if crouch is None else crouch
        widen = self.cfg.stab_widen if widen is None else widen
        lean = self.cfg.stab_lean if lean is None else lean
        align = self.cfg.turn_align_deg if align_deg is None else align_deg   # RL-modulated turn-vs-walk threshold
        if self.cfg.heading_mode == "turn_then_walk" and abs(herr) > float(np.deg2rad(align)):
            turn = float(np.sign(herr)) * self.cfg.turn_rate * rate_mult
            raw = self._turn_gait.action(env, turn=turn)
            if crouch or widen or lean:                               # stabilize the fast turn (roll counter)
                raw = np.clip(raw + self._stab_offset(crouch, widen, lean), -1.0, 1.0)
            return self._gov.govern(env, raw)
        raw = self._gait.action(env, yaw_cmd=pursuit, drive=base_drive)
        if crouch or widen or lean:                                  # keep the brace through the walk-in too
            raw = np.clip(raw + self._stab_offset(crouch, widen, lean), -1.0, 1.0)
        return self._gov.govern(env, raw)

    def _base_drive(self, herr: float, dist: float) -> float:
        """Forward-drive command: 0 within the reach radius; else 1.0, capped to ``turn_drive`` while the
        heading error is wide (``turn_first_deg`` > 0) so the robot turns toward a wide-bearing goal before
        arcing past it. # Preconditions ``dist >= 0``. # Postconditions returns a drive in ``[0, 1]``."""
        if dist <= self.cfg.reach_radius:
            return 0.0
        if self.cfg.turn_first_deg > 0.0 and abs(herr) > float(np.deg2rad(self.cfg.turn_first_deg)):
            return float(self.cfg.turn_drive)
        return 1.0

    def _apply(self, residual: np.ndarray) -> None:
        """Compose the residual with the gait per mode and step the underlying env.

        - ``leg``  : a raw 12-dim residual on the gait's leg targets (``blend_action``) — breaks phase.
        - ``steer``: a 2-dim ``(Δyaw, Δdrive)`` residual on the gait's STEERING + SPEED *parameters*.
        - ``phase``: a 12-dim residual GATED per leg by its trot phase (``phase_gates``) — it only acts
          on each leg in sync with that leg's stride, so it can bias differential stance thrust to
          steer while leaving the periodic limit cycle intact.
        - ``stab`` : a 4-dim STRUCTURED residual ``(Δrate, Δcrouch, Δwiden, Δlean)`` that modulates the
          fast turn's rate + the physical roll-stabilization DOF, state-dependently, around the
          ``stab_*`` scaffold. The action representation the other modes lack (see ``_stab_offset``).
        """
        env = self._env
        dist = float(env.dist_to_goal())
        herr = float(heading_error(env))
        pursuit = float(np.clip(1.1 * herr, -0.6, 0.6))
        base_drive = self._base_drive(herr, dist)
        r = np.clip(np.asarray(residual, dtype=np.float64), -1.0, 1.0)
        if self.cfg.residual_mode == "stab":
            rate_mult = 1.0 + self.cfg.rate_res_scale * float(r[0])           # turn faster where stable / ease off where tipping
            lean = self.cfg.stab_lean + self.cfg.lean_res_scale * float(r[3])
            if self.cfg.align_res_scale > 0.0:                               # swing-lift scaffold: r[1] is the ALIGN lever
                align = float(np.clip(self.cfg.turn_align_deg + self.cfg.align_res_scale * float(r[1]), 5.0, 30.0))
                final = self._base_gait_action(herr, pursuit, base_drive, crouch=0.0, widen=0.0,
                                               lean=lean, rate_mult=rate_mult, align_deg=align)
            else:                                                            # fast-turn scaffold: r[1]/r[2] are crouch/widen
                crouch = self.cfg.stab_crouch + self.cfg.crouch_res_scale * float(r[1])
                widen = self.cfg.stab_widen + self.cfg.widen_res_scale * float(r[2])
                final = self._base_gait_action(herr, pursuit, base_drive, crouch=crouch, widen=widen,
                                               lean=lean, rate_mult=rate_mult)   # already governed
        elif self.cfg.residual_mode == "steer":
            yaw = float(np.clip(pursuit + self.cfg.yaw_res_scale * r[0], -0.6, 0.6))
            drive = float(np.clip(base_drive + self.cfg.drive_res_scale * r[1], 0.0, 1.5))
            final = self._gov.govern(env, self._gait.action(env, yaw_cmd=yaw, drive=drive))
        elif self.cfg.residual_mode == "omni":
            base = self._base_gait_action(herr, pursuit, base_drive)
            ph = self._phase()
            final = base.copy()
            for leg in range(4):                          # per-leg abduction, phase-locked -> lateral crab
                idx = 3 * leg                             # abduction is action index 3*leg+0
                lateral = self.cfg.abd_scale * r[leg] * np.sin(ph + self._phase_pat[leg])
                final[idx] = float(np.clip(final[idx] + lateral, -1.0, 1.0))
        else:
            base = self._base_gait_action(herr, pursuit, base_drive)
            if self.cfg.residual_mode == "phase":
                r = (r.reshape(4, 3) * self.phase_gates()[:, None]).reshape(-1)   # gate per leg by stride phase
            final = self.blend_action(base, r)
        env.step(final)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        residual = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if getattr(self, "_mirror", False):
            residual = self.mirror_act(residual)          # policy acts in the mirrored view; un-mirror to apply
        self._apply(residual)
        self._step_i += 1
        dist = float(self._env.dist_to_goal())
        upright = float(self._env.data.xmat[self._env.torso].reshape(3, 3)[2, 2])
        herr = float(heading_error(self._env))
        facing = self.cfg.require_facing_deg <= 0.0 or abs(herr) <= float(np.deg2rad(self.cfg.require_facing_deg))
        reached = dist <= self.cfg.reach_radius and facing        # success needs to arrive FACING the goal, not drift in backwards
        fell = upright < self.cfg.fall_upright
        progress = self._prev_dist - dist
        self._prev_dist = dist
        reward = (self.cfg.progress_w * progress
                  + self.cfg.heading_w * float(np.cos(herr))
                  - self.cfg.ctrl_w * float(np.sum(residual ** 2)))
        if self.cfg.balance_w > 0.0:
            reward += self.cfg.balance_w * self.foot_support_entropy()   # dense stay-upright signal (fast-turn)
        if self.cfg.stability_w > 0.0:
            reward += self.cfg.stability_w * self.dynamic_stability()     # PREDICTIVE ZMP-family stability (anticipatory)
        if reached:
            reward += self.cfg.reach_bonus
        if fell:
            reward -= 5.0
        term = reached or fell
        trunc = self._step_i >= self.cfg.max_steps
        info = {"dist": dist, "reached": reached, "upright": upright, "fell": fell}
        return self._obs(), float(reward), bool(term), bool(trunc), info

    # -- evaluation helpers ----------------------------------------------------
    def rollout_min_dist(self, act_fn, goal: tuple[float, float], seed: int,
                         horizon: int | None = None) -> tuple[float, bool, float]:
        """Roll ``act_fn`` toward a FIXED (dist, bearing_deg) goal; return (min_dist, reached, min_upright)."""
        d, bdeg = goal
        self._mirror = False                              # eval is always the un-mirrored real task
        self._env.reset(seed=seed)
        b = bdeg * np.pi / 180.0
        tx = float(self._env.data.xpos[self._env.torso, 0])
        ty = float(self._env.data.xpos[self._env.torso, 1])
        self._env.goal = np.array([tx + d * np.cos(b), ty + d * np.sin(b)], np.float32)
        self._env._prev_dist = self._env.dist_to_goal()
        self._prev_dist = float(self._env.dist_to_goal())
        self._step_i = 0
        min_dist, min_up = self._prev_dist, 1.0
        min_dist_facing = self._prev_dist                 # closest approach WHILE facing the goal (the real metric)
        face_tol = float(np.deg2rad(self.cfg.require_facing_deg)) if self.cfg.require_facing_deg > 0.0 else None
        reached = False
        for _ in range(horizon or self.cfg.max_steps):
            self._apply(act_fn(self._obs()))
            d_now = float(self._env.dist_to_goal())
            min_dist = min(min_dist, d_now)
            min_up = min(min_up, float(self._env.data.xmat[self._env.torso].reshape(3, 3)[2, 2]))
            facing = face_tol is None or abs(float(heading_error(self._env))) <= face_tol
            if facing:
                min_dist_facing = min(min_dist_facing, d_now)
            if d_now <= self.cfg.reach_radius and facing:      # only an ALIGNED reach counts / stops the rollout
                reached = True
                break
        report = min_dist_facing if face_tol is not None else min_dist
        return round(report, 4), reached, round(min_up, 3)
