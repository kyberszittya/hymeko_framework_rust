"""CIP counterfactual data augmentation (Cao et al., ICLR 2025; foregrounded by Ito, Sakuma, Gu, Kato, katolab).

This is the **in-the-loop** CIP mechanism — distinct from the repo's offline CIP *diagnostic* layer
(:mod:`hymeko_rl.eval.causal`, :mod:`hymeko_rl.eval.cip.metaworld_cip`), which *proposes* structure and lets
ablations decide. Here the causal weights are put to work *during* SAC training as Cao's **CDS** (counterfactual
data source) augmentation:

    every ``refresh_every`` env steps
      1. sample a batch of replay transitions ``(s, a, r, s')``
      2. fit :class:`~hymeko_rl.eval.causal.lingam.DirectLiNGAM` over ``X = [s | a | r]`` and read the *reward
         row* of the adjacency → state-to-reward weights ``w_s`` and action-to-reward weights ``w_r``
      3. softmax-normalise + rescale by the number of dimensions (Cao/Ito)
      4. identify the lowest-causal-importance ("uncontrollable") state dimension(s) and **swap** them between
         paired transitions (in both ``s`` and ``s'``, keeping ``a, r`` of the first) → synthetic transitions
      5. append the synthetic transitions to the replay buffer

It plugs into :func:`hymeko_rl.train.sac.train_sac` through the generic ``ReplayAugmentor`` seat (default off →
byte-identical). It is the *baseline* for the CIP-continuation arc; the LLM correction of ``w_r, w_s`` (Ito's
contribution) and the HyMeKo structural corrector are separate, later directions.

**Fidelity note.** Cao's full CIP additionally carries an *empowerment* reward term (reweighting actions by
``w_r`` inside a mutual-information objective). Ito's description foregrounds the DirectLiNGAM + CDS core, which
is what this module reproduces (``cip-cds``). The empowerment term is a documented deferred component, not a
silent omission.

Paradigm: a config dataclass + pure numerical helpers + one Strategy class (:class:`CipReplayAugmentor`) that
holds no global state (§6.5 #11) and threads its RNG/config explicitly. The algorithm (LiNGAM, the swap) lives
here in ``eval/``, not behind a binding boundary (§6.5 #2).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from hymeko_rl.eval.causal.lingam import DirectLiNGAM, LingamConfig, LingamResult
from hymeko_rl.train.replay import ReplayBuffer

_DEGEN_EPS = 1e-9


def softmax_rescale(importance: np.ndarray) -> np.ndarray:
    """Softmax of ``importance`` rescaled by the number of dimensions (the Cao/Ito normalisation).

    # Preconditions ``importance`` is a finite 1-D array of length ``d >= 1``.
    # Postconditions returns a same-shape array of non-negative weights summing to ``d`` (uniform → all ones).
    """
    imp = np.asarray(importance, dtype=np.float64).ravel()
    if imp.size == 0:
        return imp
    e = np.exp(imp - imp.max())          # shift for numerical stability
    p = e / e.sum()
    rescaled: np.ndarray = p * imp.size
    return rescaled


def _abs_corr(states: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """Per-column ``|corr(state_i, reward)|`` (0 for a constant column) — the degenerate-fit importance fallback.

    # Postconditions returns a length-``d_s`` non-negative array; constant columns map to 0.
    """
    r = reward - reward.mean()
    rn = float(np.sqrt(r @ r))
    out = np.zeros(states.shape[1], dtype=np.float64)
    if rn <= _DEGEN_EPS:
        return out
    for j in range(states.shape[1]):
        c = states[:, j] - states[:, j].mean()
        cn = float(np.sqrt(c @ c))
        out[j] = abs(float(c @ r) / (cn * rn)) if cn > _DEGEN_EPS else 0.0
    return out


@dataclass(frozen=True)
class CipWeights:
    """The CIP causal weights read from one DirectLiNGAM fit. **Proposed structure, not proof** (framework doctrine)."""

    w_s: np.ndarray                 # state→reward coefficients (reward row of B), length d_s
    w_r: np.ndarray                 # action→reward coefficients, length d_a
    importance_s: np.ndarray        # softmax-rescaled state importance (|w_s| or |corr| fallback), length d_s
    lowest_dims: list[int]          # least-important ("uncontrollable") state dims — the swap targets
    degenerate: bool                # True ⇒ reward ordered without parents; fell back to |corr| importance
    result: LingamResult | None     # the full fit (None when degenerate) — for a .hymeko cross-view snapshot


def estimate_cip_weights(obs: np.ndarray, act: np.ndarray, rew: np.ndarray, *,
                         lingam: DirectLiNGAM, n_swap_dims: int) -> CipWeights:
    """Fit DirectLiNGAM over ``[s | a | r]`` and extract the reward's causal parents (Cao/Ito ``w_s, w_r``).

    # Preconditions ``obs`` ``(n, d_s)``, ``act`` ``(n, d_a)``, ``rew`` ``(n,)`` with ``n > d_s+d_a+1`` after
      dropping non-finite rows; ``1 <= n_swap_dims <= d_s``.
    # Postconditions returns :class:`CipWeights` whose ``lowest_dims`` are the ``n_swap_dims`` least-important
      state dims (ties broken by index). When the reward row is ~0 (reward ordered as a root) the importance
      falls back to ``|corr(state, reward)|`` and ``degenerate=True``.
    # Errors ``ValueError`` propagated from :meth:`DirectLiNGAM.fit` (too few samples / non-finite / <2 varying
      columns) — callers that must not abort a long run should catch it and skip the refresh.
    """
    d_s, d_a = int(obs.shape[1]), int(act.shape[1])
    x = np.concatenate([obs, act, np.asarray(rew, dtype=np.float64).reshape(-1, 1)], axis=1)
    finite = np.all(np.isfinite(x), axis=1)
    x = x[finite]
    names = [f"s{i}" for i in range(d_s)] + [f"a{i}" for i in range(d_a)] + ["reward"]
    reward_idx = d_s + d_a
    result = lingam.fit(x, names)                       # may raise ValueError — caller decides
    row = result.adjacency[reward_idx]                 # B[reward, :] = each variable's direct effect on reward
    w_s = np.asarray(row[:d_s], dtype=np.float64)
    w_r = np.asarray(row[d_s:d_s + d_a], dtype=np.float64)
    importance = np.abs(w_s)
    degenerate = float(importance.sum()) < _DEGEN_EPS
    if degenerate:                                     # reward had no parents in the recovered order → correlate
        importance = _abs_corr(x[:, :d_s], x[:, reward_idx])
    resc = softmax_rescale(importance)
    k = max(1, min(int(n_swap_dims), d_s))
    lowest = [int(i) for i in np.argsort(importance, kind="stable")[:k]]
    return CipWeights(w_s=w_s, w_r=w_r, importance_s=resc, lowest_dims=lowest,
                      degenerate=degenerate, result=None if degenerate else result)


def counterfactual_swap(obs: np.ndarray, next_obs: np.ndarray, act: np.ndarray, rew: np.ndarray,
                        done: np.ndarray, dims: list[int], rng: np.random.Generator,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cao CDS swap: replace the uncontrollable ``dims`` of each transition with a random partner's, in ``s`` and
    ``s'`` alike, keeping the original ``a, r, done``.

    Swapping a *causally-irrelevant* state dimension leaves the transition's reward valid while broadening the
    state coverage the critic/actor see — the counterfactual data source.

    # Preconditions all arrays share leading dim ``n``; ``obs``/``next_obs`` are ``(n, d_s)``; ``dims`` ⊂ ``[0,d_s)``.
    # Postconditions returns fresh arrays ``(s', a, r, next', done)`` of the same shapes; only the columns in
      ``dims`` of ``s'`` and ``next'`` differ from the inputs (permuted across the batch); ``a, r, done`` copied.
    """
    n = int(obs.shape[0])
    perm = rng.permutation(n)
    s_obs = obs.copy()
    s_next = next_obs.copy()
    for d in dims:
        s_obs[:, d] = obs[perm, d]
        s_next[:, d] = next_obs[perm, d]
    return s_obs, act.copy(), np.asarray(rew, dtype=np.float32).copy(), s_next, np.asarray(done).copy()


@dataclass
class CipAugmentConfig:
    """Configuration for :class:`CipReplayAugmentor` (Ito defaults: 10k-step cadence, single lowest dim swapped)."""

    refresh_every: int = 10_000     # env-step cadence of the LiNGAM fit + augmentation (Ito: 10,000)
    sample_n: int = 1500            # transitions drawn per refresh for the LiNGAM fit + swap (n > d ⇒ >~45)
    n_swap_dims: int = 1            # how many of the least-important state dims to swap (Ito: "the" lowest = 1)
    min_buffer: int = 2000          # do nothing until the buffer holds this many transitions
    seed: int = 0                   # augmentor RNG seed (explicit, no global — §6.5 #11)
    log: bool = True                # emit a flushed [cip] line per refresh (§3 live observability)
    lingam: LingamConfig = field(default_factory=LingamConfig)


class CdsReplayAugmentor:
    """Cao/Ito **CDS** (counterfactual data source) augmentation as a ``ReplayAugmentor`` over the SAC replay buffer.

    **This is CDS ONLY — not "full CIP".** Full CIP additionally carries the empowerment term (reverse/source
    policy + causal-weighted intrinsic reward), which is *not implemented here* (see the module fidelity note).
    The experiment arm that attaches this augmentor is therefore `sac_cds` / `sac_cda`, NOT `cip_full`.

    Implements the :class:`hymeko_rl.train.sac.ReplayAugmentor` protocol. Holds its own RNG + DirectLiNGAM
    (immutable config); no global state. Exposes ``last`` (the most recent :class:`CipWeights`), ``last_fit_ms``,
    ``n_refresh`` and ``n_augmented`` for logging, the report, and an optional ``.hymeko`` cross-view snapshot.

    # Preconditions the training env emits **flat** observations of width ``state_dim`` (the CIP setting;
      MetaWorld's 39-d obs). A 2-D hypergraph obs is rejected at :meth:`maybe_augment`.
    """

    def __init__(self, state_dim: int, action_dim: int, cfg: CipAugmentConfig | None = None) -> None:
        if state_dim < 2 or action_dim < 1:
            raise ValueError(f"state_dim>=2 and action_dim>=1 required; got {state_dim}/{action_dim}")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.cfg = cfg or CipAugmentConfig()
        self._lingam = DirectLiNGAM(self.cfg.lingam)
        self._rng = np.random.default_rng(self.cfg.seed)
        self.n_refresh = 0
        self.n_augmented = 0
        self.last: CipWeights | None = None
        self.last_fit_ms = 0.0

    def maybe_augment(self, buf: ReplayBuffer, step: int) -> None:
        """Fit LiNGAM + augment iff ``step`` is on the refresh cadence and the buffer is warm enough.

        A LiNGAM fit failure (e.g. a transient non-finite / degenerate batch) skips *this* refresh and logs —
        it never aborts training (§6.4: the specific ``ValueError`` is caught, with context)."""
        cfg = self.cfg
        if cfg.refresh_every <= 0 or step % cfg.refresh_every != 0:
            return
        if buf.size < max(cfg.min_buffer, self.state_dim + self.action_dim + 2):
            return
        n = min(cfg.sample_n, buf.size)
        obs, act, rew, next_obs, done = (t.numpy() for t in buf.sample(n, generator=self._rng))
        if obs.ndim != 2 or obs.shape[1] != self.state_dim:
            raise ValueError(f"CipReplayAugmentor expects flat obs (n, {self.state_dim}); got {obs.shape}")
        t0 = time.perf_counter()
        try:
            weights = estimate_cip_weights(obs, act, rew, lingam=self._lingam, n_swap_dims=cfg.n_swap_dims)
        except ValueError as err:                       # transient bad batch — skip this refresh, keep training
            if cfg.log:
                print(f"  [cip] step {step}: LiNGAM fit skipped ({err})", flush=True)
            return
        self.last_fit_ms = (time.perf_counter() - t0) * 1e3
        s_obs, s_act, s_rew, s_next, s_done = counterfactual_swap(
            obs, next_obs, act, rew, done, weights.lowest_dims, self._rng)
        buf.add_batch(s_obs, s_act, s_rew, s_next, s_done)
        self.n_refresh += 1
        self.n_augmented += int(s_obs.shape[0])
        self.last = weights
        if cfg.log:
            imp_max = float(np.max(np.abs(weights.w_s))) if weights.w_s.size else 0.0
            print(f"  [cip] step {step:>7}: refresh #{self.n_refresh} | swap dims={weights.lowest_dims} "
                  f"| |w_s|max={imp_max:.3g} |w_r|max={float(np.max(np.abs(weights.w_r))):.3g} "
                  f"degenerate={weights.degenerate} | +{s_obs.shape[0]} aug (tot {self.n_augmented}) "
                  f"| fit={self.last_fit_ms:.0f}ms", flush=True)

    def summary(self) -> dict[str, object]:
        """Provenance dict for the run summary/report (weights + counters + last fit cost)."""
        w = self.last
        return {
            "n_refresh": self.n_refresh,
            "n_augmented": self.n_augmented,
            "last_fit_ms": round(self.last_fit_ms, 1),
            "refresh_every": self.cfg.refresh_every,
            "sample_n": self.cfg.sample_n,
            "n_swap_dims": self.cfg.n_swap_dims,
            "last_lowest_dims": list(w.lowest_dims) if w else None,
            "last_degenerate": bool(w.degenerate) if w else None,
            "last_w_s": [round(float(v), 4) for v in w.w_s] if w else None,
            "last_w_r": [round(float(v), 4) for v in w.w_r] if w else None,
        }


# Deprecated alias: the augmentor is CDS-only. The old name conflated it with "full CIP" (which additionally needs
# the empowerment term). Kept so pre-existing imports/checkpoints keep working; new code should use CdsReplayAugmentor.
CipReplayAugmentor = CdsReplayAugmentor
