"""Synthetic chicken-flock simulator.

Multi-bird agent-based simulation that produces realistic
trajectories + ground-truth aggressor labels.  Used while waiting
for real video / pose CSV from Éva.  The output format (``Trajectories``)
is the same one ``interactions.trajectories_to_signed_graph`` consumes
when fed real DeepLabCut / SLEAP pose tracks.

Behaviour rules (tunable):
  * **Aggressor birds** (a small fraction $f_{\\rm aggr}$ of the flock)
    actively approach the nearest non-aggressor when within
    ``proximity_radius`` and trigger a peck event with probability
    $p_{\\rm peck}$ per timestep when within ``peck_radius``.
  * **Victim birds** flee (heading away from the nearest aggressor) at
    elevated speed for ``flee_steps`` after a peck.
  * **Neutral birds** random-walk with mild attraction to the flock
    centroid.

Output:
  trajectories : (T, N, 3)  per-frame (x, y, heading_rad) per bird
  events       : list[InteractionEvent]  (frame, src, dst, type)
                  where type ∈ {"peck", "proximity"}
  aggressor    : (N,) boolean — ground-truth aggressor identity

Tunable parameters live on ``ChickenFlockSim`` and default to values
that produce ~5–15 peck events over 300 frames at 20 birds.  Every
``simulate_flock(seed=k)`` is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChickenFlockSim:
    n_birds: int = 20
    n_frames: int = 300
    pen_size: float = 4.0           # metres
    base_speed: float = 0.04         # m / frame  ≈ 1 m/s @ 25 fps
    flee_speed_mult: float = 3.0
    flee_steps: int = 8
    proximity_radius: float = 0.4    # m — counted as a "peaceful" edge
    peck_radius: float = 0.15        # m — peck can fire here
    peck_prob: float = 0.18          # per-step, conditional on radius
    n_aggressors: int = 3
    centroid_pull: float = 0.005
    heading_noise: float = 0.4       # rad / frame
    seed: int = 0


@dataclass
class _BirdState:
    pos: np.ndarray
    heading: float
    flee_remaining: int = 0
    aggressor: bool = False
    last_peck_target: int = -1


@dataclass
class InteractionEvent:
    frame: int
    src: int        # actor (peck initiator or proximity contributor)
    dst: int        # target / partner
    type: str       # "peck" or "proximity"


@dataclass
class _SimResult:
    trajectories: np.ndarray  # (T, N, 3)
    events: list[InteractionEvent]
    aggressor: np.ndarray     # (N,) bool


def _wrap_into_pen(pos: np.ndarray, pen: float) -> np.ndarray:
    return np.clip(pos, -pen, pen)


def simulate_flock(cfg: ChickenFlockSim | None = None) -> _SimResult:
    """Run one simulation.  Returns trajectories + events +
    ground-truth aggressor labels."""
    cfg = cfg or ChickenFlockSim()
    rng = np.random.default_rng(cfg.seed)

    aggressor = np.zeros(cfg.n_birds, dtype=bool)
    aggr_idx = rng.choice(cfg.n_birds, size=cfg.n_aggressors,
                            replace=False)
    aggressor[aggr_idx] = True

    birds: list[_BirdState] = []
    for i in range(cfg.n_birds):
        birds.append(_BirdState(
            pos=rng.uniform(-cfg.pen_size, cfg.pen_size, size=2),
            heading=rng.uniform(-np.pi, np.pi),
            aggressor=bool(aggressor[i]),
        ))

    trajectories = np.zeros((cfg.n_frames, cfg.n_birds, 3),
                              dtype=np.float32)
    events: list[InteractionEvent] = []

    for t in range(cfg.n_frames):
        # Centroid for mild flocking.
        centroid = np.mean([b.pos for b in birds], axis=0)
        # Distance matrix for nearest-neighbour queries.
        positions = np.stack([b.pos for b in birds])      # (N, 2)
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))               # (N, N)
        np.fill_diagonal(dist, np.inf)

        # Step every bird.
        for i, b in enumerate(birds):
            # Heading update —
            if b.flee_remaining > 0:
                # already in flee mode; keep heading + extra noise
                b.heading += rng.normal(scale=cfg.heading_noise * 0.5)
                speed = cfg.base_speed * cfg.flee_speed_mult
                b.flee_remaining -= 1
            elif b.aggressor:
                # head toward nearest non-aggressor within proximity
                non_aggr_mask = ~aggressor.copy()
                non_aggr_mask[i] = False
                if non_aggr_mask.any():
                    masked = np.where(non_aggr_mask, dist[i], np.inf)
                    j = int(np.argmin(masked))
                    if dist[i, j] < cfg.proximity_radius:
                        b.last_peck_target = j
                        delta = positions[j] - b.pos
                        b.heading = float(np.arctan2(delta[1], delta[0]))
                    else:
                        b.heading += rng.normal(scale=cfg.heading_noise)
                else:
                    b.heading += rng.normal(scale=cfg.heading_noise)
                speed = cfg.base_speed
            else:
                # neutral: mild centroid pull + random walk
                pull = (centroid - b.pos)
                if np.linalg.norm(pull) > 1e-6:
                    pull /= np.linalg.norm(pull)
                    pull_h = float(np.arctan2(pull[1], pull[0]))
                    # blend: 1% centroid pull
                    b.heading = (b.heading +
                                 cfg.centroid_pull * (pull_h - b.heading))
                b.heading += rng.normal(scale=cfg.heading_noise)
                speed = cfg.base_speed

            b.pos = _wrap_into_pen(
                b.pos + speed * np.array([np.cos(b.heading),
                                            np.sin(b.heading)],
                                           dtype=np.float64),
                cfg.pen_size,
            )
            trajectories[t, i] = (b.pos[0], b.pos[1], b.heading)

        # Detect events at this frame.  Recompute distances after move.
        positions = np.stack([b.pos for b in birds])
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)

        # Peck events: aggressor within peck_radius of non-aggressor.
        for i in np.where(aggressor)[0]:
            for j in np.where(~aggressor)[0]:
                if dist[i, j] < cfg.peck_radius:
                    if rng.random() < cfg.peck_prob:
                        events.append(InteractionEvent(
                            frame=t, src=int(i), dst=int(j), type="peck"))
                        # victim flees
                        birds[j].flee_remaining = cfg.flee_steps
                        # aim flee away
                        delta = positions[j] - positions[i]
                        if np.linalg.norm(delta) > 1e-6:
                            birds[j].heading = float(
                                np.arctan2(delta[1], delta[0]))

        # Proximity events: any pair within proximity_radius and NOT a
        # peck (already logged).  We log a single proximity edge per
        # close pair per frame.  Subsampled by 1/8 to keep the graph
        # density reasonable.
        if t % 8 == 0:
            iu, jv = np.where(np.triu(dist < cfg.proximity_radius, k=1))
            for ii, jj in zip(iu.tolist(), jv.tolist()):
                events.append(InteractionEvent(
                    frame=t, src=int(ii), dst=int(jj), type="proximity"))

    return _SimResult(
        trajectories=trajectories,
        events=events,
        aggressor=aggressor,
    )


# ─── CLI smoke test ──────────────────────────────────────────────────


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-birds", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--n-aggressors", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=
                    "signedkan_wip/data/chicken_synth/sim.npz")
    args = ap.parse_args()

    cfg = ChickenFlockSim(
        n_birds=args.n_birds,
        n_frames=args.n_frames,
        n_aggressors=args.n_aggressors,
        seed=args.seed,
    )
    res = simulate_flock(cfg)

    n_peck = sum(1 for e in res.events if e.type == "peck")
    n_prox = sum(1 for e in res.events if e.type == "proximity")
    print(f"  trajectories: {res.trajectories.shape}")
    print(f"  events: {len(res.events)} total — "
          f"{n_peck} peck, {n_prox} proximity")
    print(f"  aggressors:   {res.aggressor.sum()}/{cfg.n_birds} "
          f"= {res.aggressor.tolist()}")

    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ev_arr = np.array(
        [(e.frame, e.src, e.dst, e.type) for e in res.events],
        dtype=[("frame", "i4"), ("src", "i4"),
               ("dst", "i4"), ("type", "U10")],
    )
    np.savez_compressed(
        out,
        trajectories=res.trajectories,
        events=ev_arr,
        aggressor=res.aggressor,
    )
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
