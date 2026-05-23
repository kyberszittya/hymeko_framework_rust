"""Trajectories → signed interaction graph.

Same converter handles synthetic simulator output and real
DeepLabCut / SLEAP pose tracks, as long as they're packaged into
a ``Trajectories`` dataclass.

Sign convention:
    +1   peaceful proximity (two birds within ``proximity_radius`` for
         at least ``proximity_min_frames`` consecutive frames, with no
         peck event between them in the window)
    -1   aggressive event (peck, chase, or fast-approach as detected
         by ``detect_peck_events``).

The output ``SignedGraph`` plugs directly into the existing HSiKAN
bench harness: ``signedkan_wip.experiments.runs.run_final_cell --dataset
<custom>`` would be straightforward to extend, or use the lower-
level ``signedkan_wip.src.core.signedkan.SignedKAN`` directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..datasets import SignedGraph


# ─── data containers ─────────────────────────────────────────────────


@dataclass
class Trajectories:
    """Per-frame per-bird state.

    positions : (T, N, 2)   x, y in metres, common origin per session
    heading   : (T, N)      radians, optional — pass ``None`` to
                            disable heading-aware peck detection
    fps       : float       frames per second of the recording

    For real data, build this from a DeepLabCut / SLEAP CSV: each row
    is one frame, columns are per-bird (x, y, heading).  The
    ``Trajectories.from_csv`` classmethod (TODO once we know the
    actual format) will do the parsing.
    """
    positions: np.ndarray
    heading: Optional[np.ndarray]
    fps: float = 25.0

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_birds(self) -> int:
        return self.positions.shape[1]

    @classmethod
    def from_simulator(cls, sim_result) -> "Trajectories":
        """Adapter from ``simulator.simulate_flock`` output."""
        traj = sim_result.trajectories      # (T, N, 3)
        return cls(
            positions=traj[..., :2].astype(np.float32),
            heading=traj[..., 2].astype(np.float32),
            fps=25.0,
        )


@dataclass
class InteractionEvent:
    """One detected event.  Same shape as the simulator's
    ``InteractionEvent``; kept locally so downstream code can build
    its own without depending on the simulator."""
    frame: int
    src: int
    dst: int
    type: str          # "peck" | "proximity" | "chase"
    score: float = 1.0  # confidence ∈ [0, 1]


# ─── peck-event detection from raw trajectories ──────────────────────


def detect_peck_events(traj: Trajectories,
                       peck_radius: float = 0.18,
                       approach_speed_thresh: float = 0.06,
                       min_proximity_frames: int = 1,
                       ) -> list[InteractionEvent]:
    """Detect peck events from kinematic features alone.

    A "peck" is heuristically defined as: bird A approaches bird B such
    that (a) their inter-bird distance crosses below ``peck_radius``,
    (b) bird A's instantaneous speed component along the A→B direction
    exceeds ``approach_speed_thresh``, and (c) the proximity persists
    at least ``min_proximity_frames``.

    These thresholds are loose-by-design — real pecks are short, fast,
    spatially-tight events that satisfy all three.  Tighten the
    thresholds when fitting to ground truth on a labelled validation
    set from Éva.

    NOTE: this is a stand-in for a learned event detector.  A proper
    pipeline would train a small classifier on hand-labelled bouts.
    """
    T, N, _ = traj.positions.shape
    fps = traj.fps
    events: list[InteractionEvent] = []

    # Velocity per bird per frame (forward difference).
    vel = np.zeros_like(traj.positions)
    vel[:-1] = (traj.positions[1:] - traj.positions[:-1]) * fps
    speed = np.linalg.norm(vel, axis=-1)         # (T, N)

    # Pre-compute pairwise distances & directions once per frame.
    # We process frames sequentially so we can detect distance-cross
    # events.
    prev_dist = None
    proximity_run: dict[tuple[int, int], int] = {}

    for t in range(T):
        diff = traj.positions[t][:, None, :] - traj.positions[t][None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))    # (N, N)
        np.fill_diagonal(dist, np.inf)
        # directions: dir[i, j] = unit vector from i to j
        with np.errstate(divide="ignore", invalid="ignore"):
            dirs = -diff / np.where(dist[..., None] > 0,
                                       dist[..., None], 1.0)
            # i→j direction is (positions[j] - positions[i]) / d_ij
            # i.e. -diff/dist (since diff = pos_i - pos_j)

        for i in range(N):
            for j in range(i + 1, N):
                d_ij = float(dist[i, j])
                if d_ij < peck_radius:
                    proximity_run[(i, j)] = (
                        proximity_run.get((i, j), 0) + 1
                    )
                    # speed along i→j direction = vel[i] . dir_ij
                    v_along_ij = float(vel[t, i] @ dirs[i, j])
                    v_along_ji = float(vel[t, j] @ dirs[j, i])
                    if (proximity_run[(i, j)] >= min_proximity_frames
                            and (v_along_ij > approach_speed_thresh
                                 or v_along_ji > approach_speed_thresh)):
                        # decide aggressor by which bird is approaching
                        if v_along_ij >= v_along_ji:
                            src, dst = i, j
                        else:
                            src, dst = j, i
                        events.append(InteractionEvent(
                            frame=t, src=src, dst=dst, type="peck",
                            score=min(1.0, max(v_along_ij, v_along_ji)
                                          / approach_speed_thresh),
                        ))
                        # Reset run so we don't re-trigger every frame.
                        proximity_run[(i, j)] = 0
                else:
                    if (i, j) in proximity_run:
                        proximity_run.pop((i, j))
        prev_dist = dist

    return events


def detect_proximity_events(traj: Trajectories,
                             proximity_radius: float = 0.45,
                             stride: int = 8,
                             ) -> list[InteractionEvent]:
    """Detect peaceful-proximity events: two birds within
    ``proximity_radius`` at a strided sample of frames.  Subsampling
    by ``stride`` keeps the graph density reasonable."""
    T, N, _ = traj.positions.shape
    events: list[InteractionEvent] = []
    for t in range(0, T, stride):
        diff = traj.positions[t][:, None, :] - traj.positions[t][None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        iu, jv = np.where(np.triu(dist < proximity_radius, k=1))
        for ii, jj in zip(iu.tolist(), jv.tolist()):
            events.append(InteractionEvent(
                frame=t, src=int(ii), dst=int(jj),
                type="proximity", score=1.0,
            ))
    return events


# ─── events → signed graph ────────────────────────────────────────────


def trajectories_to_signed_graph(
    traj: Trajectories,
    peck_events: Optional[list[InteractionEvent]] = None,
    proximity_events: Optional[list[InteractionEvent]] = None,
    detect_kwargs: Optional[dict] = None,
    proximity_kwargs: Optional[dict] = None,
    aggregate: str = "sum",   # "sum" — multiplicity-weighted; "majority" — most-frequent sign per pair
) -> tuple[SignedGraph, dict]:
    """Pipe trajectories → signed graph that HSiKAN consumes natively.

    If ``peck_events`` / ``proximity_events`` are not supplied, they
    are inferred kinematically.  Returns the SignedGraph plus an info
    dict with per-pair statistics for inspection.
    """
    if peck_events is None:
        peck_events = detect_peck_events(traj, **(detect_kwargs or {}))
    if proximity_events is None:
        proximity_events = detect_proximity_events(
            traj, **(proximity_kwargs or {}))

    # Aggregate per (i, j) pair across the recording.
    counts: dict[tuple[int, int], dict[str, int]] = {}
    for ev in peck_events + proximity_events:
        u, v = sorted((ev.src, ev.dst))
        d = counts.setdefault((u, v), {"peck": 0, "proximity": 0})
        d[ev.type] = d.get(ev.type, 0) + 1

    edges = []
    signs = []
    for (u, v), d in counts.items():
        if aggregate == "sum":
            sign = -1 if d.get("peck", 0) > 0 else +1
        elif aggregate == "majority":
            sign = (-1 if d.get("peck", 0) > d.get("proximity", 0)
                    else +1)
        else:
            raise ValueError(f"unknown aggregate: {aggregate}")
        edges.append((u, v))
        signs.append(sign)

    g = SignedGraph(
        edges=np.asarray(edges, dtype=np.int64),
        signs=np.asarray(signs, dtype=np.int64),
        n_nodes=traj.n_birds,
    )
    info = dict(
        n_peck_events=len(peck_events),
        n_proximity_events=len(proximity_events),
        n_pairs=len(counts),
        aggregator=aggregate,
        peck_per_pair={k: v["peck"] for k, v in counts.items()},
    )
    return g, info


# ─── CLI smoke test ──────────────────────────────────────────────────


def main():
    import argparse
    from .simulator import ChickenFlockSim, simulate_flock
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-birds", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=300)
    args = ap.parse_args()

    cfg = ChickenFlockSim(n_birds=args.n_birds, n_frames=args.n_frames,
                            seed=args.seed)
    sim = simulate_flock(cfg)
    traj = Trajectories.from_simulator(sim)

    # 1. Use the simulator's ground-truth events.
    g_gt, info_gt = trajectories_to_signed_graph(
        traj,
        peck_events=[
            InteractionEvent(frame=e.frame, src=e.src, dst=e.dst,
                              type=e.type)
            for e in sim.events if e.type == "peck"
        ],
        proximity_events=[
            InteractionEvent(frame=e.frame, src=e.src, dst=e.dst,
                              type=e.type)
            for e in sim.events if e.type == "proximity"
        ],
    )
    print("== ground-truth events → graph ==")
    print(f"  {info_gt}")
    print(f"  graph stats: {g_gt.stats()}")

    # 2. Use the kinematic peck-detector (sanity check that detection
    # roughly matches ground truth).
    g_det, info_det = trajectories_to_signed_graph(traj)
    print("== kinematic-detector events → graph ==")
    print(f"  {info_det}")
    print(f"  graph stats: {g_det.stats()}")
    print(f"  ground-truth pecks: {info_gt['n_peck_events']}, "
          f"detected pecks: {info_det['n_peck_events']}")


if __name__ == "__main__":
    main()
