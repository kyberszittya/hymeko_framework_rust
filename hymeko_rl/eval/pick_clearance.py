"""Committed, reusable clearance diagnostics for the FANUC pick-place scripted expert.

Promotes the previously-ephemeral ``scratchpad/pick_clearance.py`` into a versioned evaluation module so the v2
clearance gate is reproducible and its evidence is not lost across sessions (design bundle
``docs/plans/2026-07-07-pick-place-v2-waypoint-planner/``).

It rolls the **scripted expert** (``env.expert_action``, dispatched on ``expert_version``) on
:func:`~hymeko_rl.viz.render_pick_place.fanuc_pick_env` at the production horizon and measures, per episode, the
contact/clearance signature the gate grades: whether the gripper strikes the table *before* it is over the object,
the signed finger↔table clearance during transit, and lift/place success (with the physics-divergence guard kept
on, so a blow-up can never fake a success — the 2026-06-30 lesson). It **reads** the env only; it changes no
trajectory, reward, or training.

    # v1 smoke (reproduces the KNOWN dirty signature — expected to FAIL the gate):
    python -m hymeko_rl.eval.pick_clearance --version 1 --episodes 4 --seed0 50000 \
        --out reports/figures/pick_place_clean_expert/v1_clearance_smoke

    # v2 gate (only once the v2 expert is implemented + smoke is sane):
    python -m hymeko_rl.eval.pick_clearance --version 2 --episodes 32 --seed0 50000 \
        --out reports/figures/pick_place_clean_expert/v2_clearance_gate
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import hypot, inf, isfinite
from pathlib import Path
from typing import Any

import mujoco

from hymeko_rl.eval.evaluate import LiftPlaceMetric, eval_metric, results_to_csv
from hymeko_rl.viz.render_pick_place import expert_action_fn, fanuc_pick_env

# The over-object threshold matches the env's own ``approach_contact`` gate (tool within 6 cm horizontally of the
# object counts as "over it"); a table contact while farther than this is an approach/transit strike.
_OVER_OBJ_THRESH = 0.06
_LIFT_THRESH = 0.035
_DIVERGE_QACC = 5_000.0
_DISTMAX = 1.0                 # cap for mj_geomDistance search (m); returned distance saturates here if farther
# Gate thresholds (docs/plans/2026-07-07-pick-place-v2-waypoint-planner/clearance_gate.md).
_GATE_LIFT = 0.90             # preferred
_GATE_PLACE = 0.80           # preferred
_TRANSIT_CONTACT_NEAR_ZERO = 0.02   # "0 or near-zero" tolerance for the transit finger↔table contact rate


@dataclass(frozen=True)
class EpisodeClearance:
    """One episode's clearance signature.

    # Postconditions all fields are plain scalars (JSON/CSV-safe); ``*_step`` are 1-indexed step counts or ``None``
    when the event never occurred; ``min_transit_clearance`` is signed (metres), negative = table penetration.
    """

    seed: int
    lift: int
    place: int
    obj_to_target: float
    length: int
    first_finger_table_step: "int | None"
    first_gripper_table_step: "int | None"
    first_over_object_step: "int | None"
    min_transit_clearance: float
    forbidden_pre_object: bool
    transit_finger_contact_frac: float
    phase: "str | None"
    diverged: bool


class ClearanceMetric:
    """A :class:`~hymeko_rl.eval.evaluate.RolloutMetric` that records the pick-place clearance signature per step
    and embeds a :class:`LiftPlaceMetric` for lift/place (with the divergence guard). Static geom/body ids are
    resolved once from ``env`` at construction; per-episode state is cleared in :meth:`reset`.

    # Preconditions ``env`` is a reset-able :class:`PickPlaceEnv` exposing ``_b_tool/_b_obj/_b_fl/_b_fr/_manip``.
    # Invariants reads env state only — never steps or mutates it.
    """

    def __init__(self, env: Any, *, lift_thresh: float = _LIFT_THRESH, diverge_qacc: float = _DIVERGE_QACC,
                 over_obj_thresh: float = _OVER_OBJ_THRESH, distmax: float = _DISTMAX) -> None:
        self.over_obj_thresh, self.distmax = float(over_obj_thresh), float(distmax)
        self._lp = LiftPlaceMetric(lift_thresh=lift_thresh, diverge_qacc=diverge_qacc)
        self._finger_bodies = {int(env._b_fl), int(env._b_fr)}
        self._other_manip = {int(b) for b in env._manip} - self._finger_bodies
        tg = int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table"))
        self._table_geom = tg
        # forbidden-contact detection falls back to the env's no-touch set (floor+table) if "table" is unnamed.
        self._surf_geoms = {tg} if tg >= 0 else {int(g) for g in env._no_touch}
        self._finger_geoms = [g for b in self._finger_bodies
                              for g in range(int(env.model.body_geomadr[b]),
                                             int(env.model.body_geomadr[b]) + int(env.model.body_geomnum[b]))]
        self._seed = -1
        self.reset()

    def bind_seed(self, seed: int) -> None:
        """Record which reset seed this episode used (``eval_metric`` reseeds; the metric is told separately)."""
        self._seed = int(seed)

    def reset(self) -> None:
        self._lp.reset()
        self.length = 0
        self.first_fingtab: "int | None" = None
        self.first_griptab: "int | None" = None
        self.first_over: "int | None" = None
        self.transit_steps = 0
        self.transit_fingtab_steps = 0
        self.min_transit_clr = inf
        self.last_obj_to_target = float("nan")
        self.last_phase: "str | None" = None
        self.diverged = False

    def _table_contact(self, env: Any, bodies: "set[int]") -> bool:
        """True if any geom of a body in ``bodies`` currently contacts a work-surface geom."""
        for i in range(int(env.data.ncon)):
            con = env.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            surf = g1 if g1 in self._surf_geoms else g2 if g2 in self._surf_geoms else -1
            if surf < 0:
                continue
            other = int(env.model.geom_bodyid[g2 if surf == g1 else g1])
            if other in bodies:
                return True
        return False

    def _finger_table_clearance(self, env: Any) -> float:
        """Signed minimum distance (m) between any finger geom and the table geom; negative = penetration."""
        if self._table_geom < 0:
            return inf
        best = inf
        for g in self._finger_geoms:
            try:
                d = float(mujoco.mj_geomDistance(env.model, env.data, g, self._table_geom, self.distmax, None))
            except (AttributeError, TypeError):       # mj_geomDistance unavailable → clearance unmeasured
                return inf
            best = min(best, d)
        return best

    def on_step(self, env: Any, info: "dict[str, Any]", reward: float, done: bool) -> bool:
        if self._lp.on_step(env, info, reward, done):
            self.diverged = True
            return True                                # physics blew up — do not record this garbage step
        self.length += 1
        self.last_obj_to_target = float(info.get("obj_to_target", float("nan")))
        self.last_phase = info.get("phase") if isinstance(info.get("phase"), str) else self.last_phase
        tool = env.data.xpos[int(env._b_tool)]
        obj = env.data.xpos[int(env._b_obj)]
        horiz = hypot(float(tool[0] - obj[0]), float(tool[1] - obj[1]))
        over = horiz <= self.over_obj_thresh
        if over and self.first_over is None:
            self.first_over = self.length
        if self._table_contact(env, self._finger_bodies):
            if self.first_fingtab is None:
                self.first_fingtab = self.length
            if not over:
                self.transit_fingtab_steps += 1
        if self.first_griptab is None and self._table_contact(env, self._other_manip):
            self.first_griptab = self.length
        if not over:                                   # transit / approach phase (not yet over the object)
            self.transit_steps += 1
            self.min_transit_clr = min(self.min_transit_clr, self._finger_table_clearance(env))
        return False

    def finalize(self) -> EpisodeClearance:
        lift, place = self._lp.finalize()
        firsts = [s for s in (self.first_fingtab, self.first_griptab) if s is not None]
        first_forbidden = min(firsts) if firsts else None
        forbidden_pre_object = bool(first_forbidden is not None
                                    and (self.first_over is None or first_forbidden < self.first_over))
        frac = (self.transit_fingtab_steps / self.transit_steps) if self.transit_steps else 0.0
        clr = self.min_transit_clr if isfinite(self.min_transit_clr) else float("nan")
        return EpisodeClearance(
            seed=self._seed, lift=int(lift), place=int(place), obj_to_target=round(self.last_obj_to_target, 4),
            length=self.length, first_finger_table_step=self.first_fingtab,
            first_gripper_table_step=self.first_griptab, first_over_object_step=self.first_over,
            min_transit_clearance=round(clr, 5) if isfinite(clr) else clr,
            forbidden_pre_object=forbidden_pre_object, transit_finger_contact_frac=round(frac, 4),
            phase=self.last_phase, diverged=self.diverged)


def run_clearance(version: int, episodes: int, seed0: int) -> "list[EpisodeClearance]":
    """Roll the scripted expert on ``fanuc_pick_env(expert_version=version)`` for ``episodes`` seeded episodes.

    # Preconditions ``version in (1, 2)``, ``episodes >= 1``. # Postconditions returns one
    :class:`EpisodeClearance` per episode, seeds ``seed0 .. seed0+episodes-1``; no env mutation, no training.
    """
    if version not in (1, 2):
        raise ValueError("version must be 1 (dirty baseline) or 2 (clearance-aware)")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    env = fanuc_pick_env(expert_version=version)
    metric = ClearanceMetric(env)
    action_fn = expert_action_fn()
    results: "list[EpisodeClearance]" = []
    # eval_metric reseeds internally; mirror its seed schedule so we can label each episode's seed.
    for ep in range(episodes):
        metric.bind_seed(seed0 + ep)
        (one,) = eval_metric(env, action_fn, metric, n_episodes=1, seed0=seed0 + ep)
        results.append(one)
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return results


def aggregate(eps: "list[EpisodeClearance]") -> "dict[str, Any]":
    """Aggregate per-episode signatures into the gate's headline rates and clearance summary."""
    n = max(1, len(eps))
    clrs = [e.min_transit_clearance for e in eps if isinstance(e.min_transit_clearance, float)
            and isfinite(e.min_transit_clearance)]
    return {
        "episodes": len(eps),
        "lift_rate": round(sum(e.lift for e in eps) / n, 4),
        "place_rate": round(sum(e.place for e in eps) / n, 4),
        "forbidden_pre_object_rate": round(sum(e.forbidden_pre_object for e in eps) / n, 4),
        "transit_finger_contact_rate": round(sum(e.transit_finger_contact_frac for e in eps) / n, 4),
        "min_clearance_min": round(min(clrs), 5) if clrs else None,
        "min_clearance_mean": round(sum(clrs) / len(clrs), 5) if clrs else None,
        "diverged_rate": round(sum(e.diverged for e in eps) / n, 4),
    }


def gate_verdict(agg: "dict[str, Any]") -> "dict[str, Any]":
    """Evaluate the v2 clearance gate: criteria 1-3 HARD (no early strike / ~0 transit contact / positive
    clearance), 4-5 PREFERRED (lift/place). ``pass`` is the conjunction of the hard criteria only."""
    c1 = agg["forbidden_pre_object_rate"] == 0.0
    c2 = agg["transit_finger_contact_rate"] <= _TRANSIT_CONTACT_NEAR_ZERO
    c3 = agg["min_clearance_min"] is not None and agg["min_clearance_min"] > 0.0
    return {
        "crit1_no_early_strike": bool(c1),
        "crit2_transit_contact_near_zero": bool(c2),
        "crit3_positive_min_clearance": bool(c3),
        "pref4_lift_ge_0_90": bool(agg["lift_rate"] >= _GATE_LIFT),
        "pref5_place_ge_0_80": bool(agg["place_rate"] >= _GATE_PLACE),
        "pass": bool(c1 and c2 and c3),
    }


def _maybe_plot(out: Path, version: int, agg: "dict[str, Any]", verdict: "dict[str, Any]") -> "Path | None":
    """Best-effort aggregate bar chart; a plotting failure must not fail the run (returns ``None``)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    labels = ["lift", "place", "forbidden\npre-obj", "transit\ncontact"]
    vals = [agg["lift_rate"], agg["place_rate"], agg["forbidden_pre_object_rate"],
            agg["transit_finger_contact_rate"]]
    colors = ["#2e8b57", "#2e8b57", "#c0392b", "#c0392b"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(labels, vals, color=colors, edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"pick-place v{version} clearance — gate {'PASS' if verdict['pass'] else 'FAIL'}")
    ax.axhline(0.9, ls="--", lw=0.8, color="#666")
    png = out.with_suffix(".png")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png


def write_outputs(out: Path, version: int, episodes: int, seed0: int, eps: "list[EpisodeClearance]",
                  *, plot: bool) -> "dict[str, Any]":
    """Write ``<out>.json`` (+ ``.csv``, optional ``.png``) and return the payload dict.

    # Postconditions parent dir created; JSON holds per-episode rows + aggregate + gate verdict + provenance.
    """
    agg = aggregate(eps)
    verdict = gate_verdict(agg)
    payload = {
        "version": version, "episodes": episodes, "seed0": seed0, "horizon": "env.max_steps (620)",
        "per_episode": [asdict(e) for e in eps], "aggregate": agg, "gate": verdict,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results_to_csv(out, {f"seed{e.seed}": asdict(e) for e in eps}, key_col="episode")
    if plot:
        _maybe_plot(out, version, agg, verdict)
    return payload


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", type=int, required=True, choices=(1, 2),
                    help="scripted expert: 1 = dirty baseline, 2 = clearance-aware")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--out", type=str, required=True, help="output path stem (.json/.csv/.png appended)")
    ap.add_argument("--plot", dest="plot", action="store_true", default=True, help="write aggregate .png (default)")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    a = ap.parse_args(argv)

    print(f"[pick_clearance] v{a.version} · {a.episodes} episodes · seeds {a.seed0}..{a.seed0 + a.episodes - 1} "
          f"· horizon 620 (expert-only, no training)", flush=True)
    eps = run_clearance(a.version, a.episodes, a.seed0)
    payload = write_outputs(Path(a.out), a.version, a.episodes, a.seed0, eps, plot=a.plot)
    agg, verdict = payload["aggregate"], payload["gate"]

    for e in eps:
        print(f"  seed {e.seed}: lift={e.lift} place={e.place} len={e.length} "
              f"first_fingtab={e.first_finger_table_step} first_over={e.first_over_object_step} "
              f"min_clr={e.min_transit_clearance} forbidden_pre_obj={e.forbidden_pre_object} "
              f"transit_contact={e.transit_finger_contact_frac}", flush=True)
    print(f"\n[aggregate] lift={agg['lift_rate']} place={agg['place_rate']} "
          f"forbidden_pre_object_rate={agg['forbidden_pre_object_rate']} "
          f"transit_finger_contact_rate={agg['transit_finger_contact_rate']} "
          f"min_clearance_min={agg['min_clearance_min']} min_clearance_mean={agg['min_clearance_mean']}")
    print(f"[gate] crit1_no_early_strike={verdict['crit1_no_early_strike']} "
          f"crit2_transit_contact_near_zero={verdict['crit2_transit_contact_near_zero']} "
          f"crit3_positive_min_clearance={verdict['crit3_positive_min_clearance']} "
          f"pref4_lift={verdict['pref4_lift_ge_0_90']} pref5_place={verdict['pref5_place_ge_0_80']} "
          f"=> {'PASS' if verdict['pass'] else 'FAIL'}")
    print(f"[wrote] {Path(a.out).with_suffix('.json')} + .csv" + (" + .png" if a.plot else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
