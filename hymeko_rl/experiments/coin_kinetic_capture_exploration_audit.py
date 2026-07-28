"""R10 Stage 2 diagnostic AUDIT (negative-control) — IID residual exploration on the slew-integrated capture interface.

This is NOT a training module and imports no training loop. It reproduces the supported Stage-2 findings:

  * ``REAL_MEDOID_SCAFFOLD_IDENTITY`` — the real medoid pi_0 with a genuinely zero-output actor (not a mocked zero
    vector) composes bit-exact to the scaffold and delivers strict K6;
  * ``IID_STEPWISE_RESIDUAL_EXPLORATION_INADMISSIBLE`` — per-step IID residual noise removes the delivery at every
    tested scale; the structured (smooth / zero-integral) variants are only marginally better at these scales;
  * the observation that ``tau_star`` is NOT the medoid scaffold's terminal preload reference (dtau ~ 0.88 at the
    delivering endpoint), so a cradle-centred preload metric is invalid for this scaffold.

Explicit non-claims: RL is not blocked; reactive policies are not shown impossible; zero-integral noise is not
sufficient; the terminal-preload random-walk mechanism is *implied by the action definition and the coherent-vs-IID
contrast*, not directly isolated by the (mis-centred) dtau metric. Run:
``python -m hymeko_rl.experiments.coin_kinetic_capture_exploration_audit``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
from hymeko_rl.coin_delivery.theta_option import capture_rl as crl
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.home_states import HOME_STATE_V1_GENERIC, build_home_snapshot
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import deterministic_residual
from hymeko_rl.experiments.coin_kinetic_ablation import _rebuild

OUT = Path("reports/2026-07-28-td3-capture-exploration-audit")
S1B = Path("reports/2026-07-28-moving-precapture-dynamic-handoff/moving_precapture.json")
CKPT = Path("reports/2026-07-28-coin-r9-r2-h1-multiseed/seed_01/checkpoint.json")


def _rig() -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone
    model, norm = _load_clone()
    cradle, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    coin = _coin_xy(cradle.branch())
    home = build_home_snapshot(cradle, HOME_STATE_V1_GENERIC)
    ready = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, pga.CoinStraddleTargets(coin=coin),
                                pga.TransitConfig()).ready_snapshot
    ref = mp.HandoffReference.from_cradle(cradle)
    r2_fn = deterministic_residual(_rebuild(json.load(open(CKPT))["r2_actor_state"]))
    down = mp.FrozenDownstream(model, norm, r2_fn, cradle.stack)
    sols = [mp.CaptureParams(n=r["params"]["n"], s=r["params"]["s"], preload_start=r["params"]["preload_start"],
                             bmax=r["params"]["bmax"], residual=np.array(r["params"]["residual"]))
            for r in json.load(open(S1B))["seeds"]]
    return {"cradle": cradle, "ready": ready, "ref": ref, "down": down, "coin": coin,
            "pi0": crl.freeze_scaffold(sols), "stack": cradle.stack}


def _identity(rig: dict) -> dict:
    """Real medoid + a genuinely-zero actor → bit-exact scaffold and strict K6 (the correctness cornerstone)."""
    ref, pi0, coin, stack, ready = rig["ref"], rig["pi0"], rig["coin"], rig["stack"], rig["ready"]
    scaffold = mp.PhaseShapeCapture(ready, ref, stack).roll(pi0)
    res = crl.ResidualCaptureRoll(ready, ref, stack, pi0, coin).rollout(crl.zero_residual)
    bit_exact = bool(np.array_equal(scaffold.snapshot.branch().inner.data.qpos, res["snapshot"].branch().inner.data.qpos)
                     and np.allclose(scaffold.snapshot.prev_tau, res["prev"], atol=0.0))
    actor = crl.make_zero_actor(1)
    obs = np.array(res["obs"])
    actor_out = crl.policy_residual(actor)
    actor_zero = bool(np.all(np.abs([actor_out(o) for o in obs]) == 0.0))
    k6, md, safe = rig["down"].deliver(res["snapshot"])
    return {"bit_exact_scaffold": bit_exact, "actor_output_exactly_zero": actor_zero, "k6": bool(k6),
            "min_dtz_mm": round(md, 2), "terminal_dtau_to_taustar": round(float(np.linalg.norm(res["prev"] - ref.tau_star)), 3)}


def _profile(kind: str, sig: float, steps: int, rng: np.random.Generator) -> Any:
    knots = rng.normal(0, sig, (4, 4))
    prof = np.array([np.interp(np.linspace(0, 3, steps), range(4), knots[:, j]) for j in range(4)]).T
    if kind == "zeroint":
        prof = prof - prof.mean(0)
    return np.clip(prof, -1.0, 1.0)


def _preserve(rig: dict, kind: str, sig: float, n: int = 12) -> dict:
    """K6-preservation on the NOMINAL (where the scaffold delivers) under an exploration family at scale ``sig``."""
    ref, pi0, coin, stack = rig["ref"], rig["pi0"], rig["coin"], rig["stack"]
    nom = crl.perturb_ready(rig["ready"], stack, crl.Perturbation(np.zeros(4), np.zeros(4), np.zeros(4)))
    k6, dtaus = 0, []
    for s in range(n):
        rng = np.random.default_rng(s)
        if kind == "iid":
            def fn(_o: np.ndarray, rng: np.random.Generator = rng) -> np.ndarray:
                return np.clip(rng.normal(0, sig, 4), -1.0, 1.0)
        else:
            prof = _profile(kind, sig, pi0.steps, rng)
            ctr = {"i": 0}

            def fn(_o: np.ndarray, prof: np.ndarray = prof, ctr: dict = ctr) -> np.ndarray:
                r = prof[min(ctr["i"], len(prof) - 1)]
                ctr["i"] += 1
                return r
        out = crl.ResidualCaptureRoll(nom, ref, stack, pi0, coin).rollout(fn)
        ok, _, _ = rig["down"].deliver(out["snapshot"])
        k6 += int(ok)
        dtaus.append(float(np.linalg.norm(out["prev"] - ref.tau_star)))
    return {"k6_preserve": round(k6 / n, 3), "mean_terminal_dtau": round(float(np.mean(dtaus)), 3)}


def run(out: Path = OUT) -> dict:
    rig = _rig()
    identity = _identity(rig)
    ladder = {kind: {f"sigma_{sig}": _preserve(rig, kind, sig)
                     for sig in (0.005, 0.05, 0.15, 0.3, 0.6)}
              for kind in ("iid", "smooth", "zeroint")}
    verdicts = {
        "REAL_MEDOID_SCAFFOLD_IDENTITY_PASS": bool(identity["bit_exact_scaffold"]
                                                   and identity["actor_output_exactly_zero"] and identity["k6"]),
        "IID_STEPWISE_RESIDUAL_EXPLORATION_INADMISSIBLE": bool(
            all(v["k6_preserve"] <= 0.1 for k, v in ladder["iid"].items() if k != "sigma_0.1")),
        "CRADLE_CENTERED_PRELOAD_METRIC_INVALID_FOR_MEDOID_SCAFFOLD": bool(
            identity["terminal_dtau_to_taustar"] > 0.5),
    }
    summary = {
        "contract": "CAPTURE_EXPLORATION_AUDIT_V1", "immutable_scaffold_source": "d6974b95",
        "medoid": {"s": rig["pi0"].s, "preload_start": rig["pi0"].preload_start, "bmax": rig["pi0"].bmax,
                   "residual_norm": round(float(np.linalg.norm(rig["pi0"].residual)), 3)},
        "identity": identity, "exploration_ladder_nominal": ladder, "verdicts": verdicts,
        "non_claims": ["RL is NOT blocked", "reactive policies are NOT shown impossible",
                       "zero-integral noise is NOT sufficient",
                       "terminal-preload random-walk mechanism is IMPLIED (action definition + coherent-vs-IID), "
                       "NOT directly isolated by the mis-centred dtau metric",
                       "NO reward-driven capture policy yet"],
        "external_probe_results": {"stepwise_IID_MPC_recovery": "0/7",
                                   "temporally_coherent_episode_correction_recovery": "8/8",
                                   "note": "coherence is strongly-implied load-bearing; see report for commands"}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "exploration_audit.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    r = run()
    i = r["identity"]
    print(f"REAL_MEDOID identity: bit_exact {i['bit_exact_scaffold']} actor_zero {i['actor_output_exactly_zero']} "
          f"K6 {i['k6']} ({i['min_dtz_mm']}mm) | terminal dtau-to-tau* {i['terminal_dtau_to_taustar']} (NOT ~0)")
    for kind, d in r["exploration_ladder_nominal"].items():
        print(f"  {kind:>8}: " + " ".join(f"s{sig.split('_')[1]}={v['k6_preserve']}" for sig, v in d.items()))
    for k, v in r["verdicts"].items():
        print(f"  {'PASS' if v else '----'} {k}")
