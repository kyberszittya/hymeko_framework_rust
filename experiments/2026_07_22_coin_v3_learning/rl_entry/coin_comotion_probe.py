"""Discriminating probe (§4/§8 pre-check): can DEPLOYABLE co-motion separate seed 1447's genuine unilateral PUSH
from acquisition brushes? Record per-step (left/right contact, coin_xy, left_tip_xy, right_tip_xy) via FK, then for
every SAME-SIDE unilateral run compute coin displacement, tip displacement, directional agreement and slip over a
trailing window. Print push windows (long, delivering) vs brush windows (short, acquisition). If they don't
separate -> PHASE_GATE_UNILATERAL_COMOTION_BLOCKED.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import ActorEvalWrap, load_frozen_clip_actor  # noqa: E402
from hymeko_rl.experiments.coin_neutral_start import neutral_env  # noqa: E402

PI0 = sys.argv[1]
KIN_W = 4          # trailing kinematic window (steps)
UNI_MIN = 6        # consecutive same-side steps for the slow path


def signals(inner):
    m = inner._planar_metrics
    lc, rc = bool(m.left_contact), bool(m.right_contact)
    coin = np.array(m.disk_pos[:2], np.float64)                                  # copy
    ltip = np.array(inner.data.site_xpos[inner._tip_sites[0]][:2], np.float64)   # copy (live MuJoCo view otherwise)
    rtip = np.array(inner.data.site_xpos[inner._tip_sites[1]][:2], np.float64)   # copy
    return lc, rc, coin, ltip, rtip


def roll(actor, seed, horizon=360):
    env, cf = neutral_env(prefix_steps=0); inner = cf._env
    env.set_stage(0); env.reset(seed=int(seed))
    ew = ActorEvalWrap(actor)
    rows = []
    for _t in range(horizon):
        nf = np.asarray(inner.node_features(), np.float32).flatten()
        inner.step(np.clip(ew.act(nf), -4, 4).astype(np.float32))
        lc, rc, coin, ltip, rtip = signals(inner)
        rows.append((lc, rc, coin, ltip, rtip))
    return rows


def same_side_runs(rows):
    """Yield (side, start, end) maximal runs where exactly one side is in contact and it's the SAME side throughout."""
    runs = []; side = None; start = None
    for t, (lc, rc, *_ ) in enumerate(rows):
        cur = "L" if (lc and not rc) else ("R" if (rc and not lc) else None)
        if cur is not None and cur == side:
            continue
        if side is not None:
            runs.append((side, start, t))
        side, start = cur, (t if cur else None)
    if side is not None:
        runs.append((side, start, len(rows)))
    return [(s, a, b) for (s, a, b) in runs if s is not None]


def comotion(rows, side, t):
    """Trailing-window co-motion at step t for a contacting side."""
    a = max(0, t - KIN_W)
    coin_a, coin_t = rows[a][2], rows[t][2]
    tip_i = 3 if side == "L" else 4
    tip_a, tip_t = rows[a][tip_i], rows[t][tip_i]
    dcoin = coin_t - coin_a; dtip = tip_t - tip_a
    ncoin, ntip = np.linalg.norm(dcoin), np.linalg.norm(dtip)
    dot = float(dcoin @ dtip) / (ncoin * ntip + 1e-12)
    slip = float(np.linalg.norm(dcoin - dtip))
    return ncoin, ntip, dot, slip


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    print(f"{'seed':>6} {'side':>4} {'run':>10} {'len':>4} {'|dcoin|':>8} {'|dtip|':>8} {'dot':>6} {'slip':>7}  kind")
    push_feats, brush_feats = [], []
    for s in sorted(HEADLINE):
        rows = roll(pi0, s)
        for side, a, b in same_side_runs(rows):
            L = b - a
            if L < 3:
                continue
            # co-motion evaluated at the END of the qualifying window (t = a+UNI_MIN-1 if long enough, else b-1)
            t = min(b - 1, a + max(UNI_MIN, KIN_W) - 1)
            ncoin, ntip, dot, slip = comotion(rows, side, t)
            kind = "PUSH?" if L >= UNI_MIN else "brush"
            (push_feats if L >= UNI_MIN else brush_feats).append((ncoin, dot, slip))
            mark = " <== 1447" if s == 1447 else ""
            print(f"{s:>6} {side:>4} {str((a,b)):>10} {L:>4} {ncoin:>8.4f} {ntip:>8.4f} {dot:>6.2f} {slip:>7.4f}  {kind}{mark}")
    print("\n=== separation ===")
    if push_feats:
        pf = np.array(push_feats)
        print(f"long-run (>= {UNI_MIN}) windows n={len(pf)}: |dcoin| med {np.median(pf[:,0]):.4f} "
              f"dot med {np.median(pf[:,1]):.2f} slip med {np.median(pf[:,2]):.4f}")
    if brush_feats:
        bf = np.array(brush_feats)
        print(f"short-run (< {UNI_MIN}) windows n={len(bf)}: |dcoin| med {np.median(bf[:,0]):.4f} "
              f"dot med {np.median(bf[:,1]):.2f} slip med {np.median(bf[:,2]):.4f}")
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
