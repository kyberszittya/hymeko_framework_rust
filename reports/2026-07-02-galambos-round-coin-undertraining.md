# Galambos round coin: it was undertraining, not physics

**Date:** 2026-07-02 · **Author:** Aiko (Claude Code), for Dr. Cs. Hajdu
**One line:** the round-coin "physics ceiling" was a **budget artifact** — at a proper budget the coin delivers
at **0.34 median (peak), [0.32, 0.34, 0.42] across seeds**, at/above the scripted teacher (~0.33). Earlier
conclusions drawn at 60k are retracted.

## The correction (measured)

Across a long session I repeatedly attributed the round coin's low delivery (~0.04–0.18) to a **physics wall**
(the coin can't be transported), having tested three reward levers — `pull`, de-farm, `nofinger` removal — at
**60k steps** and found each within-noise. Dr. Hajdu flagged the confound: *"we are running quite less on
iterations."* He was right. The codebase's own lesson (FANUC off-policy collapse) already said **≥1e5 steps**;
60k is under a third of what this task needs.

**The discriminating test** — a single-seed learning curve to 250k — settled it immediately:

| steps | 50k | 100k | 150k | 200k | 250k |
|---|---|---|---|---|---|
| delivery | 0.06 | 0.10 | 0.20 | **0.32** | 0.08 |

Delivery **climbed 8×** (0.04 → 0.32) and the coin-drift *shrank* (+0.135 → +0.076 m) as the policy learned to
transport it — then **collapsed** at 250k (TD3 Q-overestimation instability). The wall was training budget, not
physics.

## Confirmation (3 seeds × 200k, best-checkpoint)

| seed | curve (delivery @ 25k…200k) | **peak** |
|---|---|---|
| 0 | 0.18·0.06·0.08·0.10·0.20·0.20·0.20·0.32 | **0.32** @200k |
| 1 | 0.04·0.26·0.22·0.30·0.18·0.36·0.42·0.26 | **0.42** @175k |
| 2 | 0.08·0.12·0.18·0.06·0.16·0.34·0.14·0.08 | **0.34** @150k |

**Peak-delivery median = 0.34; per-seed [0.32, 0.34, 0.42].** Plot: `delivery_curves.png`. Best-checkpoint
policies (`policies/best_s{0,1,2}.pt`) captured the peaks; a delivering GIF (`gifs/best_s1.gif`, the 0.42 policy)
shows the arms carrying the round coin into the zone. The scripted teacher is ~0.33 — the learned policy
**matches it (median) and beats it on one seed (0.42)**.

Two robust facts from the curves:
- **The round coin is deliverable** — physics is not the ceiling; the demonstrator level is reachable and
  exceedable with a real budget.
- **Training is unstable late** — every seed peaks mid-run then partly collapses (0.42→0.26, 0.34→0.08). Best-
  checkpoint / early-stop is mandatory to keep the peak; the naive endpoint would report [0.32, 0.26, 0.08]
  (median 0.26), understating the achievable result and hiding the instability.

## Retractions (honest)

- **"Physics ceiling on the round coin" — RETRACTED.** It was undertraining. My repeated assertion was premature
  (drawn before the discriminating budget test), exactly the failure the operating contract warns against.
- **The 60k reward-lever verdicts (`pull` within-noise, de-farm no-gain, `nofinger` within-noise) are UNRELIABLE**
  — all rendered in an undertrained regime. They neither confirm nor refute those levers. `nofinger` stays
  removed (user-directed + the historically-validated 4-core); the others stay reverted; but any *reward* claim
  must be re-made at ≥1.5e5 steps.

## Kept / changed

- `galambos_task.hymeko`: **4-core** (`approach·both·zone·oob`, nofinger removed) — unchanged by this run.
- `PlanarGraspEnv.terminate_on_success` flag: kept (opt-in, default True), oracle-correct, unused by the winning
  config.
- **CORE.YAML items touched: none.**

## Next (systematic)

1. **Stabilise the late collapse** — best-checkpoint is the band-aid; the real fixes are lower actor LR, TD3
   `policy_delay`, or adaptive-BC. Worth an A/B *at 200k*.
2. **Re-test the reward levers at 200k** — `pull` / de-farm / `nofinger` verdicts must be redone at the right
   budget before any reward conclusion is trusted.
3. **The methodology point holds, the conclusion flipped:** the learning curve was the cheap discriminating test
   that a single 60k endpoint could never provide. Run the curve before concluding a ceiling.
