# Coin release-guard line — G0 conditional-observability audit

**2026-07-27 · branch `recovery/coin-r3-physical-intent-decoder` (worktree `hymeko_coin_r9_wt`) · dev s1/s3, held-out s4/s7 untouched**

## Why

The R10 K6-decomposition localised the s1 delivery barrier to *coast-landing precision*, and C2.6 found a wide global coast
spread that defeated a point-estimate release guard. The user posed the decisive question: is that spread **irreducible**, or
**state-conditionally narrowable** given the pre-release state + short history? G0 answers it — a measurement, not a controller
search. The LAUNCH/RELEASE guard under test is a **distinct object** from the R6 final settle certificate
(`theta_option/release_certificate.py`): the guard decides *when to let go so the free coast lands in the corridor*; the R6
cert grades *whether the coin ended at rest in the zone*.

## Method

`hymeko_rl/experiments/coin_release_guard.py --g0`. Force APPROACH(qdot ∈ {1.6, 2.0, 2.4}) → passive release at step
k ∈ [2, 17] on dev cradles s1, s3; capture the observable pre-release state/history (d_remaining, v_par + short derivative,
spin, v_lat, squeeze force, imbalance, APPROACH impulse history, time-since-contact) and the post-release coast outcome
(terminal dtz, min dtz, terminal speed, zone-reached). Measure the **global** landing spread (IQR) vs the **state-conditioned**
spread (leave-one-out k-NN regression residual on standardised features), plus a determinism check and the zone-selection LOO
accuracy. Reuses `_run_approach`/`_phys_hook`/`build_panel` — no rollout or metric re-implementation.

## Results

| quantity | value | reading |
|---|---|---|
| determinism (same snapshot, repeated) | **bit-identical** | coast spread is **state-dependence**, not aleatoric friction noise |
| global landing IQR | 231.8 mm | releases land anywhere in [27.6, 425] mm |
| state-conditioned residual (k-NN LOO) | **9.5 mm** | landing is predictable to ~9.5 mm from the observable state |
| conditional / global ratio | **0.041** | the spread is **highly** conditionally narrowable — guard would be accurate |
| reach-zone (≤20 mm) | **0 / 38** | **no** passive release in the reachable set lands in the zone (best 27.6 mm) |
| top features ↔ landing | squeeze 0.82, v_lat 0.81, spin 0.63 | grip/lateral/spin state at release dominates the coast |

Plot: `release_guard_g0.png` (landing-vs-d_remaining coloured by v_par; feature↔landing bars).

**Teacher-mechanism probe.** The teacher delivers by **passive-coast, not grip-transport**: contact = 0 throughout the zone
(`grip_frac_in_zone = 0.0`). It releases the coin **close AND still moving** and lets it coast in:

| cradle | teacher release | teacher lands | my APPROACH closest reachable release | my best landing |
|---|---|---|---|---|
| s1 | dtz **23.0 mm** @ v_par **0.131** | 18.5 mm | dtz 77.9 mm @ v_par 0.207 (far+moving) | 27.6 mm |
| s3 | dtz 59.0 mm @ v_par 0.451 | 0.5 mm | dtz 85.5 mm @ v_par 0.289 | 21.3 mm |

## Conclusion — the barrier is *never let the coin stop*, not release-timing observability

Three measured facts, combined:
1. The coast-landing is **highly conditionally observable** (ratio 0.041, residual 9.5 mm) — a learned guard would be accurate;
   the C2.6 "wide spread" was **state-aliasing**, exactly as the user hypothesised.
2. A **delivering passive release exists** (the teacher: s1 → 18.5 mm, s3 → 0.5 mm, both free-coast).
3. But **no release in the current APPROACH's reachable set lands in the zone** (s1 best 27.6 mm), because the teacher releases
   the coin **close + moving** (s1: 23 mm @ 0.131) while my APPROACH can only release it **close + stopped** (lands 27.6 mm) or
   **far + moving** (overshoots).

The root cause is the scaffold's **distance-proportional velocity servo** `v_ref = k_d · d_remain`: velocity → 0 as the coin
nears the target, dropping it into the **R1 stiction regime**. This finally reconciles the whole arc: a **stopped** gripped
coin cannot be restarted (R3-C micro-transport and C3-D velocity-matched capture both failed transporting *from rest* — static
friction > the gentle grip can supply), while a **moving** gripped coin stays in the kinetic regime (the teacher maintains
v_par 0.13 to dtz 23 mm and never lets it stop). **The barrier is: keep the coin moving until a close-and-moving release** —
not release-timing observability (excellent), not grip-transport (the teacher never uses it), not the K6 monitor (calibrated).

## Next (discriminating test, redirected)

Not "learn release timing" (the guard is learnable but has no reachable in-zone target). The discriminating test is an
**APPROACH with a velocity floor** — hold v_par ≈ 0.13 (constant-velocity transport, *not* distance-proportional) until
dtz ≈ 23 mm, then release — so the reachable release set *brackets* the zone. If it reaches a close-and-moving release whose
free coast lands ⊂ [0, 20] mm on s1 (and s3, already at 21.3 mm), the G0 guard selects it and the stiction hypothesis is
confirmed. If the coin still stalls before dtz 23 mm, the stiction is deeper and the strategic pivot stands.

## Artifacts / gates

`…/release_guard_g0.json`, `release_guard_g0.png`; module `coin_release_guard.py` (`--g0`). ruff clean; all fns < CC 15;
CORE.YAML untouched; blind panel sealed; held-out s4/s7 never touched. Determinism, global/conditional spread, and the
teacher release-state comparison are all reproducible from the committed code.

---

## VELOCITY_FLOOR / KINETIC transport (built + hand-audited) — mechanism validated, contact-maintenance is the learnable skill

Built a KINETIC phase in `HybridApproachController` (`kinetic_transport=True`): after APPROACH builds momentum, hand to a
**sustained transport-velocity** servo (`base_qref = clip(k_q·v_floor, 0, kinetic_vcap)` — constant, *not* the
distance-proportional `k_d·d_remain` that decays to 0 and stictions the coin) with a **light grip**, releasing to a passive
coast once the coin is close-AND-moving. It preempts LAUNCH/REACHABILITY (KINETIC owns the release). Default
`kinetic_transport=False` ⇒ update-zero (a default controller never enters KINETIC; test-asserted).

**Diagnosis against the teacher trace (the design driver).** The teacher transports at **fn ≈ 0.95 N** (light grip → the coin
**slides**) with **sustained v_par ≈ 0.25** to dtz 30 mm, then the grip fades and it releases at 23 mm / v_par 0.13 → coasts
to 18.5 mm. Two hand-tuning failure modes were measured and fixed in turn: (1) a **firm grip** (fn 3–4 N) *clamps* the coin —
it oscillates in place and stalls (bounces backward); (2) flooring the servo at the *release* velocity (0.13) capped by the
base `v_max` pushes at only ~0.65 rad/s — the coin decelerates and stalls. The sustained-transport + light-grip design fixes
both **in direction**: the coin now **transports while moving, 75 → 48 mm, with no stiction stall** — a genuine win over the
R1 wall that arrested every prior from-rest attempt (R3-C, C3-D).

**But hand-tuning does not reach the release corridor.** A grip × momentum sweep (qdot_approach ∈ {2.4, 2.8}, kinetic_squeeze
∈ {0.08…0.16}, v_floor ∈ {0.20…0.32}) **loses the delicate light bilateral contact at ~48–56 mm** (fn fades to 0 as the coin
slides ahead of the tips), well short of the teacher's 23 mm release point — best min_dtz **48 mm**, still worse than the
27.6 mm free-coast. **`VELOCITY_FLOOR_RELEASE_MANIFOLD` does NOT pass by hand-tuning** (0 in-zone releases).

**Conclusion.** The strategy is confirmed correct (keep the coin moving — no stiction stall), and the barrier is now precisely
localised to **maintaining delicate light contact while the coin slides to a close-and-moving release** — the teacher's
optimized-θ skill. This is the legitimate learning target the release-guard line was opened for: G0 already proved the
coast-landing is state-observable (residual 9.5 mm), so a small learned/searched transport-and-release policy (grip + push
profile that holds contact to the corridor, with the G0 predictor as the release guard) is the well-posed next step — not more
hand grip-tuning. If a bounded search over the KINETIC transport still cannot hold contact to a delivering release, the
strategic pivot stands.

Verdict: `COAST_LANDING_STATE_OBSERVABLE · KINETIC_TRANSPORTS_WITHOUT_STALL · LIGHT_CONTACT_MAINTENANCE_IS_THE_LEARNABLE_SKILL`.
Gates: KINETIC mode `hybrid_approach.py`; 8 tests pass; ruff clean; all fns < CC 15; CORE.YAML untouched; blind sealed; s4/s7 untouched.

---

## G-VF bounded CEM over the KINETIC transport profile — FAIL (strategic-pivot boundary)

Generalised KINETIC to a **6-param position-varying profile** (forward taper `v_hi → v_lo` + squeeze decay `→ 0` near release —
the teacher's shape) and ran a bounded CEM (216 evals, s1 dev) with a **lexicographic scorer on the realized contact**
(safety ≻ K6 ≻ reached-moving-release ≻ light-contact-maintained ≻ no-sign-reversal ≻ closeness; clamp + early-contact-loss
penalties scored on *measured* fn, not the squeeze command). Snapshot: `release_guard_gvf.json`.

| quantity | value |
|---|---|
| teacher positive control | **K6 True** (task feasible from the same cradle) |
| best K6 / released | **False / False** |
| best min_dtz | **50.9 mm** (converged from iteration 1 — flat) |
| best contact_frac / sign-reversals | 0.69 / **0** (smooth, well-contacted transport) |
| verdict | **`KINETIC_CANNOT_HOLD_CONTACT_TO_CORRIDOR`** (the user's FAIL condition) |

**Reading.** The "keep the coin moving" strategy is **validated** — the searched profile transports the coin smoothly with
good light contact (69%, zero sign-reversals, no stiction). But **no profile holds contact past ~50 mm** to the 20–30 mm
release corridor (an earlier qdot 2.4–2.8 momentum sweep also plateaued at ~48 mm — it is not a momentum limit). The teacher
(positive control, K6 True) proves a solution exists in the **full torque-primitive θ space**. Therefore the limit is the
**hand/profile servo's tip-trajectory**: it pushes the tips toward the *zone*, but does not **track the sliding coin** the way
the teacher's learned θ does, so contact is lost at ~50 mm. The remaining gap is precisely the **coin-following tip-trajectory
to a close-and-moving release** — what θ encodes and a deterministic scaffold + parametric profile cannot.

**Strategic-pivot boundary (per the pre-registered stop condition).** Not more profile/squeeze tuning. Two options:
1. **A small learned KINETIC torque policy** over the transport segment only (θ *is* a learned torque primitive; learn the
   coin-following tip-trajectory that hand-design can't, with the G0 predictor as the release guard). Keeps the validated
   "keep it moving" strategy; puts the learning exactly where the measurement says the skill lives.
2. **Accept the arc negative**: fine transport-to-close is a per-cradle learned skill (the teacher's θ); the deterministic
   *single-controller* generalization goal is not reachable by hand-design — it requires learning.

Gates: KINETIC profile `hybrid_approach.py`; search `coin_release_guard.py --gvfsearch`; 8 tests pass; ruff clean; all
fns < CC 15; CORE.YAML untouched; blind sealed; s4/s7 untouched; teacher = positive control.
