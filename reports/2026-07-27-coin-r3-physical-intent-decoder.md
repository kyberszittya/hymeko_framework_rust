# R3 — physical-intent → deterministic authority-aware decoder: D0 pass, D2 insufficient

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-r3-physical-intent-decoder` · **Start commit:** `ce3d6f41`
(frozen R3 contract) · **Frozen contract:** `reports/2026-07-27-coin-r3-physical-intent-decoder-contract.md`.

**Result in one line:** the deterministic authority-aware decoder is a *valid inverse* — **D0 teacher round-trip = 4/4**
(every teacher intent decodes back to a K6-delivering θ, held-out included) — but a **learned** predictor of the intent from
the dev cradles does **not** transfer: **D2 = dev 2/2, held-out 0/2, total 2/4**, the *same* ceiling as R1 (flat) and R2
(relational). Verdict **`PHYSICAL_INTENT_FACTORISATION_ALONE_INSUFFICIENT`** (Case C). **SAC/TD3 remain BLOCKED.**

---

## 1. The isolated axis (frozen everything else)

R2's basin audit proved the held-out failure is a **wrong factorisation**: a direct cradle → 6-D θ regressor lands 5–8
search-stds from the working basin. R3 replaces the *target of prediction* with a **cradle-agnostic physical intent** and a
**deterministic authority-aware decoder** that turns intent → cradle-specific θ. Frozen and reused unchanged: the physics /
`CradleSnapshot`, the 6-D PUSH→BRAKE→RELEASE velocity-feedback option, the budget-8 centre-inclusive search, frozen K6
(zone 0.02 m, settle 0.06 m/s, dwell 6), the 4-state panel (s1,s3 dev; s4,s7 held-out), held-out discipline, the teacher
sets. **CORE.YAML items touched: none.**

## 2. Intent vocabulary + extraction (Stage 1)

`physical_intent.py` — 7 roles in the **canonical** frame (mirror-invariant intent). Six map invertibly to θ through the
measured authority; `peak_velocity` is a load-bearing diagnostic (a *dynamic* outcome — s3 reaches 0.45 m/s with a tiny
forward push, so it does not factor as forward×ramp).

| role | unit | θ | authority | extract (canonical) |
|---|---|---|---|---|
| forward_drive | m/s | forward_mag θ[1] | forward_push_reach | θ₁·reach/slew |
| peak_velocity | m/s | — (achievable-estimate + weak-link) | fwd_reach·ramp | rollout peak_coin_speed |
| lateral | m/s | balance θ[2] | lateral_reach | θ₂·reach/slew |
| squeeze | v_n | squeeze_mag θ[0] | normal_force_reach (B_τ null) | θ₀·reach/slew |
| brake_entry | frac horizon | ramp_steps θ[3] | — | θ₃/H |
| braking_demand | N·m/(m/s) | brake_gain θ[5] | brake_opposed_reach | θ₅·reach/slew |
| release | frac horizon | release_step θ[4] | — | θ₄/H |

Extraction is deterministic, canonical (`to_canonical_theta` for θ, `canonicalise(r1_grouped_features)` for the authority),
uses **no new features and no held-out outcomes**.

## 3. Deterministic authority-aware decoder (Stage 2)

`authority_decoder.py` — `decode_from_canonical` (pure) / `decode_intent` (snapshot): canonical θ = `demand·slew/reach`
(object `fr`/`br`/`lr`; internal `nf`; timings·H) → box-clip → `from_canonical_theta` (balance θ[2] sign flips iff swapped).
Object-motion authority (B_coin: forward/brake/lateral reach) and internal-force authority (B_τ null-space: squeeze) stay
**separate**. Diagnostics: authority, per-θ saturation, achievable-peak estimate, per-role residual, weak-link
(push/squeeze/brake/release). **It never falls back to a teacher θ.** Sources are the already-frozen
`identify_Bcoin`/`object_authority`/`reachable`/`admissible_dtau_box`/`contact_internal_authority` + canonical R1 groups.

## 4. Tests + lint (Stage 3)

`test_coin_r3_decoder.py` — **7/7 pass** (6 fast + 1 slow physics): mirror equivariance (incl. balance sign), bounds/slew,
object/internal separation, monotonicity, determinism+provenance, teacher round-trip (dev, K6), intent bounds. **ruff
clean.** 45 canonical/authority/theta/relational regression tests still pass. New functions ≤ CC 14.

## 5. D0 — teacher-intent round-trip (`decoder_d0.json`)

`extract → deterministic decode → budget-8 → K6` on all 4 teacher states:

| state | split | reconstruction ‖dec−teacher‖ | K6 | weak-link |
|---|---|--:|:--:|---|
| s1 | dev | 0.0000 | ✅ | brake |
| s3 | dev | 0.0002 | ✅ | squeeze |
| s4 | held-out | 0.0000 | ✅ | push |
| s7 | held-out | 0.0000 | ✅ | squeeze |

**`D0_TEACHER_INTENT_ROUNDTRIP_PASS` (4/4).** The decoder is an *exact inverse* — the physics factorisation is valid, with
no teacher fallback. Every cradle's *own* intent decodes to a delivering θ, held-out included. This is the strong positive:
the failure that follows is **not** the decoder.

## 6. D1 — development update-0

`intent_predictor.py` — Nadaraya-Watson kernel regression, canonical R1 features (43-D) → 7 intent roles, bandwidth by
**dev-only** leave-one-out. **Dev-only LODO = 0/6 at every bandwidth** {0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0} — the intent
does not transfer even *within* dev. D1 deploy (predictor fit on all 6 dev, evaluated on the panel dev states, which are in
the training set): **s1 ✅, s3 ✅ → dev 2/2, D1 PASS** (via self-weight).

## 7. D2 — one frozen panel (`frozen_panel.json`)

`learned intent → deterministic decode → budget-8 → K6`. Every state used 8 candidates; motion within contract (peak q̇ ≤
3.0); peak RSS 0.25 GB.

| state | split | K6 | dtz start→end (mm) | zone | weak-link | dominant intent error (pred−teacher) |
|---|---|:--:|---|:--:|---|---|
| s1 | dev | ✅ | 76.4 → 17.4 | ✓ | brake | (all ≤ 0.017) |
| s3 | dev | ✅ | 99.9 → 6.7 | ✓ | squeeze | (all ≤ 0.028) |
| s4 | held-out | ❌ | 96.1 → **48.7** | ✗ | squeeze | **peak_velocity +0.21**, squeeze +0.09, release +0.10 |
| s7 | held-out | ❌ | 137.7 → **107.4** | ✗ | brake | **release +0.22**, braking_demand +0.19 |

**dev 2/2, held-out 0/2, total 2/4.** Both held-out cradles `NEVER_REACHED_ZONE`.

### Why held-out fails (measured)

The predictor gives the held-out cradles **dev-like** intents. The decisive witness is **s4's peak_velocity**: the teacher
needs a *gentle* transport (0.263 m/s — s4 rides the motion limit) but the predictor, trained on dev cradles that all
transport at ≈0.45 m/s, predicts **0.47** (+0.21). The over-strong push (plus too much squeeze and a late release) misses
the zone. s7 fails on brake/release timing (release +0.22). The held-out cradles' *intents* are qualitatively outside the
dev manifold, and 6 dev cradles do not teach them — exactly what the 0/6 dev-LODO foretold.

## 8. Honest decision-tree verdict

**Case C — `PHYSICAL_INTENT_FACTORISATION_ALONE_INSUFFICIENT`.** Development is preserved (2/2), D0 passes, provenance and
motion are valid → a *genuine scientific negative*, not an implementation failure (Case D excluded). R3 does **not** improve
over R2's 2/4 / held 0/2 (`improves_over_r2 = False`; Case B excluded).

**What R3 adds to the arc (measured vs inferred).** *Measured:* the decoder is a valid inverse (D0 4/4, own-intent 6/6 on
dev), yet a learned intent predictor generalises no better than raw θ (dev-LODO 0/6, held 0/2). *Inferred:* the bottleneck
is neither the representation (R1/R2), nor the search (R2 basin audit), nor the decoder physics (D0) — it is the
**amortisation itself**: the held-out *decision* (θ **or** intent) is not inferable from 6 dev cradles. Re-parameterising the
decision as a physically-factored intent does not make it more predictable when the held-out intent regime (s4's low peak
velocity) lies outside what the dev cradles demonstrate.

The cumulative result is unusually clean — **R1 (flat), R2 (relational), R3 (physical-intent decoder) all land at exactly
2/4, held-out 0/2**, while every *forward* factorisation (canonical frame, relational graph, authority decoder) is verified
correct in isolation. Plot: `r3_panel.png`.

- **SAC/TD3 authorisation: BLOCKED.** Only Case A (4/4 incl. held-out 2/2) authorises it.
- **Exact next action (not built here; needs a fresh contract):** the open-loop amortisation of a *fixed* decision is
  exhausted across three representations. The next axis is **not another open-loop predictor** but either (a) a
  **closed-loop / basin-aware** policy that measures the cradle's early response and corrects the intent online (the
  decoder makes this cheap — it already maps intent→θ per cradle), or (b) genuine **RL exploration** — which remains gated
  behind a 4/4 update-0 that no open-loop method has reached. The decisive datum for the next contract is the concrete
  witness: s4 needs a *qualitatively* gentler transport (peak velocity 0.26 vs dev 0.45) that no dev cradle teaches.

## 9. Files touched (no CORE.YAML items)

| file | Δ |
|---|---|
| `theta_option/physical_intent.py` | +103 (new) |
| `theta_option/authority_decoder.py` | +92 (new) |
| `theta_option/intent_predictor.py` | +61 (new) |
| `tests/test_coin_r3_decoder.py` | +120 (new) |
| `experiments/coin_theta_rl_benchmark.py` | +227 (`--r3-decoder-d0`, `--r3-update0` + helpers) |
| `reports/2026-07-27-coin-r3-physical-intent-decoder/` | contract_audit / teacher_intents / decoder_d0 / development_update0 / frozen_panel / decoder_parameters / r3_update_zero .json, r3_panel.png |

No §6.5 anti-patterns (modes on the one harness, not v-files; shared `_r3_deploy_one`; no globals). Plan-of-record: the
frozen contract `ce3d6f41`; pdflatex absent — the four-format bundle was not built (recorded in `contract_audit.json`).

## 10. Provenance

Start commit `ce3d6f41` · branch `recovery/coin-r3-physical-intent-decoder`. Predictor: NW kernel regression, canonical R1
43-D features, bandwidth 3.0 (dev-LODO tie at 0/6), 6 dev cradles. Deploy RNG `90000+i·131`; dev-LODO RNG `70000+i`. Env
`.venv` torch 2.12.0, NumPy 2, mujoco; Apple-Silicon CPU. D0 wall 96 s; D1+D2 wall 258 s; peak RSS 0.25 GB (≪ 16 GB).
Working tree: only the documented pre-existing untracked artefacts plus this run's outputs.
