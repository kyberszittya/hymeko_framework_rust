# R11.6D — Conclusion: the r9 far-angle recovery is unreachable by transport-summary retrieval

**Date:** 2026-08-06
**Worktree:** `hymeko_coin_r9_wt` · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Base SHA:** `e7d69f2a`
**Result:** `R11_6D_HANDOFF_CONDITIONING_NO_GAIN` — three independent selection methods all cap at the descriptor-nearest
baseline (dev 0.571); the r9-delivering θ is a handoff-specific outlier that **inverts** the θ-signature ordering.

---

## The escalation ladder (all on the fixed 2244-cell train transfer matrix, test sealed)

| method | dev top-1 K6 | c3 far-angle | vs baseline 0.571 |
|---|---|---|---|
| per-θ median score | 0.429 (overfit) | 0/3 | worse |
| per-θ reach score (p90) | **0.571** | 1/3 (r7_a+45) | ties |
| best-in-grid, any weights | **0.571** | — | ties (ceiling) |
| **handoff-conditioned ridge** (θ-profile × d_req × interactions) | **0.571** | 1/3 (r7_a+45) | ties |

Every method recovers the `r7_a+45` blend artifact and reaches exactly the baseline — **none recovers the two r9
far-angle handoffs**, and none exceeds descriptor-nearest.

## Why — the decisive mechanism (the θ-signature ordering is inverted from the truth)

From the r9 handoffs, the θ that *actually* delivers is a descriptor-far, low-typical-transport θ whose reach from r9
is an **out-of-profile outlier**:

| r9 handoff (d_req) | delivering θ | its transport **from r9** | its train-avg signature | the θ the models pick |
|---|---|---|---|---|
| `r9_a-30` (100mm) | `c0_3` → K6 12.77mm | **87.2mm** | typical 62, **p90 84** | `r7_a+15` (p90 **86**, transports 71.7mm → FAIL 28.5mm) |
| `r9_a-15` (98mm) | `c1_+0.01_+0.02` → K6 11.0mm | **87.2mm** | typical 71, **p90 85** | `r7_a+15` (p90 **86**, transports 65.9mm → FAIL 32.8mm) |

- The delivering θ's r9 transport (87mm) **exceeds its own p90** (84–85mm) — the reach it shows from r9 is above its
  entire train profile.
- The generic θ the models pick (`r7_a+15`) has a **higher** p90 signature (86mm) yet transports **less** from r9
  (66–72mm). **By any per-θ transport summary, the wrong θ looks better.**

So no selection built on per-θ transport summaries — median, p90, or a linear model over them + d_req + bearing — can
prefer `c0_3` over `r7_a+15` for r9: the summaries rank them backwards. Knowing that `c0_3` reaches 87mm *from the r9
handoff specifically* requires the (θ × r9-handoff) physics, which is not in any train-averaged feature and cannot be
learned from 44 train handoffs when r9 is the held-out far case (no train handoff is near enough to teach the
interaction). This is a genuine limit of transport-summary retrieval, not a tuning miss.

## Honest close of R11.6D

- **R11.6C stands: the system delivers the coin autonomously from exact-zero** (48 certs, no teacher/CEM/oracle).
- **R11.6D localized the residual causally** to target geometry (canonicalizer refuted), proved a delivering θ exists in
  the bank for r9, and then proved — across per-θ scoring, reach-aware scoring, and a handoff-conditioned ridge — that
  **the retrieval ceiling is the descriptor-nearest baseline (0.571 dev)**. The r9 recovery is unreachable by
  transport-summary retrieval because the winning θ's reach is a handoff-specific outlier that inverts the signature.
- **Deploy retrieval stays descriptor-nearest** (R11.6C's frozen policy, unchanged). **No test unseal** — the gate did
  not clear on any method. Dev is a spent development panel; the sealed test is untouched.
- **The only remaining lever for r9** is the one the user deferred: **targeted densification at the far radius with a
  NEW sealed test panel** (adding train handoffs near r9 so the interaction is covered). That is a separate, explicitly
  gated decision, not part of this arc. Otherwise r9 far-angle is a documented, localized coverage limit and R11.6D
  closes; the ladder continues at R11.7 (other objects).

---

## Files / tests / provenance

| File | Δ |
|---|---|
| `hymeko_rl/coin_delivery/transport_predictor.py` | ridge (θ,handoff)→dtz predictor, features, top-1 select (new) |
| `hymeko_rl/experiments/r11_6d_conditioned.py` | LOSO λ-calibration + dev eval + coefficients (new) |
| `hymeko_rl/coin_delivery/transport_retrieval.py` | +`transport_p90`, reach score mode, re-examine helpers |
| `hymeko_rl/experiments/r11_6d_transport_retrieval.py` | +reach mode, re-examine (median vs reach + overfit scan) |
| `hymeko_rl/tests/test_r11_6d_transport.py`, `test_r11_6d_conditioned.py` | 8 + 5 tests |
| `reports/2026-08-06-r11-6d-{retrieval,conditioned}/*.json` | matrix eval, re-examine, conditioned result |

- **CORE.YAML:** none. **Deps:** none (ridge is closed-form numpy; no sklearn). No canonicalizer, no blending, no runtime
  oracle/CEM/teacher. Reach/capture/descriptor/rollout/frozen-table read-only.
- **Env:** framework `.venv`, torch 2.12.0, macOS, CPU. All Phase-4/4.1 selection is pure compute on the existing matrix
  (no new rollouts). 13 tests, ruff/radon clean (all A/B). `8f2d796f` preserved; sealed test untouched.

## Boundary

R11.6D closes: composition PASS + causal audit + a rigorous negative on retrieval-side r9 recovery (ceiling = baseline,
mechanism = signature inversion). Next (gated): targeted densification at the far radius with a new sealed test, or R11.7.
