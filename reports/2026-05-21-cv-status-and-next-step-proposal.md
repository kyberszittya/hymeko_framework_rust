# Computer-vision status + next-step proposal — 2026-05-21

## TL;DR (one-paragraph status)

HyMeYOLO's VOC2007 ceiling is **0.053 mAP_50** (Stage H, 1-class) /
**0.013** (Stage D-3b, 20-class with gate-aware nodelet head); visit
gate is 0.20.  The day-18 overview pinned the bottleneck to the K+1
softmax over-provisioning every query, *not* the backbone, *not* the
matcher, *not* pretraining.  Stage D-3-bis (loss-balance tightening on
the nodelet head) is the queued fix.  Between now and the next
architectural review, the highest-EV path is: **(a) run Stage D-3-bis
to settle the loss-balance question, then (b) port the *outer-HSIKAN
residual* lever from link-prediction to the HymeYOLO head only if
D-3-bis still misses 0.20**.  Outer-HSIKAN-as-vision-head is
speculation, not measurement — keep it gated behind the cheaper recipe
fix.

## 1. Where the CV stack is, right now

### 1.1 The ceiling

| Date | Stage | Head | Classes | mAP_50 | Gap to visit gate |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2026-05-17 | D-2d | Hungarian K+1 (softmax) | 20 | 0.0077 | 26× |
| 2026-05-18 | H | Hungarian 1+1 (softmax) | 1 | **0.053** | 4× |
| 2026-05-18 | D-3b | Nodelet (objectness gate) | 20 | 0.0127 | 16× |

Stage H said the multi-class signal-to-noise collapse is the *primary*
bottleneck (1-class → 20-class costs ~7×).  Stage D-3b said the
architectural fix (explicit per-query objectness gate, replacing the K+1
softmax) *works* but the loss balance hasn't been tuned for the new
head.

### 1.2 What's queued, what's not

| Plan dir | Status |
| --- | --- |
| `docs/plans/2026-05-18-hymeyolo-stage-d2-head-bottleneck/` | shipped (Stage D-3 / D-3b results in 2026-05-18 report) |
| `docs/plans/2026-05-18-hymeyolo-stage-d1-pretrain/` | shipped (D-1 pretrain probe) |
| Stage D-3-bis (loss-balance tightening) | **planned, not started** |
| Outer-HSIKAN port to HymeYOLO head | **idea only** — no plan, no smoke |

### 1.3 The unaddressed thread from this session

The most productive lever on the link-prediction side this week is
**outer-HSIKAN with highway-gated residual composition**:

```
x_embed = (1 - g) * base_embed + g * HSIKAN(base_embed)
                                  ↑
                            g init = σ(-3) ≈ 0.05
```

5-seed Bitcoin Alpha paired-Δ vs plain Gömb: **+0.0066 AUC, σ_d 5.68,
5/5 wins** (see `reports/2026-05-21-outer-hsikan-gomb-residual-WIN.md`).
Lever generalises to Bitcoin OTC (+0.0045, 1.73σ).

The temptation is to immediately port outer-HSIKAN to HymeYOLO's
classification head.  Resist the temptation *until D-3-bis runs* —
because the bottleneck in the CV stack is most likely the loss recipe,
not head capacity, and outer-HSIKAN at the head adds parameters where
the gradient signal is already saturated (matched-cls accuracy 0.875).

## 2. Why outer-HSIKAN might transfer to HymeYOLO — and might not

### 2.1 The optimistic story

HymeYOLO's nodelet head produces per-query `(box, cls_logits, gate)`
triples.  The cls and gate predictions are **mutually entangled**: a
high-cls + low-gate query should be suppressed; a high-cls + high-gate
query should rank high; a low-cls + high-gate query is ambiguous.  The
current head computes these from a shared trunk feature via three
parallel linear maps.  An outer-HSIKAN, if treated as a *gating module
over a signed-cycle graph of query interactions*, could make the
high-recall queries cooperate (suppress duplicate detections) and the
low-recall queries compete (preserve diversity).

That's the architectural fantasy.  It is structurally similar to the
problem outer-HSIKAN solved on Bitcoin: featurising the *interaction
graph* over training examples.

### 2.2 The pessimistic story

Outer-HSIKAN on Bitcoin trained on **signed-cycle data with a built-in
sign-product axiom** (Cartwright-Harary balance).  HymeYOLO queries
have no analogous sign structure — the natural "edge" between two
queries is just IoU overlap or feature similarity, neither of which is
signed.  Without the signed-cycle inductive bias, outer-HSIKAN is just
an expensive MLP — and the 528k-param two-Gömb bridge result this
afternoon (paired-positive vs plain Gömb but underperforms the simpler
single-cortex variant) is a cautionary tale: *more capacity, on a
dataset where the existing capacity isn't saturated, hurts*.

### 2.3 The honest reading

Outer-HSIKAN's win on Bitcoin came from **highway-gated residual
composition** (init g ≈ 0.05).  That lever is *transferable* to
HymeYOLO independent of the signed-cycle story: any module added to the
nodelet head's trunk feature should be added as `(1 - g) · base + g ·
new` rather than substitutively.  This is the lever to port, not the
HSIKAN block itself.

## 3. The proposed roadmap

### Stage 1 — settle D-3-bis (cheap, blocking) ≤ 1 day GPU

The loss balance tightening planned in the day-18 overview:

- Re-weight the gate-fraction loss: `λ_gate * BCE(gate, in_image)` with
  λ_gate ∈ {0.5, 1.0, 2.0, 5.0}.
- Re-weight the no-object class loss: `λ_no_obj * CE(cls, no_obj_idx)`
  with λ_no_obj ∈ {0.1, 0.5, 1.0, 2.0}.
- 4 × 4 grid × 1 seed VOC2007 ≈ 16 jobs × 90 min = 24 GPU-hours.

**Falsification criterion:** if the best D-3-bis cell clears 0.10 mAP_50,
the bottleneck *is* loss balance and outer-HSIKAN is unnecessary.  If
the best cell stays below 0.05, escalate to Stage 2.

### Stage 2 — port the *residual* lever (not the HSIKAN block) ≤ 0.5 day

Independent of outer-HSIKAN itself: change every additive composition
in HymeYOLO's head from substitutive to highway-gated:

```python
# before
trunk_feat = trunk_block(x)
cls_feat = cls_head(trunk_feat)

# after
trunk_feat = (1 - g_trunk) * x + g_trunk * trunk_block(x)
cls_feat   = (1 - g_cls)   * trunk_feat + g_cls * cls_head(trunk_feat)
# g_* are learnable per-channel scalars init at sigmoid(-3) ≈ 0.05
```

Cost: 5 lines of code, ~0 params, and 5-seed VOC re-run.  This is the
cheapest possible port of the link-prediction win.

**Falsification:** if highway-gating doesn't move the needle on VOC at
matched D-3-bis recipe (5 seeds, paired), the architectural-lever
hypothesis is wrong and we look elsewhere.

### Stage 3 — only if Stage 1+2 still miss 0.20: outer-HSIKAN block ≤ 3 days

Genuinely speculative.  Plan dir + 4-format-plan first, no early
implementation.  At this point the falsifications from Stage 1 + 2 will
have narrowed the search space enough that the outer-HSIKAN block can
be specified concretely (which query graph, which signs, which
predicate).

## 4. Risk audit (CLAUDE.md §2 risk anticipation)

- **Stage 1 cost is bounded**, but the GPU is also queued for the
  Wiki-OOM follow-up and the outer-HSIKAN MSG/ABB grid.  Sequence:
  Wiki first (lowest priority), then D-3-bis, then MSG/ABB.  D-3-bis
  uses VOC dataloader which doesn't compete with cycle-enum-bound
  jobs.
- **Stage 2's "highway-gating-as-universal-lever" claim is broader
  than the Bitcoin evidence supports.**  Mitigate: 5-seed paired vs.
  the non-gated baseline; do not promote to paper text until n ≥ 5.
- **Stage 3 is unbounded in design space.**  Mitigate: do not enter
  Stage 3 without a 4-format plan that names the query graph
  construction, the sign predicate, and the integration point.

## 5. What I'd ask the user before promoting Stage 2 or 3

1. **Is the visit gate genuinely 0.20 mAP_50, or is the demo
   negotiable at 0.10?**  At 0.10, Stage 1 alone is plausibly
   sufficient.  At 0.20, Stage 2 is a hard requirement and Stage 3 a
   likely one.
2. **Is the family paper's CV chapter targeted for the *current*
   manuscript (T-SMC-S journal v1) or the *next-Niitsuma-visit*
   companion paper?**  T-SMC-S can ship without HymeYOLO's natural-image
   number; the companion paper cannot.
3. **Has the SignedKAN-in-HymeYOLO-FPN thread (Stage C-5seed,
   2026-05-17) been read?**  If FPN-SignedKAN improved over plain FPN,
   that's evidence the signed-cycle prior *does* transfer to image
   features and Stage 3 deserves a real plan rather than a deferral.

## 6. Recommended single action (sleep-safe)

**Queue Stage D-3-bis only.**  Stages 2 and 3 wait for the user's
morning read of this proposal.  D-3-bis is fully specified in the
day-18 overview; the queue script need only sweep λ_gate × λ_no_obj.

Estimated wall time on the RTX 2070 SUPER 8 GiB: 24 GPU-hours, fits in
overnight if it starts now.

**Not queued in this report** — the user is asleep and the queueing
decision should wait for explicit go-ahead, per CLAUDE.md §11 "When in
doubt: silence is preferable to a wrong action."  Per the same clause,
no Stage 2 / Stage 3 plan-dir created either; both wait on the user.

## 7. Files referenced

- `reports/2026-05-18-vision-bottleneck-day-overview.md` — the day-18
  dossier, source of the D-3-bis plan.
- `reports/2026-05-21-outer-hsikan-gomb-residual-WIN.md` — Bitcoin
  Alpha outer-HSIKAN result.
- `reports/2026-05-21-gomb-bridge-gomb.md` — bridge result (cautionary
  tale on capacity without inductive-bias justification).
- `reports/2026-05-17-hymeyolo-stage-c-5seed.md` — SignedKAN-in-FPN
  5-seed (question 5.3).
- `docs/plans/2026-05-18-hymeyolo-stage-d2-head-bottleneck/` — Stage
  D-3 plan.

## 8. Acceptance check

- [x] Status table accurate against day-18 overview and 2026-05-21
      WIN report.
- [x] Proposal stages each have a falsification criterion.
- [x] Risk audit per §2 in CLAUDE.md.
- [x] **No new experiments queued** — the user explicitly authorized
      autonomous work on HTL, Unity, CV survey; long GPU runs require
      explicit go-ahead.
- [x] Report on disk.
