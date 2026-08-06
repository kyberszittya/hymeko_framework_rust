# HSiKAN — article skeleton (draft)

**Working title:** *HSiKAN: a parameter-frugal signed-hypergraph KAN for structure-aware, accountable
control — and where its structural prior helps (and where it doesn't).*

**One-sentence thesis.** Replacing a Kolmogorov–Arnold network's per-edge spline explosion with a
**per-channel Catmull-Rom activation over shared signed-hypergraph message passing** yields a KAN-expressive
but GNN-frugal operator; compiled from a single declarative model, it makes structure, state, action and
reward share one auditable source — at parity with an MLP on simple morphologies, with the structural prior
expected to pay off on rich (branched / signed / contact-cycle) ones.

**Honest framing (the spine of the paper).** We report *why it's better, when it's worse, and the
trade-offs* — including the parity/null results — not a cherry-picked SOTA.

**Candidate venues.** Systems/applications framing → CogInfoCom / SMC / a robotics-systems venue (matches the
group's record). ML-architecture framing → needs the two A/Bs below before submission.

---

## 1. Introduction / motivation
- KANs: expressive (learnable univariate edge functions) but **parameters scale with edges** — O(E·G).
- GNNs: parameter-frugal (shared weights) but **fixed activations**.
- Control on a robot needs: structure-awareness (the policy should *see* the body), parameter frugality
  (embedded/edge deployment), and — increasingly — **accountability** (an auditable model, not a black box).
- Gap: a learnable-activation operator that is frugal *and* structure-aware *and* compiled from a declarative,
  auditable source. → HSiKAN.

## 2. HSiKAN architecture
- Signed-hypergraph message passing: `h' = CR(W_self·h + W₊·A⁺h + W₋·A⁻h)`, mean-pooled.
- **Per-channel** Catmull-Rom activation (cardinal spline; control values only, tangents derived → no
  learnable basis/knots; cheap local cubic; fused kernel).
- **Parameter accounting** (the central architecture claim): per-channel CR on shared weights scales with
  *channels*, not *edges*. *(measured: SignedConv 8→16 = 512 params = 3 shared matrices (432) + CR 16×5 (80).)*
- The HyMeKo compilation: `.hymeko` → MJCF + the kinematic hypergraph the operator runs on (Fig. 1).

## 3. The unification (systems contribution)
- One declarative model → **kinematic structure + state domain + action interface + reward**.
- Reward-as-`.hymeko` parity (measured: declarative == procedural, bit-for-bit — the Phase-2 test).
- Why it matters: the steering channel (reward), the perception domain (state) and the body (structure) are
  all *declared and auditable*, not hidden glue. (Fig. 2: the four roles on one hypergraph.)

## 4. Experiments
### 4.1 Parameter efficiency — HSiKAN vs KAN vs MLP  **[A/B #1 — to run, see plan]**
- Match accuracy on 1–2 tasks (a tabular/graph task + a control task); report params at iso-accuracy.
- Expected: HSiKAN ≪ KAN params; HSiKAN ≈/> MLP params at equal accuracy.
### 4.2 When the structural prior helps — serial vs branched morphology  **[A/B #2 — to run, see plan]**
- HSiKAN vs params-matched MLP, same off-policy algorithm (DDPG/SAC — ~250× faster than PPO), on:
  serial chain (reach) **vs** branched/contact-cycle (pick-and-place, two-arm Galambos).
- Hypothesis: parity on serial (already measured: 0.240 vs 0.226 m, indistinguishable, MLP at ½ params);
  HSiKAN advantage on branched/contact morphologies. *Either outcome is publishable.*
### 4.3 Control results (in hand)
- Reach parity (measured). Pick-and-place: scripted expert reliable; PPO learns approach+contact;
  curriculum PPO holds positive return to difficulty 0.94.
### 4.4 CR ablation (in hand)
- CR vs ReLU: **no accuracy gain**, ~1.5–3.8× slower — so CR's justification is *KAN-expressiveness at low
  parameter cost* + spike-like/embedded dynamics, **not** accuracy. Reported honestly.

## 5. Adversarial accountability (distinctive)
- The "evil-environment" generator: one difficulty knob perturbs the *same* structure into adversarial scenes
  (object→reach edge, heavy, small, slippery). Robustness curve = the honest counterpart to a single success.
- *(in hand: scripted-expert place rate 0.8→0.0 easy→evil; learned-policy curve from the curriculum run.)*

## 6. Limitations / trade-offs (do not hide)
- No advantage on serial chains (MLP ties/wins at ½ params).
- CR: no accuracy edge, slower eval.
- The branched-morphology advantage is the paper's **load-bearing hypothesis** — stands or falls on §4.2.
- Sim-only; real-robot transfer not shown.

## 7. Conclusion
- A frugal, structure-aware, signed KAN operator + a declarative-accountability framework; an honest map of
  where the structural prior pays for itself.

---
## Figures / tables
- Fig.1 `.hymeko` → MJCF + hypergraph (one source, three+ outputs).  Fig.2 four roles on one hypergraph.
- Fig.3 param-vs-accuracy (HSiKAN/KAN/MLP).  Fig.4 serial-vs-branched control curves.
- Fig.5 evil-env robustness curves.  Table.1 param accounting; Table.2 the A/B results.

## What is in hand vs to-run
- **In hand (measured):** param accounting; reward-parity; reach A/B (null); CR-vs-ReLU; evil-env (expert);
  pick-place + curriculum.
- **To run (plan b):** A/B #1 (vs KAN), A/B #2 (serial-vs-branched, off-policy). These convert the
  load-bearing claims from "by construction / hypothesis" to "measured."
