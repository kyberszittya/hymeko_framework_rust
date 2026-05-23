# Optimising HSiKAN structure via RL / active learning

Now that `hymeko_train_walker.py` lets a single `.hymeko` config drive a full
training cell and return metrics, the .hymeko file IS the action a controller
emits. Two natural directions: reinforcement learning (RL) and active learning
(AL). Both treat the .hymeko description as the search space.

## Action space (the same for both)

A single point in the search space is a tuple of HSiKAN structural knobs
expressible in HyMeKo:

| dimension | values | source field |
|---|---|---|
| `arities` | subsets of {2, 3, 4, 5} | arch.hymeko `signedkan_layer { arity }` |
| `hidden` | {16, 32, 64, 128} | arch.hymeko `hidden` |
| `n_layers` | {1, 2, 3} | arch.hymeko (count of `signedkan_layer`) |
| `spline_kind` | {bspline, catmull_rom, kochanek_bartels, cr_kb, kb_cr, ...} | arch.hymeko `spline_kind` |
| `kb_preset` | {smooth, tense, cusp, skew, sharp, flat} | env (HSIKAN_KB_PRESET) |
| `topk_mode` | {global, per_vertex} | training.hymeko `mode` |
| `m_per_vertex` | {16, 32, 64, 128} | training.hymeko `m_per_vertex` |
| `pruner` | {none, balance, davis, unbalanced, frustration} | training.hymeko `pruner` |
| `scorer` | {fraction_negative, mi, …} | training.hymeko `scorer` |
| `entropy_lambda` | {0, 0.005, 0.01, 0.02} | training.hymeko `entropy_lambda` |
| `attention` | {none, dot, quaternion} | env (HSIKAN_ATTENTION_M_E) |
| `direct_msg` | {0, 1} | env (HSIKAN_DIRECT_MESSAGING) |

That's roughly 4 × 4 × 3 × 7 × 6 × 2 × 4 × 5 × 2 × 4 × 3 × 2 ≈ 4M cells. Far
too many to enumerate; this is what makes a controller worthwhile.

## Reward signal (the same for both)

Per cell: validation AUC minus a cost term. Two natural cost terms:

- **Wall time** (penalises huge m_per_vertex / hidden): R = AUC − λ · (train_time / T_ref)
- **Memory** (penalises configs that may OOM): R = AUC − λ · (peak_gpu_gb / 8)

8 GB is the 2070 SUPER limit; OOM should be a hard penalty (R = −1) so the
controller learns the boundary, not just the middle of the feasible region.

## RL formulation (REINFORCE on a config controller)

Smallest viable controller:

- **Policy**: per-dimension softmax `π(a_i | θ_i)` — independent across knobs
  (mean-field). Independent is a strong assumption but loops fast.
- **Sampling**: emit one cell per step, run via the walker, observe reward.
- **Update**: `θ ← θ + α (R − b) ∇log π(a | θ)` — vanilla REINFORCE with a
  running-mean baseline `b`.

A few practical concessions:
- 30 epochs × ~30 s/cell = ~15 min/cell on bitcoin_alpha. Even 100 controller
  steps = 25 hours. Budget gates this; needs a smaller proxy training run
  (≤5 epochs) to evaluate cells faster, then a final 30-epoch run on the top
  candidate.
- Bandit-style alternative: treat each knob as an independent ε-greedy bandit.
  Even simpler, often as good for ~100-step budgets.

**RL question worth thinking about**: is "structure" continuously
parameterisable, or fundamentally discrete? `hidden` is continuous in spirit
but discrete in practice (we round to 2^k). `arities` is discrete (subset of
{2,3,4,5}). `pruner` is categorical. So the search is mostly combinatorial,
which favours bandits over policy-gradient.

## Active-learning formulation (Bayesian optimisation over configs)

If the goal is **explore the response surface**, not commit to a controller:

- **Surrogate**: a Gaussian process (or random-forest) over the config space,
  predicting AUC. Categorical knobs use one-hot; ordinal knobs (m, hidden) use
  log-scale integers.
- **Acquisition**: expected improvement (EI) or upper confidence bound (UCB)
  to pick the next config.
- **Initial design**: 16 random cells to seed the GP, then 50–100 EI-driven
  cells.

Compared to RL: AL is more sample-efficient when cells are expensive. We've
seen both cells of HSiKAN take 2 min (BA, m=16) and 15 min (Slashdot, m=128).
At ~100-step budgets, AL wins.

## Implementation sketch (whichever direction we pick)

The walker already gives us "one cell = one .hymeko = one reward." We need:

1. **Config emitter**: serialise an action vector → `arch.hymeko` + `training.hymeko`
   text via Python f-strings (NOT a HyMeKo template, those are for parsing).
2. **Walker call**: shell out `python -m signedkan_wip.src.hymeko_train_walker
   --arch /tmp/arch_proposal.hymeko --training /tmp/train_proposal.hymeko
   --dataset bitcoin_alpha --seed 0` and parse the JSON.
3. **Controller**: REINFORCE / GP / bandit, mutating action distribution per
   reward.
4. **Logbook**: JSONL of (action, reward, metrics) so we can replay and
   visualise the response surface.

## Open questions before starting

- **Per-dataset or cross-dataset?** Optimising for bitcoin_alpha alone is
  fast but probably overfits to that dataset's quirks. Cross-dataset reward
  (mean AUC over BA + OTC + Slashdot) is more robust but ~3× slower per cell.
- **Proxy fidelity**: how many epochs are enough to predict full-30 AUC?
  Quick experiment: correlation(AUC@5, AUC@30) across 50 random cells.
- **Categorical or continuous αₖ**: the αₖ mixer learns continuously inside
  the model. Should the controller also propose αₖ priors, or let the model
  do it? My instinct: let the model learn αₖ, controller picks structure.

## Recommendation

Start with **bandit + 5-epoch proxy on bitcoin_alpha**, ~100 cells (~3 hrs).
Validate the top-3 cells at 30 epochs. If the bandit picks something
surprising, that's the headline; if it picks the known-good config (m=128,
balance, k=4+5), that's a confirmation experiment but less novel.

Active-learning (GP + EI) is the principled second step once we know the
proxy fidelity is acceptable. RL with cross-dataset reward is the final
form but needs the most compute.
