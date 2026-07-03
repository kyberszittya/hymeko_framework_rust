# Cayley-Rotor Embeddings — distribution package (2026-06-17)

An inductive, leakage-free, parameter-light node embedding for signed-graph
learning: a Clifford/quaternion **rotor** optimized by the **Cayley map**
(`q = [1;b]/‖[1;b]‖`), so plain SGD is optimization on the rotor manifold with no
re-orthogonalization. Used as an embedding, a rotor turns one learned reference
vector into a point on a sphere; computed from structural features (`b = Wx`) it
needs no per-node lookup table — which is both the parameter saving and the
removal of the leakage channel.

## What is verified (measured, honest)

- **Parameter efficiency (5 seeds × 5 datasets, strict leakage protocol).** As the
  node feature of a signed GNN, the inductive rotor matches or beats DADSGNN/SGCN
  at **9–270× fewer parameters** (count *constant* at ~16k vs the table growing
  with the graph), **1.4× faster inference** (measured), all **honest under
  sign-shuffle**. It does **not** beat the attention model SiGAT (trails ~0.02–0.03)
  — so the claim is *Pareto-efficiency*, not "most accurate". See
  `reports/rotor_*_20260616.jsonl`, the article §5, and the figures.
- **What was ruled out (also honest):** cycle features as input = redundant with
  message-passing; jump-propagated cycles = a wash + a leak on one dataset; k=4
  cycles = dilution. The investigation is in the article's "what we ruled out".

## What is a strong but UNVERIFIED signal (do not cite yet)

- Inside **HSiKAN** (bilinear endpoint head, deduped true-held-out split, gate-
  passing), the rotor beat the transductive table by **+0.20 AUROC** at equal
  params on bitcoin_otc, seed 0. This is a single-seed signal under
  multi-seed × multi-dataset verification at time of packaging. Treat as
  hypothesis until the grid (`reports/hsikan_rotor_verify_20260617.jsonl`) confirms.

## Contents

- `hymeko_neuro/graph/embeddings/cayley_rotor.py` — the primitive (+ tests).
- `hymeko_neuro/baselines/cayley_rotor_baseline.py` — signed-link strategies
  (rotor, rotor+cycles, jumped cycles, SiGAT-with-rotor) for the audit harness.
- `hymeko_neuro/experiments/runs/run_hsikan_rotor.py` — HSiKAN injection driver
  (`--head bilinear --dedup`), with the leakage shuffle-gate.
- `docs/articles/cayley-rotor-embeddings/article.{tex,pdf}` — the write-up.
- `docs/plans/2026-06-1{6,7}-*` — plans.
- `reports/*.jsonl` — all raw multi-seed numbers.
- `docs/seminar/figures/rotor_*.png` + `make_rotor_param_figure.py` — figures.

## Reproduce

```
# signed-link audit (strict + shuffle gate), any registered model:
python -m hymeko_neuro.experiments.runs.run_baseline_audit \
    --model cayley_rotor --dataset bitcoin_otc --seed 0 [--shuffle-train-signs]
# HSiKAN with the rotor, honest held-out protocol:
python -m hymeko_neuro.experiments.runs.run_hsikan_rotor \
    --dataset bitcoin_otc --embed rotor --head bilinear --dedup --seed 0
```

Every headline here passed a sign-shuffle leakage gate (shuffle → chance); numbers
that did not are flagged as signals, not results.
