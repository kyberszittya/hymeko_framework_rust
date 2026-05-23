#!/usr/bin/env bash
# Phase-12 — total_correlation_mi (Path I) characterisation.
#
# Path I is the L-way generalisation of Path F (cross_layer_mi) with two
# additional design moves carried over from kl_trajectory + entropy_lyapunov:
#
#   (a) L-way joint Gram (Hadamard, renormalised stepwise) instead of the
#       pairwise sum — captures multi-way redundancy that pairwise misses.
#         TC_2 = Σ_l H_2(K_l) − H_2(K_join_all_layers)
#   (b) KL-of-joint-spectrum-between-steps damps λ (Lyapunov-style):
#         lam_factor = exp(−η · KL(p_{t-1} ‖ p_t)),  clamp [0.1, 10]
#   (c) Variance-momentum on λ via EMA(σ²(p_t)). Three modes are tested
#       to settle the design direction:
#         damp    : var_factor = (1 − inertia)  — transient-conservative
#         amplify : var_factor = inertia        — steady-state-aggressive
#         mix     : stage-aware blend
#                   var_factor = w·inertia + (1−w)·(1−inertia),
#                   w = exp(−η·KL) clipped to [0, 1].
#
# Three head-to-head comparisons:
#   PH12-Q1: spirals 100×50, all three modes vs baseline at λ=0.1.
#   PH12-Q2: spirals λ sweep (5 values) at default mode (mix).
#   PH12-Q3: cross-dataset (circles + 3 MNIST siblings) at default mode.
#   PH12-Q4: β momentum sweep (β∈{0.0, 0.99}) on spirals at mix mode.
#
# Mirror ph11's structure so the headlines slot into the same table.
# Math: docs/total_correlation_path_i.md (to be written from results).

set -u
LOG=/tmp/thesis_iv_views_ph12.log
OUT=data/benchmarks
mkdir -p $OUT

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH12 (total_correlation_mi / Path I): $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG

RUN() {
  local name="$1"; shift
  echo "" | tee -a $LOG
  echo "[$(date +%H:%M:%S)] START: $name" | tee -a $LOG
  echo "  cmd: $*" | tee -a $LOG
  python3 python/benches/thesis_iv_hard/run_benchmark.py "$@" >> $LOG 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE:  $name" | tee -a $LOG \
    || echo "[$(date +%H:%M:%S)] FAIL:  $name" | tee -a $LOG
}

# ---------- Q1: variance-mode head-to-head on spirals -----------------
# baseline shipped once; mode-specific TC arms compared against the
# same baseline rows.
RUN "spirals total_correlation_mi mode=damp (anchor + baseline)" \
  --datasets spirals \
  --arms baseline total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode damp

RUN "spirals total_correlation_mi mode=amplify" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode amplify

RUN "spirals total_correlation_mi mode=mix" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# ---------- Q2: λ sweep on spirals (mode = mix, the principled blend) -
RUN "spirals TC mix λ=0.01" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.01 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "spirals TC mix λ=0.03" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.03 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "spirals TC mix λ=0.3" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.3 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "spirals TC mix λ=1.0" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 1.0 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# ---------- Q3: cross-dataset paired runs (mode = mix) ----------------
RUN "circles TC mix 100×50" \
  --datasets circles --arms baseline total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "mnist_small TC mix 33×15" \
  --datasets mnist_small --arms baseline total_correlation_mi \
  --seeds 33 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "fashion_mnist TC mix 33×15" \
  --datasets fashion_mnist --arms baseline total_correlation_mi \
  --seeds 33 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

RUN "kmnist TC mix 33×15" \
  --datasets kmnist --arms baseline total_correlation_mi \
  --seeds 33 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.9 --tc-variance-mode mix

# ---------- Q4: β momentum sensitivity at mode = mix -----------------
RUN "spirals TC mix β=0.0" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.0 --tc-variance-mode mix

RUN "spirals TC mix β=0.99" \
  --datasets spirals --arms total_correlation_mi \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --lyapunov-eta 5.0 --tc-momentum-beta 0.99 --tc-variance-mode mix

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH12 finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
