#!/usr/bin/env bash
# Phase-16 — Multi-term wrapping on the WINNERS (Path A, Path B).
#
# ph12 lumped multi-way (TC) + multi-term (KL-feedback λ) together;
# Path I's null tells us only that THE COMBINATION fails. ph16 tests
# whether multi-term wrapping helps a working base regulariser:
#
#   `entropy_lyapunov` arm: reg = lam_factor · (lam_a · A + lam_b · B)
#     where  A = (H_norm − H*)²       (Path A flavour)
#            B = H_norm                (Path B flavour)
#            lam_factor = exp(−η · KL_step)  (Lyapunov damping)
#
# By zeroing one of (lam_a, lam_b), we isolate Path A + Lyapunov vs
# Path B + Lyapunov. Compare these against vanilla scalar_entropy_normalized
# (Path B without Lyapunov) at the same λ.
#
# Hypothesis: multi-term *should* help the winners — KL-feedback damps
# λ during transients, lets it grow back when stable. Path B already
# lands +0.624 *** on spirals, +0.232 *** on mnist_small. If multi-term
# is a real win, we should see those numbers improve, especially in
# Path B's weak spots (circles, capsmlp).
#
# 3 datasets × ~2 invocations each = 6 RUN calls. ~3h on RTX 2070S.

set -u
LOG=/tmp/thesis_iv_views_ph16.log
mkdir -p data/benchmarks

echo "==================================" | tee -a $LOG
echo "Thesis IV views PH16 (multi-term on Path A/B): $(date)" | tee -a $LOG
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

# ─── spirals: Path B is a strong positive (+0.624 ***); does Lyapunov help further?
RUN "spirals baseline + Path B + Path B+Lyapunov" \
  --datasets spirals \
  --arms baseline scalar_entropy_normalized entropy_lyapunov \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 0.0 --lam-b 1.0

RUN "spirals Path A+Lyapunov (single arm, paired against above baseline)" \
  --datasets spirals \
  --arms entropy_lyapunov \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 0.0

# ─── circles: Path B's neutralisation point (-0.016 ns); can multi-term turn it positive?
RUN "circles baseline + Path B + Path B+Lyapunov" \
  --datasets circles \
  --arms baseline scalar_entropy_normalized entropy_lyapunov \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 0.0 --lam-b 1.0

RUN "circles Path A+Lyapunov (single arm)" \
  --datasets circles \
  --arms entropy_lyapunov \
  --seeds 100 --epochs 50 --lam 0.1 --reg-every-n 10 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 0.0

# ─── mnist_small: Path B's MNIST anchor (+0.232 ***); does Lyapunov shift it?
RUN "mnist_small baseline + Path B + Path B+Lyapunov" \
  --datasets mnist_small \
  --arms baseline scalar_entropy_normalized entropy_lyapunov \
  --seeds 33 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 0.0 --lam-b 1.0

RUN "mnist_small Path A+Lyapunov (single arm)" \
  --datasets mnist_small \
  --arms entropy_lyapunov \
  --seeds 33 --epochs 15 --lam 0.01 --reg-every-n 50 --view dataflow \
  --target-entropy 0.5 --lyapunov-eta 5.0 --lam-a 1.0 --lam-b 0.0

echo "" | tee -a $LOG
echo "==================================" | tee -a $LOG
echo "PH16 finished: $(date)" | tee -a $LOG
echo "==================================" | tee -a $LOG
