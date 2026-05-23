r"""Bayesian-optimisation wrapper around `run_final_cell` for HSiKAN
hyperparameter search.  Each Optuna trial is one training run; the
controller proposes configurations via the GP-EI sampler and picks
up the AUC from `run_final_cell`'s stdout JSON line.

Search space (per trial):
- per-slot booleans for $\mathcal{K} \subseteq \{c2,c3,c4,c5,w2,w3,w4,w5\}$
  (rejected as `TrialPruned` if all-empty)
- hidden $\in \{4, 8, 16\}$
- attention kind $\in \{\text{none}, \text{dot}, \text{quaternion}\}$ on
  large GPUs; **small GPUs** (total VRAM below 12 GiB by default) search
  **none** only — see ``_attention_kind_candidates()`` / env
  ``HSIKAN_OPTUNA_ATTENTION_*``.
- Highway gate $\in \{0, 1\}$ when attention is on; max ∈ [0.1, 1.0]
- $\lambda_\alpha, \lambda_{\rm attn}$ as log-uniform conditional on use
- max_k caps shared across $k\in\{2,3,4\}$, $\in \{50K, 100K, 200K\}$
- (n_epochs fixed at 80 unless overridden)

Storage: SQLite for resumability.  Use `--study-name X --storage
sqlite:///hsikan_bo.db` to persist across launches.

``main()`` holds the repo-wide **CUDA job flock** (``cuda_job_lock``) for the
whole study so Optuna does not overlap the SOTA gate or a second Optuna
process on the same GPU host (disable with ``HYMEKO_CUDA_DISABLE_JOB_LOCK=1``).

To block until every in-flight ``run_optuna_search`` has exited (then e.g.
start a fresh study that picks up the latest code on disk), use
``signedkan_wip/experiments/wait_until_no_optuna_search.sh``.

Example:
    python -m signedkan_wip.experiments.runs.run_optuna_search \\
        --dataset slashdot --n-trials 30 \\
        --study-name slashdot_bo --storage sqlite:///bo.db
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import optuna


SLOT_TAGS = ("c2", "c3", "c4", "c5", "w2", "w3", "w4", "w5")


def _attention_kind_candidates() -> list[str]:
    """Return Optuna ``attention_kind`` search space.

    Dot / quaternion attention materialises large score tensors on ``M_e``;
    on **small** GPUs this routinely OOMs before any AUC is emitted.  By
    default we **drop** ``dot`` and ``quaternion`` when the **primary CUDA
    device** reports **total** VRAM strictly below **12 GiB** (configurable).

    Override env (first match wins):

    * ``HSIKAN_OPTUNA_ATTENTION_KINDS`` — comma list, e.g. ``none,dot``.
    * ``HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION=1`` — force ``none`` only.
    * ``HSIKAN_OPTUNA_ATTENTION_VRAM_GIB_MIN`` — float threshold in GiB;
      if ``total_memory < min * 1024**3``, use ``none`` only (ignored when
      ``ATTENTION_KINDS`` or ``SKIP`` is set).
    """
    raw = os.environ.get("HSIKAN_OPTUNA_ATTENTION_KINDS", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    if os.environ.get("HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return ["none"]
    try:
        import torch

        if torch.cuda.is_available():
            total = int(torch.cuda.get_device_properties(0).total_memory)
            gib = float(os.environ.get("HSIKAN_OPTUNA_ATTENTION_VRAM_GIB_MIN", "12"))
            limit = int(gib * (1024**3))
            if total < limit:
                return ["none"]
    except Exception:
        pass
    return ["none", "dot", "quaternion"]


def _parse_auc(stdout: str) -> float | None:
    """Parse the last JSON line emitted by `run_final_cell` and return
    the AUC field; None if no valid JSON line found."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{"dataset"'):
            try:
                return float(json.loads(line).get("auc", float("nan")))
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def objective(trial: optuna.Trial, dataset: str, seed: int = 0,
              n_epochs: int = 80, run_timeout_s: int = 7200) -> float:
    # 1. Mixed-tuples set: bool per slot.
    used = [trial.suggest_categorical(t, [True, False]) for t in SLOT_TAGS]
    slots = [t for t, u in zip(SLOT_TAGS, used) if u]
    if not slots:
        raise optuna.TrialPruned()
    mix = ",".join(slots)

    # 2. Architectural categorical knobs.
    hidden = trial.suggest_categorical("hidden", [4, 8, 16])
    attn_choices = _attention_kind_candidates()
    attn_kind = trial.suggest_categorical("attention_kind", attn_choices)
    if attn_kind != "none":
        use_highway = trial.suggest_categorical("highway", [True, False])
        highway_max = (
            trial.suggest_float("highway_max", 0.1, 1.0)
            if use_highway else 1.0
        )
    else:
        use_highway = False
        highway_max = 1.0

    # 3. Aux entropy λ's, conditional on use.
    use_alpha_ent = trial.suggest_categorical(
        "use_alpha_entropy", [True, False]
    )
    lam_alpha = (
        trial.suggest_float("alpha_entropy_lambda", 1e-5, 1e-1, log=True)
        if use_alpha_ent else 0.0
    )
    if attn_kind != "none":
        use_attn_ent = trial.suggest_categorical(
            "use_attn_entropy", [True, False]
        )
        lam_attn = (
            trial.suggest_float("attn_entropy_lambda", 1e-5, 1e-1, log=True)
            if use_attn_ent else 0.0
        )
    else:
        lam_attn = 0.0

    # 4. Cap (shared k=2/3/4).
    cap = trial.suggest_categorical("max_k_cap", [50000, 100000, 200000])

    # Build env and command.
    env = os.environ.copy()
    env["HSIKAN_MIXED_TUPLES"] = mix
    env["HSIKAN_CYCLE_BATCH"] = "2000"
    env["HSIKAN_MAX_K3"] = str(cap)
    env["HSIKAN_MAX_K2"] = str(cap)
    if attn_kind != "none":
        env["HSIKAN_ATTENTION_M_E"] = attn_kind
        if use_highway:
            env["HSIKAN_ATTENTION_HIGHWAY"] = "1"
            env["HSIKAN_ATTENTION_HIGHWAY_MAX"] = f"{highway_max}"
    if lam_alpha > 0:
        env["HSIKAN_ALPHA_ENTROPY_LAMBDA"] = f"{lam_alpha}"
    if lam_attn > 0:
        env["HSIKAN_ATTN_ENTROPY_LAMBDA"] = f"{lam_attn}"

    cmd = [
        sys.executable, "-m", "signedkan_wip.experiments.runs.run_final_cell",
        "--dataset", dataset, "--hidden", str(hidden),
        "--n-epochs", str(n_epochs), "--max-k4", str(cap),
        "--seed", str(seed),
    ]
    print(f"[trial {trial.number}] mix={mix} h={hidden} attn={attn_kind} "
          f"hw={use_highway}/{highway_max:.2f} "
          f"lam_a={lam_alpha:.1e} lam_attn={lam_attn:.1e} cap={cap}",
          flush=True)
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=run_timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"[trial {trial.number}] TIMEOUT after {run_timeout_s}s",
              flush=True)
        raise optuna.TrialPruned()

    auc = _parse_auc(proc.stdout)
    if auc is None or auc != auc:  # NaN check
        # Print stderr tail for diagnosis.
        tail = "\n".join(proc.stderr.splitlines()[-5:])
        print(f"[trial {trial.number}] FAIL no auc; stderr tail:\n{tail}",
              flush=True)
        raise optuna.TrialPruned()
    print(f"[trial {trial.number}] AUC={auc:.4f}", flush=True)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["bitcoin_alpha", "bitcoin_otc", "slashdot",
                              "epinions", "sbm_n200", "sbm_n400"])
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0,
                    help="Single training seed per trial (multi-seed is "
                          "the validation step, not the search step)")
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--study-name", required=True)
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:///bo.db")
    ap.add_argument("--run-timeout-s", type=int, default=7200)
    ap.add_argument("--sampler", default="tpe",
                    choices=["tpe", "gp", "random"])
    args = ap.parse_args()

    from signedkan_wip.src.benchmarks.cuda_job_lock import cuda_job_lock

    with cuda_job_lock():
        if args.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif args.sampler == "gp":
            sampler = optuna.samplers.GPSampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)

        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            sampler=sampler,
        )
        study.optimize(
            lambda t: objective(
                t, args.dataset, seed=args.seed, n_epochs=args.n_epochs,
                run_timeout_s=args.run_timeout_s,
            ),
            n_trials=args.n_trials,
        )

        print()
        print(f"Best trial #{study.best_trial.number}: AUC={study.best_value:.4f}")
        print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
