# Reproducibility Package — skeleton for the signed-link paper

**Status**: DRAFT for coworker review. NComms requires this package
to be assemble-able by an editor in < 30 min of effort; we should
target < 10 min. The audit code is the central artifact — without it
the title claim has no path to verification.

NComms reporting requirements (verified 2026-06):
- **Data Availability statement** (mandatory)
- **Code Availability statement** (mandatory)
- **Reporting Summary** (NComms-specific checklist; ~3 pages)
- **Competing Interests** (mandatory)
- Author contribution roles (CRediT taxonomy)

---

## Directory layout (anchor for the reproducibility container)

```
signed_link_audit_package/
├── README.md                          # ENTRY POINT
├── LICENSE                            # [TODO/CW] which OSI license?
├── CITATION.cff                       # Citation metadata
├── requirements.txt                   # Pinned Python deps
├── pyproject.toml                     # Package metadata
├── Dockerfile                         # The container recipe
├── reproduce.sh                       # One-command reproduce script
│
├── audit/                             # ★ THE CENTRAL ARTIFACT ★
│   ├── __init__.py
│   ├── label_shuffle.py               # The audit itself
│   ├── strict_protocol.py             # Strict-protocol split builder
│   ├── leaky_protocol.py              # Reference leaky-protocol impl
│   └── tests/
│       ├── test_label_shuffle.py      # Audit correctness tests
│       └── test_strict_protocol.py
│
├── protocols/                         # Protocol definitions
│   ├── README.md                      # Formal definitions
│   ├── strict.py                      # Training-edge-only feature builder
│   └── leaky.py                       # Reference impl of standard protocol
│
├── splits/                            # Frozen split files (hashed)
│   ├── bitcoin_alpha/
│   │   ├── train.npz
│   │   ├── test.npz
│   │   └── SHA256
│   ├── bitcoin_otc/
│   ├── slashdot/
│   ├── epinions/
│   └── reddit_hyperlinks/
│
├── baselines/                         # Wrapper implementations
│   ├── README.md                      # Provenance per baseline
│   ├── sgcn/
│   │   ├── run.py                     # CLI entry: --method sgcn --dataset ...
│   │   ├── model.py                   # Reference impl or wrapper
│   │   └── hyperparams.yaml           # Locked HPO results
│   ├── sigat/
│   ├── sgcl/
│   ├── sigformer/
│   ├── sesgformer/
│   └── dadsgnn/
│
├── gomb_strict/                       # Our model
│   ├── run.py
│   ├── model.py                       # Cascade definition
│   └── hyperparams.yaml
│
├── datasets/                          # Loaders with content hashes
│   ├── README.md                      # Hash table per dataset
│   ├── bitcoin_alpha.py
│   ├── bitcoin_otc.py
│   ├── slashdot.py
│   ├── epinions.py
│   └── reddit_hyperlinks.py
│
├── hpo_logs/                          # Per-method HPO logs
│   ├── sgcn_bitcoin_alpha.optuna.db
│   └── … (35 files: 7 methods × 5 datasets)
│
├── tables/                            # Generated tables (re-runnable)
│   ├── table1_audit_matrix.py         # Builds Table 1 from logs
│   ├── table2_strict_protocol.py
│   └── figures.py                     # All paper figures
│
├── results/                           # Pre-computed reproducibility outputs
│   ├── phase_b/                       # Per (method, dataset, protocol, seed)
│   │   └── … (~210 JSONs at ~5 KB each ≈ 1 MB)
│   └── final_tables/
│
└── docs/
    ├── DATA_AVAILABILITY.md
    ├── CODE_AVAILABILITY.md
    ├── REPRODUCIBILITY_CHECKLIST.md
    └── REPORTING_SUMMARY.md           # NComms-specific
```

---

## The audit code — public API

The central scientific artifact. Must run without ML/GPU knowledge:

```python
from signed_link_audit import label_shuffle_audit

# Run the audit on any signed-link-prediction model.
audit_result = label_shuffle_audit(
    model_runner=lambda split, signs: run_my_method(split, signs),
    dataset="bitcoin_alpha",
    protocol="leaky",           # or "strict"
    n_seeds=3,
    n_shuffles=10,
)

# audit_result is a dataclass with:
#   A_unshuffled: float (regular AUROC)
#   S_shuffled: float (AUROC after shuffling test-edge signs)
#   delta: float (= A_unshuffled - S_shuffled; > 0 means audit fires)
#   audit_passes: bool (delta within tolerance of 0.5, i.e., model
#                      is recovering the unshuffled signal as expected
#                      without leakage)
```

The audit is the falsifiable test. The paper's central methodological
claim is: **a publishable signed-link-prediction model must pass
this audit**.

> **[TODO/CW]** Implementation status:
> - Strict-protocol split builder: [exists? location?]
> - Label-shuffle code: [exists? location?]
> - Public API as above: NOT YET (~1 day to wrap)

---

## Container recipe

Minimum target: `docker build` + `docker run --gpus all` reproduces
Table 1 in < 24 hours on a single consumer GPU.

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION=2.11.0
ARG CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       torch==${PYTORCH_VERSION} \
       --index-url ${CUDA_INDEX_URL}

COPY . /workspace/
RUN pip install --no-cache-dir -e .

CMD ["./reproduce.sh"]
```

`reproduce.sh` should:
1. Verify dataset hashes (fail early if corrupted).
2. Run the audit on a single (method, dataset) pair as a smoke test.
3. Echo the audit result so the user knows the package works.
4. Print the command for the full Table 1 reproduction.

> **[TODO/CW]** Decide: pin to torch 2.11 (current dev version) or
> torch 2.12 (CORE.YAML pin)? The audit code must be torch-version-stable.

---

## Data Availability statement (draft for NComms)

```
The signed-link prediction datasets used in this study are all
publicly available:

- Bitcoin-Alpha:  Kumar et al. (2016), available at
  http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html

- Bitcoin-OTC:    Kumar et al. (2016), available at
  http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html

- Slashdot:       Leskovec et al. (2010), available at
  http://snap.stanford.edu/data/soc-sign-Slashdot090221.html

- Epinions:       Leskovec et al. (2010), available at
  http://snap.stanford.edu/data/soc-sign-epinions.html

- Reddit Hyperlinks: Kumar et al. (2018), available at
  http://snap.stanford.edu/data/soc-RedditHyperlinks.html

The exact dataset versions used in this study, together with their
SHA-256 hashes and the train/test splits used for the strict
training-edge-only protocol, are deposited in the reproducibility
container (Code Availability) under `splits/`.

No new datasets were generated for this study.
```

> **[TODO/CW]** Confirm all 5 SNAP URLs are current (they sometimes
> rot; SNAP redirects). Generate the SHA-256 of the actual files we
> use and commit them to `splits/<dataset>/SHA256`.

---

## Code Availability statement (draft for NComms)

```
All code, hyperparameters, frozen train/test splits, and pre-computed
audit results are available at:

https://github.com/[TODO/CW: org]/[TODO/CW: repo]
DOI: 10.5281/zenodo.[TODO/CW: get a Zenodo DOI for archival]
License: [TODO/CW: pick one — MIT or Apache-2.0 strongly recommended
        for NComms; LGPL/GPL acceptable; non-commercial restricts]

A pre-built Docker container that reproduces every number in Table 1
and Table 2 is at:

  docker pull [TODO/CW: docker hub user]/signed-link-audit:1.0

A single command, `docker run --gpus all signed-link-audit:1.0
./reproduce.sh table1`, regenerates the central audit matrix in
~24 hours on a single NVIDIA RTX 2070 SUPER. The audit module
`audit/label_shuffle.py` exposes a Python API that allows any new
signed-link-prediction method to be tested for leakage in three
lines of code; we recommend its inclusion in future papers.
```

> **[TODO/CW]** Zenodo DOI request is a 24-hour turnaround; do it now
> so the DOI is ready by submission day. License decision is yours
> + coworker — but for NComms editorial compatibility, MIT or
> Apache-2.0 is the safe bet.

---

## Reproducibility Checklist (NComms-specific)

NComms requires a "Reporting Summary" PDF and a "Reproducibility
Checklist." Pre-compiled checklist responses:

| Item | Status | Note |
|---|---|---|
| Code publicly available | [Y]/[N] | Pending the GitHub push |
| Data publicly available | [Y]/[N] | All 5 from SNAP |
| Random seeds reported | [Y]/[N] | seeds {0, 1, 2} per cell |
| Hardware reported | [Y]/[N] | RTX 2070 SUPER, 8 GiB VRAM |
| Compute budget reported | [Y]/[N] | ~24 GPU-hours headline + ~245 hr full audit |
| HPO procedure reported | [Y]/[N] | Optuna 50 trials per (method, dataset) |
| Statistical tests reported | [Y]/[N] | Bootstrap-95 CI on inflation |
| Error bars on every reported number | [Y]/[N] | n=3 seeds, mean ± std |
| External baselines run from official code | [Y]/[N] | [TODO/CW] confirm per-baseline |
| Container provided | [Y]/[N] | Docker image |

> **[TODO/CW]** Each `[Y]/[N]` becomes `[Y]` only when the
> corresponding artifact exists. The cleanest submission has zero
> `[N]`s.

---

## Author Contributions section (CRediT taxonomy)

```
[A.B.] Conceptualisation; Methodology; Investigation; Writing —
       original draft; Visualisation; Software.
[C.D.] Methodology; Investigation; Validation; Writing —
       review & editing.
[E.F.] Funding acquisition; Supervision; Resources.
```

> **[TODO/CW]** Coordinate with coworker on the actual contribution
> roles. The CRediT taxonomy has 14 roles; pick the 3-5 that match.

---

## Competing Interests

```
The authors declare no competing interests.
```

(Standard line — adjust only if there's an actual competing interest
to disclose, e.g., commercial license to the framework.)

---

## What's blocking the package right now

1. **Audit code not yet a public-API module.** Currently scattered
   across notebooks / training scripts. ~1 day to wrap as the public
   `audit/label_shuffle.py` API.
2. **Frozen splits not yet committed**. Need to generate train/test
   split files for each dataset under the strict protocol and hash
   them. ~2 hours.
3. **Phase B audit matrix incomplete** (see `02_phase_b_audit_matrix.md`).
   Without it, Table 1 has placeholders.
4. **GitHub repo + Zenodo DOI not yet set up**. ~1 day for the
   repository hygiene (LICENSE, README, CI tests on the audit code).
5. **Docker container not yet built / tested**. ~1 day to build + run
   the smoke test.

> **[TODO/CW]** Order these by who-does-what; this is a coworker
> coordination question, not a technical one.
