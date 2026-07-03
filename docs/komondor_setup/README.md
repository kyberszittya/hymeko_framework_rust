# Komondor HPC setup — probe-first staging

**Goal:** stand up enough of the hymeko_neuro environment on Komondor
(KIFÜ HPC at HUN-REN, `komondor.hpc.dkf.hu`) to submit ONE SGCN smoke
run via SLURM, confirm it produces the expected JSON, then decide on
the bigger NComms Phase B sweep.

**Status (2026-06-02):** all account-specific values filled in from
the user's actual `sacctmgr` output. Submit script is ready; only the
Singularity image build needs to run (Step 2) before the probe.

**This account:**
- Username:    `pr_szhc`
- Project:     `pr_szevis` (shared with K. Környei; subdir `hajdu/` to keep
                            our work isolated)
- Workspace:   `/scratch/pr_szevis/hajdu/hymeko/hymeko_framework_rust/`
- MaxWall:     7 days/job
- QOS:         `lowpri,normal` (default `normal`)
- Singularity: `singularity-ce 4.0.1` ✓ (supports `--fakeroot --fix-perms`)

---

## Cluster facts (from docs.hpc.dkf.hu, 2026-06-02)

| Concern | Value |
|---|---|
| Login node | `komondor.hpc.dkf.hu` |
| SSH | `ssh pr_szhc@komondor.hpc.dkf.hu` |
| Auth | SSH key + EduID 2FA (browser link during connection) |
| Restriction | Client IP must be registered in Hungary (or NAT'd via one) |
| Username | **Per-project** — find in HPC Portal → Projects |
| Billing | `#SBATCH --account=<account>` is **required** (per-project) |
| GPU partition | `gpu` — 58 nodes × 4× A100 40GB SXM, 64-core EPYC 7763, 256 GB RAM |
| AI partition | `ai` — 4 nodes × 8× A100 40GB SXM (single-node DDP) |
| Containers | **Singularity** (not Apptainer); `module load singularity`; `--fakeroot --fix-perms` supported |
| PyTorch modules | 2.2.2 / 2.4.1 / 2.6.0 — **none match our local torch 2.11.0**, hence the container route |
| CUDA | Auto-provided by `pytorch` module; loading CUDA separately is "possibly disruptive" |
| Filesystem | `/home/<user_ID>` (20 GB / 100k inodes, code only); `/scratch/<project_ID>` (NVMe, **per-project allocation — squota to verify**, repo + datasets + outputs); `/project/<project_ID>` (HDD, **per-project allocation — squota to verify**, archive) |
| `pr_szevis` actual quotas (2026-06-02 squota) | `/scratch`: **125 GB soft / 137.5 GB hard, 300k inode soft**; `/project`: 500 GB soft, 1M inode soft. Docs-page numbers ("1 TB / 300k") are the default project allocation, not ours. |

**Implication of torch version mismatch:** the local env uses
torch 2.11.0; the cluster's newest module is 2.6.0. Three options:

1. **Singularity container with torch 2.11.0 baked in** (recommended; this
   folder is set up for it).
2. Pin local to torch 2.6.0 and use Komondor's `pytorch/2.6.0` module
   directly (faster job start, but breaks the "identical env"
   guarantee for the NComms reproducibility hook).
3. Build torch 2.11.0 against the cluster's CUDA from source — slow,
   fragile, not recommended.

---

## Step 0 — Verify your account + module surface

SSH in and capture:

```bash
ssh pr_szhc@komondor.hpc.dkf.hu        # complete EduID 2FA in browser

# Confirm SLURM account code (this is what #SBATCH --account needs)
squota                                # shows your project allocations
sacctmgr show user $USER -s          # shows associations + accounts

# Confirm singularity is available
module avail singularity 2>&1 | head -5
module load singularity && singularity --version

# Confirm pytorch module versions (sanity, not used for our container path)
module avail pytorch 2>&1 | head -10

# Confirm scratch path for YOUR project
ls -ld /scratch/$(id -ng)/           # adjust if project_id ≠ primary group
```

Paste the `squota` + `sacctmgr` output back and I'll fill the
`#SBATCH --account=` line.

## Step 1 — Stage the repo on /scratch (not $HOME)

`$HOME` has a 20 GB quota — our repo + datasets eat that. Put
everything on `/scratch/<project_ID>` (1 TB, NVMe).

**Option A: git clone from a remote** (if you push to one):

```bash
PROJECT=$(id -ng)   # or set to the actual project_ID
mkdir -p /scratch/$PROJECT/hymeko && cd /scratch/$PROJECT/hymeko
git clone <your_remote_url> hymeko_framework_rust
cd hymeko_framework_rust
git log -1   # confirm SHA matches your workstation
```

**Option B: rsync from workstation** (no remote needed):

```bash
# 1. Create the namespace dir first (rsync doesn't auto-create deep paths
#    on older versions; --mkpath needs rsync 3.2.3+).
ssh komondor 'mkdir -p /scratch/pr_szevis/hajdu/hymeko/'

# 2. rsync from workstation (NOT inside the repo dir):
rsync -avz --exclude='target/' --exclude='.venv/' --exclude='.venv-rapport-ros2/' \
      --exclude='*.pdf' --exclude='__pycache__' \
      --exclude='hymeko_neuro/experiments/results/' \
      --exclude='.git/' \
      /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust/ \
      komondor:/scratch/pr_szevis/hajdu/hymeko/hymeko_framework_rust/
```

(The `komondor:` short form uses the SSH alias from `~/.ssh/config`.)

The exclusions trim ~5+ GB of build artefacts, venvs, run logs, and
the .git history (omit `--exclude='.git/'` if you want git on the
cluster too).

## Step 2 — Build the Singularity image (on the login node)

```bash
cd /scratch/pr_szevis/hajdu/hymeko/hymeko_framework_rust
module load singularity

# ~10-15 min; produces a ~5-7 GB .sif on /scratch (not $HOME)
singularity build --fakeroot --fix-perms \
    hymeko_signedkan.sif \
    docs/komondor_setup/hymeko_signedkan.def

# Verify
singularity inspect --labels hymeko_signedkan.sif
singularity exec --nv hymeko_signedkan.sif \
    python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
# (--nv may print "no GPU" on the login node; that's fine; the .sif is good.)
```

The .sif file is the canonical NComms Phase B execution environment.
Its hash (`singularity inspect --labels` → see `org.hymeko.*` labels)
becomes the citation pin for the paper's reproducibility section.

**If the build fails** (network restrictions on the docker pull,
`--fakeroot` not actually enabled, etc.), see the fallback paths in
`hymeko_signedkan.def` comments or fall back to a venv per the local
torch 2.11.0 wheels.

## Step 3 — The probe SLURM job

The script is already filled in (`--account=pr_szevis`, `--partition=gpu`).
Switch to `--partition=ai` only if you need 8 GPUs per node for DDP.

```bash
cd /scratch/pr_szevis/hajdu/hymeko/hymeko_framework_rust
sbatch docs/komondor_setup/submit_sgcn_smoke.sh
squeue -u $USER   # watch it queue / run
```

Expected output (on success, in `slurm_logs/sgcn-smoke-<JOBID>.out`):
- one JSON line:
  `{"dataset": "bitcoin_alpha", "model": "SGCN", "auc": 0.87xx, ...}`
- Wall < 60 s on an A100
- Peak RSS < 2 GB

## Step 4 — Sanity gates before scaling up

| Gate | What it confirms |
|---|---|
| JSON line in output | dispatch + data load + training loop wired |
| AUC ∈ [0.85, 0.92] for SGCN/bitcoin_alpha | reproduces local 0.870 from `sgcn_baseline.json` |
| Wall < 60 s | partition + GPU are right-sized |
| No CUDA errors in stderr | container ↔ host driver wiring OK |
| `nvidia-smi` shows A100 40GB | partition routing OK |

If any of these fails, do NOT submit the bigger Phase B sweep. The
cluster's environment can have surprises (driver ABI mismatch with
container's CUDA, fakeroot quirks, scratch permissions) that overwhelm
the audit's budget if they show up at run 50/200 instead of 1/1.

## Pre-sweep checklist (run before any multi-job array)

Before submitting any SLURM array, run these on the login node and
size the array against the actual numbers (not against docs-page
defaults or my prior wall-time guesses):

```bash
# 1. Filesystem headroom — will outputs fit?
squota                  # /scratch + /project, soft/hard, inode caps

# 2. Concurrent-job cap — how many cells run in parallel?
sacctmgr -p show assoc user=$USER \
    format=Account,Partition,QOS,MaxJobs,MaxNodes,MaxSubmit,GrpJobs,GrpNodes

# 3. QOS-level caps (separate from association)
sacctmgr -p show qos normal lowpri format=MaxJobsPU,MaxNodesPU,MaxSubmitPU,MaxWall

# 4. Current usage on this account (in CPU-hours)
sreport cluster AccountUtilizationByUser \
    Accounts=$(sacctmgr -nP show user $USER format=defaultaccount) \
    -t hours start=2026-01-01 end=now

# 5. Sample memory footprint for the workload (run ONE cell first)
sbatch <single_probe>.sh    # then check `sacct -j <id> -o MaxRSS,Elapsed,State`
```

The HSiKAN-Optuna round-1 OOM (2026-06-02, 40 cells at `--mem=24G`)
happened because I skipped step 5 — never measured an HSiKAN cell's
peak RSS on Komondor before fanning out 40 of them. Don't repeat.

## Step 5 — When the probe passes

Three options unlock:

1. **HSiKAN-Optuna full audit** (4 datasets × 4 conditions × 5 seeds
   = 80 jobs, ~5-15 min each on A100 40GB) — fills the HSiKAN rows of
   NComms Table 2 completely. Likely ≤6 GPU-hr queue, depending on
   priority.
2. **SGCN + SiGAT audit fan-out** (replicates the local sweep at
   5 datasets × 4 conditions × 3 seeds = 60 jobs, parallel) — ~1 hr
   queue time vs the local ~12 min serial; redundant if the local
   sweep tonight already lands clean.
3. **Phase B baseline port + sweep** — port SE-SGformer / DADSGNN /
   SiGformer / SGCL from their published code (1-2 days workstation
   work, no GPU needed), then run all 4 × 5 × 4 × 3 = 240 jobs
   parallel. ~4-8 hr queue time on `gpu` partition.

Decide after the probe lands.

---

## Files in this folder

| File | Purpose |
|---|---|
| [README.md](README.md) (this) | concrete staging guide with Komondor specifics |
| [submit_sgcn_smoke.sh](submit_sgcn_smoke.sh) | probe SLURM submission (1 SGCN run via Singularity) |
| [hymeko_signedkan.def](hymeko_signedkan.def) | Singularity definition file (torch 2.11.0 + CUDA 12.4) |
| [env_requirements.txt](env_requirements.txt) | frozen local pip env (fallback only; image is canonical) |

## Resolved (2026-06-02)

All staging values now concrete:

- `--account=pr_szevis` ✓
- `/scratch/pr_szevis/` ✓ (project-keyed, not user-keyed)
- `module load singularity` ✓ (singularity-ce 4.0.1)

Next concrete action: rsync the repo to
`/scratch/pr_szevis/hajdu/hymeko/hymeko_framework_rust/`, then run Steps 2
and 3 in order.
