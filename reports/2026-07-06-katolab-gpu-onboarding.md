# Katolab GPU (`kato15`) onboarding — 2026-07-06

**Author:** Aiko (agent) · **Host:** kato15.katolab.nitech.ac.jp · **Window:** 2026-07-06 15:49–17:19 +09:00

## Summary

Brought up passwordless SSH to the Katolab GPU box and provisioned it **fully** for the
`hymeko_rl` line: Python + CUDA env, the compiled Rust `hymeko` CLI, the PyO3 `hymeko` module,
and headless GPU rendering. Final `hymeko_rl/tests`: **657 passed, 4 failed, 14 skipped**. The
4 failures are `test_topology_zoo`, **pre-existing in the working tree** (they fail identically
on the local Windows machine). The box is ready to train.

## Access (password → key, then retired)

- Server offers `publickey,password`. Installed the shared empty-passphrase `~/.ssh/id_ed25519`
  into the remote `~/.ssh/authorized_keys` (`tee -a`; password entered **once**, then retired —
  nothing stored). Added `Host kato15` to `~/.ssh/config`. `ssh kato15` is passwordless.

## Hardware

| Resource | Value |
|---|---|
| GPU | 1× NVIDIA RTX 6000 Ada, 48 GB VRAM (cc 8.9), driver 570.153.02 (CUDA 12.8) |
| CPU / RAM | 32 cores / 125 GiB |
| Home | NFS `quartz:/export/home/hajdu`, 49 TB (38 free); no `/scratch`, only `/tmp` |
| OS / shell | Linux 5.15 (Ubuntu 20.04, gcc 9.4); base python 3.8; login shell **tcsh (noclobber)** |

## Provisioning

1. **uv** 0.11.26 → `~/.local/bin`; venv `~/envs/hymeko` = CPython 3.12.13.
2. **torch 2.11.0+cu128** (cu128, not local cu132 — driver 570 caps at CUDA 12.8) + triton 3.6,
   numpy 2.5, mujoco 3.10, gymnasium 1.3, matplotlib, imageio(+ffmpeg), scipy 1.18, pytest,
   hypothesis. GPU verified: `cuda True`, 4096² matmul finite.
3. **Rust:** rustup/cargo 1.96.1 in `~/.cargo` (no sudo). Full 27-crate workspace transferred;
   `cargo build -p hymeko_cli --release` (61 s) → `target/release/hymeko` v0.1.0. The MuJoCo envs
   emit MJCF by shelling out to this binary.
4. **Headless rendering:** `MUJOCO_GL=egl` — passed per run (persisting to `~/.cshrc` was declined
   by the sandbox). EGL offscreen on the RTX 6000 verified (240×320 RGB render). Without it,
   MuJoCo's GLFW renderer `abort()`s on the display-less box (`test_render_reach.py` probes a
   `Renderer` at import → SIGABRT that took down the whole test process).
5. **PyO3 `hymeko` module** (`import hymeko`, for StructuralCritic / graph enumeration):
   `maturin develop --release --uv`. Three no-sudo fixes were needed, each surfaced by the prior:
   `--uv` (uv venvs have no pip) → `libclang` pip wheel + `LIBCLANG_PATH` (the `ipc` feature pulls
   iceoryx2/zenoh → bindgen, no libclang on box) → `BINDGEN_EXTRA_CLANG_ARGS=-isystem
   /usr/lib/gcc/x86_64-linux-gnu/9/include` (clang's builtin `stddef.h` missing from the wheel).

## Code delivery

Python working tree tar'd from Windows (**option B** — uncommitted edits, not git): `hymeko_rl`
+ hard dep `hymeko_neuro` + `data/robotics` + the Rust crates. Excluded `target/`, checkpoints,
`*.pt/.so`, and `assets/data` (867 MB datasets).

## Test results — `hymeko_rl/tests` (CLI + PyO3 built, `MUJOCO_GL=egl`)

`657 passed, 4 failed, 14 skipped` in 121 s. Progression: 477 (Python only) → 647 (+ CLI + egl)
→ 657 (+ PyO3 wheel).

| Failure class | Count | Cause | Box-caused? |
|---|---|---|---|
| `test_topology_zoo` | 4 | `assert n_vertices == 9` logic mismatch | **no — pre-existing**; fails identically locally |

`test_structural_critic` (10 previously red) now **passes** after the PyO3 build.

## Run recipe

```sh
ssh kato15
cd ~/hymeko_framework_rust
env MUJOCO_GL=egl ~/envs/hymeko/bin/python -m pytest hymeko_rl/tests        # suite
env MUJOCO_GL=egl ~/envs/hymeko/bin/python -m hymeko_rl.train...            # a training script
```

## Open items

- **`test_topology_zoo`** (4): pre-existing working-tree failure, tracked separately from onboarding.

CORE.YAML items touched: none. New repo dependencies: none (remote-only provisioning).
Remote userspace toolchains installed: uv 0.11.26, rustup/cargo 1.96.1, maturin 1.14, libclang 18.
