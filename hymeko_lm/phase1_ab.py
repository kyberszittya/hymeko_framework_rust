"""Phase-1 go/no-go: byte-level language A/B — FSR-LM vs a matched-parameter causal transformer.

Metrics (CLAUDE.md §3): validation **bits-per-byte** (CE / ln 2 on the held-out tail), **tokens/s**
(median over a timed window after warm-up), and **parameter count**. Multi-seed median/IQR; one entry,
``--mode smoke|full`` (§6.5#13). Not bit-exact (stochastic training, RL-style carve-out): seeded for
resume, claims rest on the multi-seed median.

    uv run python -m hymeko_lm.phase1_ab --mode smoke --corpus <path-to-text>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from hymeko_lm.baselines import CausalTransformerLM
from hymeko_lm.config import FSRConfig, GateMode, ResidualMode
from hymeko_lm.model import FSRLanguageModel
from hymeko_lm.text_data import BYTE_VOCAB, ByteCorpus

LM = FSRLanguageModel | CausalTransformerLM   # both expose loss/n_parameters + the nn.Module surface

_MODES: dict[str, dict[str, int]] = {
    "smoke": {"n_blocks": 16, "n_layers": 3, "seq": 64, "batch": 16, "steps": 150, "eval_batches": 10},
    "full": {"n_blocks": 32, "n_layers": 4, "seq": 128, "batch": 16, "steps": 1500, "eval_batches": 40},
}


def _match_dim_ff(target_params: int, d: int, n_layers: int, seq: int) -> int:
    """Pick the transformer FFN width so its parameter count ≈ ``target_params`` (FSR's)."""
    fixed = 2 * BYTE_VOCAB * d + seq * d + n_layers * (4 * d * d + 6 * d)   # embeds+head+attn+norms
    return max(8, round((target_params - fixed) / (n_layers * 2 * d)))


def _fsr(cfg_d: dict[str, int], seq: int, residual: ResidualMode) -> FSRLanguageModel:
    return FSRLanguageModel(FSRConfig(vocab_size=BYTE_VOCAB, n_blocks=cfg_d["n_blocks"],
                                      n_layers=cfg_d["n_layers"], max_seq_len=seq, gate_rank=32,
                                      gate_mode=GateMode.SOFTMAX, residual_mode=residual))


def _build(kind: str, cfg_d: dict[str, int], seq: int) -> LM:
    if kind == "fsr_sphere":
        return _fsr(cfg_d, seq, ResidualMode.SPHERE)
    if kind == "fsr_prenorm":
        return _fsr(cfg_d, seq, ResidualMode.PRENORM)
    d = 3 * cfg_d["n_blocks"]
    n_heads = 6 if d % 6 == 0 else 4
    dim_ff = _match_dim_ff(_fsr(cfg_d, seq, ResidualMode.SPHERE).n_parameters(), d, cfg_d["n_layers"], seq)
    return CausalTransformerLM(BYTE_VOCAB, d, cfg_d["n_layers"], n_heads, seq, dim_ff)


@torch.no_grad()
def _val_bpb(model: LM, corpus: ByteCorpus, *, seq: int, batch: int, n: int,
             device: torch.device) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(12345)        # fixed val crops, independent of train seed
    nats = [float(model.loss(*(t.to(device) for t in corpus.batch(batch, seq, "val", gen))))
            for _ in range(n)]
    return statistics.mean(nats) / math.log(2.0)


def _tokens_per_s(model: LM, corpus: ByteCorpus, *, seq: int, batch: int,
                  device: torch.device) -> float:
    model.train()
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # measure compute, not learning
    def step() -> None:
        ids, tgt = (t.to(device) for t in corpus.batch(batch, seq, "train", gen))
        opt.zero_grad(set_to_none=True)
        model.loss(ids, tgt).backward()   # type: ignore[no-untyped-call]
        opt.step()
    for _ in range(5):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    durs = []
    for _ in range(10):
        t0 = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        durs.append(time.perf_counter() - t0)
    return (batch * seq) / statistics.median(durs)


def _train_one(kind: str, corpus: ByteCorpus, cfg_d: dict[str, int], *, seed: int,
               device: torch.device) -> dict[str, float]:
    torch.manual_seed(seed)
    seq, batch = cfg_d["seq"], cfg_d["batch"]
    model = _build(kind, cfg_d, seq).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(1000 + seed)
    for _ in range(cfg_d["steps"]):
        ids, tgt = (t.to(device) for t in corpus.batch(batch, seq, "train", gen))
        loss = model.loss(ids, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()   # type: ignore[no-untyped-call]
        opt.step()
    bpb = _val_bpb(model, corpus, seq=seq, batch=batch, n=cfg_d["eval_batches"], device=device)
    tps = _tokens_per_s(model, corpus, seq=seq, batch=batch, device=device)
    return {"val_bpb": round(bpb, 4), "tokens_per_s": round(tps, 1),
            "n_params": model.n_parameters()}


def run_ab(corpus_path: str, mode: str = "smoke", *, seeds: tuple[int, ...] = (0, 1, 2),
           device: str | None = None) -> dict[str, object]:
    """Train both models over ``seeds``; return per-model median val-bpb, tokens/s, params."""
    cfg_d = _MODES[mode]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    corpus = ByteCorpus(corpus_path)
    seed_list = (0,) if mode == "smoke" else seeds
    out: dict[str, object] = {"mode": mode, "device": dev.type, "seeds": list(seed_list)}
    for kind in ("fsr_sphere", "fsr_prenorm", "xfmr"):
        runs = [_train_one(kind, corpus, cfg_d, seed=s, device=dev) for s in seed_list]
        bpbs = [r["val_bpb"] for r in runs]
        out[kind] = {
            "val_bpb_median": round(statistics.median(bpbs), 4),
            "val_bpb_iqr": round((max(bpbs) - min(bpbs)), 4),
            "tokens_per_s_median": round(statistics.median(r["tokens_per_s"] for r in runs), 1),
            "n_params": runs[0]["n_params"], "per_seed_bpb": bpbs,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=sorted(_MODES), default="smoke")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="reports/2026-06-29-fsr-lm-phase1-ab.json")
    a = ap.parse_args(argv)
    report = run_ab(a.corpus, a.mode, device=a.device)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
