"""Stage 0 completion — live model-size × gating sweep on the synthetic task, with performance.

For each local ollama model: does the HyMeKo parse+faithfulness gate raise its HTL-spec F1 (H4, augmentation), and
does the benefit grow as the model shrinks? Reported alongside per-model performance (round-trip, calc time,
tokens/s, footprint) — because "cheap + local" is half the value proposition. No API key, no metaworld: fully local.

    python -m hymeko_rl.eval.spec_bench.run_model_sweep --models llama3.2:1b gemma2:2b phi3:3.8b qwen2.5:3b
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from hymeko_rl.eval.spec_bench.ollama_model import OllamaChatModel, installed_models, ollama_available
from hymeko_rl.eval.spec_bench.openai_model import OpenAIChatModel, openai_available
from hymeko_rl.eval.spec_bench.spec_bench import (
    ChatModel,
    formula_f1,
    propose_and_gate,
    score_raw,
    synth_rollouts,
)


def _rows(report: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", report["models"])


def _num(x: object) -> float:
    return float(x) if isinstance(x, (int, float)) else 0.0

_FORMAL = "F(in_place >= 0.9)"          # the expert ceiling for the synthetic task
_SIGNALS = ("near_object", "grasp_success", "in_place", "obj_to_target")

_SYSTEM = ("You are a formal-methods assistant. Output ONLY one HTL formula and nothing else — no prose, no "
           "explanation, no code fences.")

_PROMPT = f"""Write an HTL success formula for a robot manipulation episode.

The episode SUCCEEDS when the object is delivered to and settles at the target location.

Available per-step signals (numbers): {', '.join(_SIGNALS)}.
  - in_place: how well the object is placed at the target (0..1, ~1 when placed)
  - obj_to_target: distance object->target (~0 when placed)
  - near_object: gripper proximity to object (0..1)
  - grasp_success: 1 if grasped else 0

HTL grammar (STRICT):
  - temporal ops F (eventually) and G (globally), ALWAYS with ROUND parentheses: F(...) or G(...);
  - an optional interval uses square brackets with two numbers only: F[0,5](...);
  - boolean ops are UPPERCASE: AND, OR, NOT;
  - a predicate is  SIGNAL CMP NUMBER  where CMP is one of  <  <=  >  >=  ==  (use == for equality, never a single =).
Full valid example (syntax only, NOT the answer): F(near_object >= 0.5 AND grasp_success == 1)

Output exactly one HTL formula that is TRUE iff the episode succeeds:"""


@dataclass
class SweepConfig:
    k: int = 4                          # proposals per arm
    retries: int = 2                    # parse-gate error-loop retries
    n_verif: int = 40
    n_test: int = 80
    seed_verif: int = 100
    seed_test: int = 200


def run_model(name: str, model: ChatModel, cfg: SweepConfig, size_bytes: int = 0) -> dict[str, object]:
    """Raw vs gate F1 for one model on the synthetic task (+ perf if it is an OllamaChatModel)."""
    verif = synth_rollouts(cfg.n_verif, seed=cfg.seed_verif)
    test = synth_rollouts(cfg.n_test, seed=cfg.seed_test)
    raw_formula, n_valid, n_att = score_raw(model, _PROMPT, k=cfg.k, system=_SYSTEM)
    gate = propose_and_gate(model, _PROMPT, verif, k=cfg.k, retries=cfg.retries, system=_SYSTEM)
    raw_f1 = formula_f1(raw_formula, test) if raw_formula else 0.0
    gate_f1 = formula_f1(gate.formula, test) if gate.formula else 0.0
    _ps = getattr(model, "perf_summary", None)
    perf = _ps() if callable(_ps) else {}
    return {
        "model": name, "size_gb": round(size_bytes / 1e9, 2) if size_bytes else None,
        "raw_f1": round(raw_f1, 4), "gate_f1": round(gate_f1, 4),
        "gate_minus_raw": round(gate_f1 - raw_f1, 4),
        "raw_parse_rate": round(n_valid / max(1, n_att), 3),
        "raw_formula": raw_formula, "gate_formula": gate.formula, "gate_attempts": gate.n_attempts,
        "perf": perf,
    }


def run_sweep(models: list[str], cfg: SweepConfig | None = None, *, openai_model: "str | None" = "gpt-4o-mini",
              ) -> dict[str, object]:
    """Live sweep over local ``models`` (skipped if ollama is down) + the OpenAI strength ablation (if a key is
    present) + the formal ceiling."""
    cfg = cfg or SweepConfig()
    test = synth_rollouts(cfg.n_test, seed=cfg.seed_test)
    formal_f1 = formula_f1(_FORMAL, test)
    rows: list[dict[str, object]] = []
    sizes = installed_models()
    if ollama_available():
        for name in models:
            try:
                rows.append(run_model(name, OllamaChatModel(model=name), cfg, sizes.get(name, 0)))
            except Exception as e:                          # a dead/slow model must not sink the sweep
                rows.append({"model": name, "error": f"{type(e).__name__}: {e}"})
    if openai_model and openai_available():
        try:
            rows.append(run_model(f"openai:{openai_model}", OpenAIChatModel(model=openai_model), cfg))
        except Exception as e:
            rows.append({"model": f"openai:{openai_model}", "error": f"{type(e).__name__}: {e}"})
    return {
        "config": asdict(cfg), "formal_formula": _FORMAL, "formal_f1": round(formal_f1, 4),
        "ollama_available": ollama_available(), "openai_available": openai_available(), "models": rows,
    }


def plot_sweep(report: dict[str, object], out_path: str | Path) -> Path:
    """Grouped bars: raw vs gate F1 per model (sorted by size), formal ceiling line (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in _rows(report) if "error" not in r]
    rows.sort(key=lambda r: _num(r.get("size_gb")))
    names = [f"{r['model']}\n{r.get('size_gb','?')} GB" for r in rows]
    raw = [_num(r["raw_f1"]) for r in rows]
    gate = [_num(r["gate_f1"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(rows)), 4.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, raw, 0.4, label="raw", color="#ff7f0e", edgecolor="black")
    ax.bar(x + 0.2, gate, 0.4, label="+ HyMeKo gate", color="#2ca02c", edgecolor="black")
    ax.axhline(cast("float", report["formal_f1"]), ls="--", color="#1f77b4", label=f"formal ceiling {report['formal_f1']}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("HTL-spec F1 vs native success")
    ax.set_ylim(0, 1.05)
    ax.set_title("Does HyMeKo gating augment a small local LLM? (synthetic task)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["llama3.2:1b", "gemma2:2b", "phi3:3.8b", "qwen2.5:3b"])
    ap.add_argument("--out-dir", default="reports/figures/2026_07_13_spec_bench_model_sweep")
    a = ap.parse_args(argv)
    report = run_sweep(a.models)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(report, indent=2, default=float))
    if any("error" not in r for r in _rows(report)):
        plot_sweep(report, out / "sweep")
    print(json.dumps({"formal_f1": report["formal_f1"],
                      "models": [{k: r.get(k) for k in ("model", "size_gb", "raw_f1", "gate_f1",
                                                        "gate_minus_raw", "raw_parse_rate", "perf")}
                                 for r in _rows(report)]}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
