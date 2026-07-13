"""Live ollama-backed ``ChatModel`` for spec_bench, with per-call performance capture.

Mirrors akoire's ``OllamaModel`` config (``localhost:11434/api/generate``, ``think:false`` — thinking models
otherwise return an empty ``response``). Stdlib HTTP only (``urllib``) — no new dependency (§1). Every call records
round-trip wall time + ollama's own timing (load / eval / tokens-per-second) so the model-size × gating sweep can
report performance alongside faithfulness.
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class OllamaChatModel:
    """A ``ChatModel`` (``complete(system, prompt) -> str``) over a local ollama server, timing every call.

    # Invariants no global state; ``timings`` grows one entry per ``complete`` call."""

    model: str
    host: str = "http://localhost:11434"
    num_predict: int = 96
    temperature: float = 0.2
    connect_timeout: float = 10.0
    read_timeout: float = 120.0
    timings: list[dict[str, float]] = field(default_factory=list)

    def complete(self, system: str, prompt: str) -> str:
        body = json.dumps({
            "model": self.model, "system": system, "prompt": prompt,
            "stream": False, "think": False,                 # thinking on → empty response (akoire's lesson)
            "options": {"num_predict": self.num_predict, "temperature": self.temperature},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.read_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rtt_ms = (time.perf_counter() - t0) * 1e3
        ev, ev_ns = int(data.get("eval_count", 0)), int(data.get("eval_duration", 0))
        self.timings.append({
            "rtt_ms": round(rtt_ms, 1),                      # round-trip (wall)
            "total_ms": round(int(data.get("total_duration", 0)) / 1e6, 1),
            "load_ms": round(int(data.get("load_duration", 0)) / 1e6, 1),
            "eval_ms": round(ev_ns / 1e6, 1),                # calculation time (generation compute)
            "eval_count": float(ev),
            "tokens_per_s": round(ev / (ev_ns / 1e9), 2) if ev_ns > 0 else 0.0,
        })
        return str(data.get("response", ""))

    def perf_summary(self) -> dict[str, float]:
        """Median performance over all calls this model made (empty → zeros)."""
        if not self.timings:
            return {"calls": 0.0, "rtt_ms_median": 0.0, "eval_ms_median": 0.0, "tokens_per_s_median": 0.0}
        return {
            "calls": float(len(self.timings)),
            "rtt_ms_median": round(statistics.median(t["rtt_ms"] for t in self.timings), 1),
            "eval_ms_median": round(statistics.median(t["eval_ms"] for t in self.timings), 1),
            "load_ms_median": round(statistics.median(t["load_ms"] for t in self.timings), 1),
            "tokens_per_s_median": round(statistics.median(t["tokens_per_s"] for t in self.timings), 2),
        }


def ollama_available(host: str = "http://localhost:11434", timeout: float = 5.0) -> bool:
    """True iff the ollama server answers ``/api/tags`` (so a live arm can be skipped cleanly when absent)."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def installed_models(host: str = "http://localhost:11434", timeout: float = 5.0) -> dict[str, int]:
    """Map ``model name -> size bytes`` from ``/api/tags`` (size = the memory/footprint proxy)."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            tags = json.loads(resp.read().decode("utf-8")).get("models", [])
    except (urllib.error.URLError, OSError):
        return {}
    return {m["name"]: int(m.get("size", 0)) for m in tags}
