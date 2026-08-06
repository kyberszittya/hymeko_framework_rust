"""OpenAI-backed ``ChatModel`` for the spec_bench strength ablation — the strong reference vs the local models.

Raw HTTP (``urllib``) to the Chat Completions API — no ``openai`` pip dependency (§1). The key is read from
``.keys/OPENAI_API_AUTH.key`` (gitignored); the arm is skipped cleanly when the key is absent. Records round-trip
time + completion tokens per call for the performance table.
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_KEY_PATH = Path(__file__).resolve().parents[3] / ".keys" / "OPENAI_API_AUTH.key"
_URL = "https://api.openai.com/v1/chat/completions"


def load_openai_key(path: Path = _KEY_PATH) -> "str | None":
    """The OpenAI key from ``.keys/OPENAI_API_AUTH.key`` (stripped), or ``None`` if the file is absent/empty."""
    try:
        key = path.read_text(encoding="utf-8").strip()
        return key or None
    except OSError:
        return None


def openai_available(path: Path = _KEY_PATH) -> bool:
    return load_openai_key(path) is not None


@dataclass
class OpenAIChatModel:
    """A ``ChatModel`` (``complete(system, prompt) -> str``) over the OpenAI Chat Completions API, timing calls."""

    model: str = "gpt-4o-mini"
    max_tokens: int = 100
    temperature: float = 0.4
    read_timeout: float = 60.0
    key: "str | None" = None
    timings: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.key is None:
            self.key = load_openai_key()
        if not self.key:
            raise ValueError("no OpenAI key at .keys/OPENAI_API_AUTH.key")

    def complete(self, system: str, prompt: str) -> str:
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(_URL, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.read_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rtt_ms = (time.perf_counter() - t0) * 1e3
        completion = int(data.get("usage", {}).get("completion_tokens", 0))
        self.timings.append({
            "rtt_ms": round(rtt_ms, 1), "completion_tokens": float(completion),
            "tokens_per_s": round(completion / (rtt_ms / 1e3), 2) if rtt_ms > 0 else 0.0,
        })
        return str(data["choices"][0]["message"]["content"])

    def perf_summary(self) -> dict[str, float]:
        if not self.timings:
            return {"calls": 0.0, "rtt_ms_median": 0.0, "tokens_per_s_median": 0.0}
        return {
            "calls": float(len(self.timings)),
            "rtt_ms_median": round(statistics.median(t["rtt_ms"] for t in self.timings), 1),
            "tokens_per_s_median": round(statistics.median(t["tokens_per_s"] for t in self.timings), 2),
        }
