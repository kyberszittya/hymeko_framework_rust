"""Stage 0 core — synthetic labelled rollouts, HTL-predicate F1, and the LLM propose→gate loop.

The shared representation is an HTL success formula over signal names (``F``/``G`` temporal ops, predicates
``IDENT CMP NUMBER``, ``NOT/AND/OR``), evaluated by the unified ``hymeko_neuro/eval/htl`` engine. A rollout is a
trace of per-step signal dicts + a native-success label; an arm's predicate is graded by F1 of ``robustness>0`` vs
the label. The gate is AKOIRE's synthesize→gate→feedback applied at the HTL-spec level: parse-gate (reject
malformed, feed the parser error back for a retry) + faithfulness-select (pick the candidate with best F1 on a
held-out verification split). ``ScriptedModel`` makes the whole loop deterministic and testable offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from hymeko_neuro.eval.htl.event import HypergraphEvent
from hymeko_neuro.eval.htl.eval import robustness
from hymeko_neuro.eval.htl.parser import parse

_SIGNALS = ("near_object", "grasp_success", "in_place", "obj_to_target")
_BOOL_WORD = re.compile(r"\b(and|or|not)\b", re.IGNORECASE)   # lowercase bool ops → uppercase
_LONE_EQ = re.compile(r"(?<![<>=!])=(?!=)")                    # a lone '=' (not <= >= == !=) → '=='
# a predicate literal: SIGNAL CMP NUMBER (the threshold is group 3 — what calibration tunes).
_PRED = re.compile(r"([a-z_][a-z0-9_]*)\s*(<=|>=|==|<|>)\s*(\d+(?:\.\d+)?)")


def _repair(formula: str) -> str:
    """Conservative syntax normalisation of common LLM HTL mistakes — the framework's tolerant front-end (part of
    the *gate*, not the raw arm). Fixes only mechanical errors (bool-op case, lone ``=``, ``F[``/``G[`` brackets
    used for a sub-formula rather than an interval); it never changes which signals/thresholds were chosen."""
    f = _BOOL_WORD.sub(lambda m: m.group(1).upper(), formula)
    f = _LONE_EQ.sub("==", f)
    # brackets: protect real intervals [num,num], then flip any stray [ ] (used for a sub-formula) to ( ).
    intervals: list[str] = []

    def _protect(m: "re.Match[str]") -> str:
        intervals.append(m.group(0))
        return f"\x00{len(intervals) - 1}\x00"

    f = re.sub(r"\[\s*[\d.]+\s*,\s*[\d.]+\s*\]", _protect, f)
    f = f.replace("[", "(").replace("]", ")")
    for i, iv in enumerate(intervals):
        f = f.replace(f"\x00{i}\x00", iv)
    return f


@dataclass(frozen=True)
class Rollout:
    """One labelled rollout: a per-step signal trace + the native-success label.

    # Invariants ``trace`` is a non-empty list of ``{signal: value}`` dicts (one per step); ``success`` is the
      ground-truth label the predicates are graded against."""

    trace: list[dict[str, float]]
    success: bool

    def events(self) -> list[HypergraphEvent]:
        return [HypergraphEvent(t=float(i), scalar_signals=dict(step)) for i, step in enumerate(self.trace)]


# ── synthetic data (balanced; a KNOWN success predicate the arms must recover) ─────────────────────────────
def synth_rollouts(n: int, *, seed: int, steps: int = 20, noise: float = 0.15) -> list[Rollout]:
    """``n`` balanced rollouts. Ground truth: success iff ``in_place`` eventually reaches ``>= 0.9`` (i.e. the
    target predicate is ``F(in_place >= 0.9)``); other signals are correlated-but-not-decisive distractors, so a
    faithful spec must pick ``in_place``, not a confound. Half positive, half negative.

    # Postconditions ``len == n``; both classes present when ``n >= 2``; deterministic in ``seed``."""
    if n < 2:
        raise ValueError("need n >= 2 for a balanced set")
    rng = np.random.default_rng(seed)
    out: list[Rollout] = []
    for k in range(n):
        positive = k % 2 == 0
        peak = rng.uniform(0.92, 1.0) if positive else rng.uniform(0.3, 0.85)   # in_place peak decides success
        trace: list[dict[str, float]] = []
        for s in range(steps):
            frac = s / max(1, steps - 1)
            ip = float(np.clip(peak * frac + rng.normal(0, noise), 0.0, 1.2))
            trace.append({
                "in_place": ip,
                "near_object": float(np.clip(0.5 + 0.4 * frac + rng.normal(0, noise), 0.0, 1.0)),  # distractor
                "grasp_success": float(rng.random() < 0.6),                                        # distractor
                "obj_to_target": float(np.clip(1.0 - peak * frac + rng.normal(0, noise), 0.0, 1.5)),
            })
        out.append(Rollout(trace=trace, success=positive))
    return out


# ── HTL-predicate evaluation + F1 ──────────────────────────────────────────────────────────────────────────
def evaluate_formula(formula: str, rollouts: list[Rollout]) -> list[bool]:
    """Accept/reject each rollout under an HTL ``formula`` (accept iff robustness > 0).

    # Preconditions ``formula`` parses (call inside a parse-gated context). # Postconditions one bool per rollout.
    # Raises whatever :func:`parse` raises on a malformed formula."""
    node = parse(formula)
    return [robustness(node, r.events()) > 0.0 for r in rollouts]


def f1_score(pred: list[bool], labels: list[bool]) -> dict[str, float]:
    """Binary F1/precision/recall of ``pred`` vs ``labels`` (positive class = success)."""
    tp = sum(1 for p, y in zip(pred, labels) if p and y)
    fp = sum(1 for p, y in zip(pred, labels) if p and not y)
    fn = sum(1 for p, y in zip(pred, labels) if not p and y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)}


def formula_f1(formula: str, rollouts: list[Rollout]) -> float:
    """F1 of a parse-valid ``formula`` on ``rollouts`` (0.0 if it fails to parse — an unusable proposal)."""
    try:
        return f1_score(evaluate_formula(formula, rollouts), [r.success for r in rollouts])["f1"]
    except Exception:
        return 0.0


# ── threshold calibration (the arbiter's "refine": keep the LLM's structure, fit its constants to data) ──────
def _signal_grid(rollouts: list[Rollout], signal: str, n: int = 13) -> list[float]:
    """Candidate thresholds for ``signal`` — ``n`` evenly-spaced values over its observed range."""
    vals = [step[signal] for r in rollouts for step in r.trace if signal in step]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    return [lo] if hi <= lo else [round(lo + (hi - lo) * i / (n - 1), 3) for i in range(n)]


def _rebuild(formula: str, spans: list[tuple[int, int]], thresholds: list[float]) -> str:
    """Substitute new ``thresholds`` at the numeric ``spans`` of ``formula`` (right-to-left, spans stay valid)."""
    out = formula
    for (s, e), t in sorted(zip(spans, thresholds), key=lambda x: -x[0][0]):
        out = out[:s] + f"{t:g}" + out[e:]
    return out


def calibrate_thresholds(formula: str, rollouts: list[Rollout], *, grid: int = 13, passes: int = 2) -> str:
    """Coordinate-ascent fit of the formula's numeric thresholds to ``rollouts`` (max F1), **keeping the structure**
    (signals, comparison ops, temporal/logical form) exactly as the LLM proposed. This is the augmenter/arbiter
    division of labour: the LLM picks *what* to look at; HyMeKo (holding the data) picks the *constants*.

    # Preconditions ``formula`` parses. # Postconditions returns a parse-valid formula with the same structure and
      (weakly) higher F1 on ``rollouts``."""
    preds = list(_PRED.finditer(formula))
    if not preds:
        return formula
    spans = [(m.start(3), m.end(3)) for m in preds]
    signals = [m.group(1) for m in preds]
    thresholds = [float(m.group(3)) for m in preds]
    best_f1 = formula_f1(_rebuild(formula, spans, thresholds), rollouts)
    for _ in range(passes):
        for i, sig in enumerate(signals):
            for t in _signal_grid(rollouts, sig, grid):
                trial = thresholds[:]
                trial[i] = t
                f1 = formula_f1(_rebuild(formula, spans, trial), rollouts)
                if f1 > best_f1:
                    best_f1, thresholds[i] = f1, t
    return _rebuild(formula, spans, thresholds)


# ── the LLM seam + the gate ────────────────────────────────────────────────────────────────────────────────
class ChatModel(Protocol):
    """The one seam to any LLM (mirrors akoire's ``ChatModel``). Returns a completion string."""

    def complete(self, system: str, prompt: str) -> str: ...


@dataclass
class ScriptedModel:
    """Deterministic ``ChatModel``: returns queued replies in order (test/bench, no network)."""

    replies: list[str]
    _i: int = field(default=0)

    def complete(self, system: str, prompt: str) -> str:
        r = self.replies[min(self._i, len(self.replies) - 1)]
        self._i += 1
        return r


def _parse_or_repair(formula: str) -> "str | None":
    """Return a parse-valid form of ``formula`` (as-is, else :func:`_repair`-ed), or ``None`` if neither parses."""
    for cand in (formula, _repair(formula)):
        try:
            parse(cand)
            return cand
        except Exception:
            continue
    return None


def _extract_formula(text: str) -> str:
    """Strip code fences / a leading label and take the first non-empty line (akoire's ``extract_hymeko`` shape)."""
    body = text.replace("```htl", "").replace("```", "").strip()
    for line in body.splitlines():
        line = line.strip()
        if line and not line.lower().startswith(("here", "the ", "formula", "htl", "answer")):
            return line
    return body.strip()


@dataclass(frozen=True)
class GateResult:
    formula: str | None            # the selected parse-valid formula (None if none parsed)
    candidates: list[str]          # parse-valid candidates collected
    n_attempts: int                # total model calls (incl. error-loop retries)
    parsed_any: bool


def propose_and_gate(model: ChatModel, base_prompt: str, verif: list[Rollout], *, k: int = 3, retries: int = 2,
                     system: str = "", calibrate: bool = True, prune: bool = True) -> GateResult:
    """Propose ``k`` HTL formulas; **parse-gate** each (repair + error-loop); **refine** each — ``prune`` reduces
    conjunct-selection to a ``hymeko_pgraph`` SSG solution (drop noise conjuncts) and ``calibrate`` fits the
    thresholds to the verification split; then **faithfulness-select** the best on verification.

    # Postconditions returns the max-verification-F1 refined candidate, or ``None`` if nothing parsed;
      ``n_attempts`` counts every model call so the raw parse-success rate is measurable."""
    candidates: list[str] = []
    attempts = 0
    for _ in range(k):
        prompt = base_prompt
        for _r in range(retries + 1):
            attempts += 1
            raw = _extract_formula(model.complete(system, prompt))
            valid = _parse_or_repair(raw)                   # gate's tolerant front-end: raw, then repaired
            if valid is not None:
                candidates.append(valid)
                break
            prompt = (f"{base_prompt}\n\nYour previous attempt {raw!r} was not valid HTL. Use round parentheses "
                      f"F(...)/G(...), == for equality, uppercase AND/OR/NOT. Output one formula only. Fix it.")
    if not candidates:
        return GateResult(formula=None, candidates=[], n_attempts=attempts, parsed_any=False)
    if prune:
        from hymeko_rl.eval.spec_bench.pgraph_refine import refine_via_pgraph   # late: avoids import cycle
        refined = [refine_via_pgraph(c, verif, calibrate=calibrate) for c in candidates]
    else:
        refined = [calibrate_thresholds(c, verif) if calibrate else c for c in candidates]
    best = max(refined, key=lambda f: formula_f1(f, verif))
    return GateResult(formula=best, candidates=candidates, n_attempts=attempts, parsed_any=True)


def score_raw(model: ChatModel, base_prompt: str, *, k: int = 3, system: str = "") -> "tuple[str | None, int, int]":
    """Raw arm: sample ``k`` formulas with NO faithfulness-select; return (first parse-valid | None, n_valid,
    n_attempts). The raw arm still needs a parse-valid formula to be gradable — how often it isn't is a result."""
    valid: list[str] = []
    for _ in range(k):
        formula = _extract_formula(model.complete(system, base_prompt))
        try:
            parse(formula)
            valid.append(formula)
        except Exception:
            pass
    return (valid[0] if valid else None), len(valid), k
