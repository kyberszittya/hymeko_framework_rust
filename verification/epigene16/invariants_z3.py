"""Z3 starter checks for EPIGENE-16 finite-bucket safety invariants.

We model the committed profile as sixteen 4-bit buckets, represented by Ints in
0..15. This is deliberately smaller than the eventual runtime. The point is to
pin down the proof style:

* positive theorems are proved by asserting the invariant system and asking Z3
  for a counterexample to the desired property; UNSAT means proved;
* negative/load-bearing checks remove one architectural guard and ask Z3 to find
  an unsafe model; SAT means the guard is necessary.
"""
from __future__ import annotations

from dataclasses import dataclass

import z3


LOW = 4
MEDIUM = 8
HIGH = 12
MAX_BUCKET = 15


@dataclass(frozen=True)
class EpigeneVars:
    expression_gain: z3.ArithRef
    safety_margin: z3.ArithRef
    exploration_budget: z3.ArithRef
    repair_priority: z3.ArithRef
    reuse_bias: z3.ArithRef
    mutation_temperature: z3.ArithRef
    transaction_strictness: z3.ArithRef
    monitor_sensitivity: z3.ArithRef
    phase_adherence: z3.ArithRef
    contact_conservatism: z3.ArithRef
    baseline_guard: z3.ArithRef
    credit_assignment_scope: z3.ArithRef
    llm_authority: z3.ArithRef
    formalization_pressure: z3.ArithRef
    evidence_requirement: z3.ArithRef
    rollback_readiness: z3.ArithRef
    contact_critical: z3.BoolRef
    monitor_attached: z3.BoolRef

    def channels(self) -> list[z3.ArithRef]:
        return [
            self.expression_gain,
            self.safety_margin,
            self.exploration_budget,
            self.repair_priority,
            self.reuse_bias,
            self.mutation_temperature,
            self.transaction_strictness,
            self.monitor_sensitivity,
            self.phase_adherence,
            self.contact_conservatism,
            self.baseline_guard,
            self.credit_assignment_scope,
            self.llm_authority,
            self.formalization_pressure,
            self.evidence_requirement,
            self.rollback_readiness,
        ]


def vars(prefix: str = "e") -> EpigeneVars:
    names = [
        "expression_gain",
        "safety_margin",
        "exploration_budget",
        "repair_priority",
        "reuse_bias",
        "mutation_temperature",
        "transaction_strictness",
        "monitor_sensitivity",
        "phase_adherence",
        "contact_conservatism",
        "baseline_guard",
        "credit_assignment_scope",
        "llm_authority",
        "formalization_pressure",
        "evidence_requirement",
        "rollback_readiness",
    ]
    ints = {name: z3.Int(f"{prefix}_{name}") for name in names}
    return EpigeneVars(
        **ints,
        contact_critical=z3.Bool(f"{prefix}_contact_critical"),
        monitor_attached=z3.Bool(f"{prefix}_monitor_attached"),
    )


def bucket_bounds(e: EpigeneVars) -> z3.BoolRef:
    return z3.And([z3.And(ch >= 0, ch <= MAX_BUCKET) for ch in e.channels()])


def core_invariants(e: EpigeneVars) -> z3.BoolRef:
    return z3.And(
        bucket_bounds(e),
        e.llm_authority <= e.transaction_strictness,
        z3.Implies(e.expression_gain >= HIGH, e.evidence_requirement >= HIGH),
        z3.Implies(e.mutation_temperature >= MEDIUM, e.rollback_readiness >= MEDIUM),
        z3.Implies(e.monitor_sensitivity >= HIGH, e.monitor_attached),
        z3.Implies(
            e.contact_critical,
            z3.And(
                e.phase_adherence >= HIGH,
                e.contact_conservatism >= HIGH,
                e.baseline_guard >= HIGH,
            ),
        ),
    )


def theorem_authority_cannot_bypass_transaction() -> bool:
    e = vars("auth")
    s = z3.Solver()
    s.add(core_invariants(e))
    s.add(e.llm_authority > e.transaction_strictness)
    result = s.check()
    proved = result == z3.unsat
    print(f"T1 authority <= transaction_strictness: negation is {result} -> {proved}")
    return proved


def theorem_contact_critical_implies_contact_guards() -> bool:
    e = vars("contact")
    s = z3.Solver()
    s.add(core_invariants(e))
    s.add(e.contact_critical)
    s.add(
        z3.Or(
            e.phase_adherence < HIGH,
            e.contact_conservatism < HIGH,
            e.baseline_guard < HIGH,
        )
    )
    result = s.check()
    proved = result == z3.unsat
    print(f"T2 contact-critical guard theorem: negation is {result} -> {proved}")
    return proved


def theorem_high_expression_requires_evidence() -> bool:
    e = vars("evidence")
    s = z3.Solver()
    s.add(core_invariants(e))
    s.add(e.expression_gain >= HIGH)
    s.add(e.evidence_requirement < HIGH)
    result = s.check()
    proved = result == z3.unsat
    print(f"T3 high expression requires evidence: negation is {result} -> {proved}")
    return proved


def negative_contact_guard_is_load_bearing() -> bool:
    """Without the contact-critical implication, Z3 should find an unsafe model."""
    e = vars("neg_contact")
    weakened = z3.And(
        bucket_bounds(e),
        e.llm_authority <= e.transaction_strictness,
        z3.Implies(e.expression_gain >= HIGH, e.evidence_requirement >= HIGH),
        z3.Implies(e.mutation_temperature >= MEDIUM, e.rollback_readiness >= MEDIUM),
        z3.Implies(e.monitor_sensitivity >= HIGH, e.monitor_attached),
    )
    unsafe_contact = z3.And(
        e.contact_critical,
        z3.Or(
            e.phase_adherence < HIGH,
            e.contact_conservatism < HIGH,
            e.baseline_guard < HIGH,
        ),
    )
    s = z3.Solver()
    s.add(weakened, unsafe_contact)
    result = s.check()
    found = result == z3.sat
    print(f"T4 negative: remove contact guard -> unsafe model is {result} -> {found}")
    if found:
        model = s.model()
        print(
            "    witness: "
            f"phase={model[e.phase_adherence]}, "
            f"contact={model[e.contact_conservatism]}, "
            f"baseline={model[e.baseline_guard]}"
        )
    return found


def negative_authority_guard_is_load_bearing() -> bool:
    """Without llm_authority <= transaction_strictness, Z3 should find bypass."""
    e = vars("neg_auth")
    weakened = z3.And(
        bucket_bounds(e),
        z3.Implies(e.expression_gain >= HIGH, e.evidence_requirement >= HIGH),
        z3.Implies(e.mutation_temperature >= MEDIUM, e.rollback_readiness >= MEDIUM),
        z3.Implies(e.monitor_sensitivity >= HIGH, e.monitor_attached),
        z3.Implies(
            e.contact_critical,
            z3.And(
                e.phase_adherence >= HIGH,
                e.contact_conservatism >= HIGH,
                e.baseline_guard >= HIGH,
            ),
        ),
    )
    s = z3.Solver()
    s.add(weakened, e.llm_authority > e.transaction_strictness)
    result = s.check()
    found = result == z3.sat
    print(f"T5 negative: remove authority guard -> bypass model is {result} -> {found}")
    if found:
        model = s.model()
        print(
            "    witness: "
            f"llm={model[e.llm_authority]}, "
            f"transaction={model[e.transaction_strictness]}"
        )
    return found


def run() -> bool:
    print("EPIGENE-16 Z3 invariant checks:\n")
    checks = [
        theorem_authority_cannot_bypass_transaction(),
        theorem_contact_critical_implies_contact_guards(),
        theorem_high_expression_requires_evidence(),
        negative_contact_guard_is_load_bearing(),
        negative_authority_guard_is_load_bearing(),
    ]
    ok = all(checks)
    print(f"\nEPIGENE-16 finite-bucket invariants verified: {ok}")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)

