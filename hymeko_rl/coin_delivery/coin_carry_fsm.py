"""HyMeKo-driven carry-option automaton — the executed control TOPOLOGY is sourced from
``data/robotics/coin_carry_option_v1.hymeko`` (parsed by the framework's `ControllerSpec`), not hard-coded in the executor.

This closes the dual-source-of-truth: `execute_one_option` walks THIS automaton for its phase transitions; Python binds only
the named guard predicates (evaluated on the live sim state) and the phase laws (→ push/brake/release amplitude, or the
frozen settling policy). Editing the phase graph is an edit to the `.hymeko`, not to the executor. The equivalence of the
automaton-driven and the (legacy) hard-coded topology is enforced by `tests/test_coin_carry_fsm.py`.
"""
from functools import lru_cache

from hymeko_rl.control.controller_spec import ControllerSpec

CARRY_OPTION_HYMEKO = "data/robotics/coin_carry_option_v1.hymeko"

# Guard events the coin backend evaluates on the live sim state (bound in execute_one_option):
CARRY_GUARDS = ("handoff", "abort", "push_reached", "push_timeout", "brake_centered", "brake_slow",
                "brake_timeout", "release_done", "delivered", "settle_horizon")
# Phase law → backend action binding (the executor maps these to θ amplitudes / the frozen settling policy):
CARRY_LAWS = {"push_amplitude": "a_push", "brake_amplitude": "a_brake", "release_amplitude": "a_release",
              "frozen_settling": "pi0_settle", "terminal": "noop"}
MACRO_PHASES = ("PUSH", "BRAKE", "RELEASE")
TERMINAL_MARKS = ("HANDOFF", "COMPLETED", "ABORTED", "DELIVERED", "SETTLED")


@lru_cache(maxsize=2)
def load_carry_automaton(path: str = CARRY_OPTION_HYMEKO) -> ControllerSpec:
    """Parse (and cache) the carry-option FSM. # Postcondition: initial phase is PUSH; every macro phase binds an
    amplitude law and routes handoff→HANDOFF and abort→ABORTED."""
    spec = ControllerSpec.from_hymeko(path)
    assert spec.initial == "PUSH", f"carry automaton must start in PUSH, got {spec.initial}"
    return spec
