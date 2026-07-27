"""CIP-0 conformance suite --- the ten guarantees of the control profile.

Run: ``.venv/bin/python -m pytest -p no:randomly hymeko_control/conformance/tests``.
Each ``test_NN_*`` maps 1:1 to a numbered CIP-0 conformance requirement; the
leading ``test_00`` is a positive full-lifecycle integration.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from pathlib import Path

import pytest

import hymeko_control
from hymeko_control.cip.authority import (
    AuthorityChannel,
    AuthorityMap,
    AuthorityProvenanceError,
    AuthoritySource,
)
from hymeko_control.cip.certificate import Certificate, CertificateSuite
from hymeko_control.cip.option import (
    AffineAuthorityDecoder,
    OptionEnd,
    ResponseTrace,
)
from hymeko_control.cip.physical_intent import IntentBoundsError, PhysicalIntent
from hymeko_control.cip.runtime import (
    CausalityError,
    CIP0Runtime,
    DeterminismError,
    ModeError,
    ProvenanceError,
)
from hymeko_control.cip.structured_state import ControlState
from hymeko_control.conformance import battery
from hymeko_control.conformance.toy import ToyReachAdapter, toy_model, toy_reach_spec
from hymeko_control.language.schema_v0 import CertificateKind


# --------------------------------------------------------------------------
# 0. positive integration: the toy adapter drives a certified HOLD
# --------------------------------------------------------------------------
def test_00_full_lifecycle_reaches_certified_hold() -> None:
    model = toy_model()
    records = battery.run_positive_lifecycle(model, ToyReachAdapter(), max_ticks=40)
    last = records[-1]
    assert last.mode == "HOLD"
    assert last.next_mode == "HOLD"
    assert last.certificate.passed
    assert last.trace.end in (OptionEnd.COMPLETED, OptionEnd.HANDOFF)


# --------------------------------------------------------------------------
# 1. schema validation
# --------------------------------------------------------------------------
def test_01_schema_validation() -> None:
    battery.assert_schema_accepts(toy_reach_spec())

    missing = toy_reach_spec()
    del missing["physics"]
    battery.assert_schema_rejects(missing, "missing required section")

    bad_kind = toy_reach_spec()
    bad_kind["entities"][0]["kind"] = "warp_core"
    battery.assert_schema_rejects(bad_kind, "unknown entity kind")

    dangling = toy_reach_spec()
    dangling["ports"][0]["owner"] = "nonexistent"
    battery.assert_schema_rejects(dangling, "dangling port owner")

    bad_trans = toy_reach_spec()
    bad_trans["hybrid"]["transitions"][0]["dest"] = "GHOST"
    battery.assert_schema_rejects(bad_trans, "transition to undeclared mode")

    inverted = toy_reach_spec()
    inverted["task"]["intents"][0]["lower"] = 5.0
    battery.assert_schema_rejects(inverted, "inverted intent bounds")

    no_success = toy_reach_spec()
    no_success["task"]["certificates"] = [
        {"name": "safe", "kind": "safety", "predicate": "x>=0"}
    ]
    battery.assert_schema_rejects(no_success, "no success certificate")


# --------------------------------------------------------------------------
# 2. causal observation (tick index non-decreasing)
# --------------------------------------------------------------------------
def test_02_causal_observation() -> None:
    rt = CIP0Runtime(model=toy_model(), adapter=ToyReachAdapter())
    rt.tick()  # first tick sets _last_t causally
    good = ControlState(t=rt._last_t + 1, phase="APPROACH", signals={"dist": 0.5})
    rt._check_causal(good)  # forward is fine
    regressed = ControlState(t=0, phase="APPROACH", signals={"dist": 0.5})
    with pytest.raises(CausalityError):
        rt._check_causal(regressed)


# --------------------------------------------------------------------------
# 3. legal mode transitions
# --------------------------------------------------------------------------
class _IllegalTransitionAdapter(ToyReachAdapter):
    def transition(self, mode, certificate):  # type: ignore[override]
        return "HOLD"  # APPROACH -> HOLD is not a declared transition


def test_03_legal_mode_transitions() -> None:
    model = toy_model()
    assert model.is_legal_transition("APPROACH", "TOUCH")
    assert model.is_legal_transition("HOLD", "HOLD")  # self-loop
    assert not model.is_legal_transition("APPROACH", "HOLD")
    rt = CIP0Runtime(model=model, adapter=_IllegalTransitionAdapter())
    with pytest.raises(ModeError):
        rt.tick()


# --------------------------------------------------------------------------
# 4. bounded physical intent
# --------------------------------------------------------------------------
def test_04_bounded_physical_intent() -> None:
    bounds = {"v": (0.0, 1.0)}
    ok = PhysicalIntent(components={"v": 0.5}, bounds=bounds)
    assert ok.is_bounded()
    with pytest.raises(IntentBoundsError):
        PhysicalIntent(components={"v": 2.0}, bounds=bounds)
    with pytest.raises(IntentBoundsError):
        PhysicalIntent(components={"v": 0.5}, bounds={"v": (1.0, 0.0)})  # inverted
    clipped = PhysicalIntent.clipped(components={"v": 9.0}, bounds=bounds)
    assert clipped.is_bounded() and clipped.get("v") == 1.0


# --------------------------------------------------------------------------
# 5. authority provenance
# --------------------------------------------------------------------------
def test_05_authority_provenance() -> None:
    good = AuthorityMap(
        channels={
            "a": AuthorityChannel("a", 1.0, AuthoritySource.OBSERVED, "measured on rig")
        }
    )
    good.require_provenance()  # no raise
    with pytest.raises(AuthorityProvenanceError):
        AuthorityChannel("a", 1.0, AuthoritySource.OBSERVED, "")  # empty provenance


# --------------------------------------------------------------------------
# 6. deterministic decoding
# --------------------------------------------------------------------------
class _NonDeterministicAdapter(ToyReachAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    def decode(self, intent, authority):  # type: ignore[override]
        self._n += 1
        opt = super().decode(intent, authority)
        return dataclasses.replace(opt, command=(opt.command[0] + self._n,))


def test_06_deterministic_decoding() -> None:
    decoder = AffineAuthorityDecoder(
        name="d", order=("v",), gain=(0.5,), mode="APPROACH",
        initiation=frozenset({"APPROACH"}), termination="reached",
    )
    intent = PhysicalIntent(components={"v": 0.8}, bounds={"v": (0.0, 1.0)})
    auth = AuthorityMap(
        channels={"v": AuthorityChannel("v", 1.0, AuthoritySource.MODELED, "unit")}
    )
    o1 = decoder.decode(intent, auth)
    o2 = decoder.decode(intent, auth)
    assert (o1.command, o1.name, o1.mode) == (o2.command, o2.name, o2.mode)
    rt = CIP0Runtime(model=toy_model(), adapter=_NonDeterministicAdapter())
    with pytest.raises(DeterminismError):
        rt.tick()


# --------------------------------------------------------------------------
# 7. option execution provenance
# --------------------------------------------------------------------------
class _BadProvenanceAdapter(ToyReachAdapter):
    def execute(self, option):  # type: ignore[override]
        return ResponseTrace(
            option=option.name,
            commands=(option.command,),
            signals=({"dist": self.dist},),
            end=OptionEnd.HANDOFF,
            provenance={"option": "SOMETHING_ELSE"},  # breaks the link
        )


def test_07_option_execution_provenance() -> None:
    model = toy_model()
    records = battery.run_positive_lifecycle(model, ToyReachAdapter(), max_ticks=40)
    for rec in records:
        assert rec.trace.references(rec.option)
    rt = CIP0Runtime(model=model, adapter=_BadProvenanceAdapter())
    with pytest.raises(ProvenanceError):
        rt.tick()


# --------------------------------------------------------------------------
# 8. certificate independence from reward
# --------------------------------------------------------------------------
def test_08_certificate_independent_of_reward() -> None:
    # signature carries no reward parameter
    for fn in (Certificate.evaluate, CertificateSuite.evaluate):
        params = list(inspect.signature(fn).parameters)
        assert params == ["self", "state", "trace"], params
    # the certificate module never uses reward in code (docstrings may DISCUSS it)
    import ast

    tree = ast.parse(Path(inspect.getfile(Certificate)).read_text(encoding="utf-8"))
    code_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    arg_names = {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    assert "reward" not in code_names and "reward" not in arg_names
    # evaluation is reproducible from (state, trace) alone
    suite = CertificateSuite(
        certificates=(
            Certificate("s", CertificateKind.SUCCESS, lambda s, t: s.signal("d") <= 0.1),
            Certificate("safe", CertificateKind.SAFETY, lambda s, t: True),
        )
    )
    state = ControlState(t=0, phase="X", signals={"d": 0.05})
    trace = ResponseTrace("o", ((0.0,),), ({"d": 0.05},), OptionEnd.COMPLETED,
                          {"option": "o"})
    r1 = suite.evaluate(state, trace)
    r2 = suite.evaluate(state, trace)
    assert r1.passed and r1.passed == r2.passed


# --------------------------------------------------------------------------
# 9. no hidden state modification
# --------------------------------------------------------------------------
def test_09_no_hidden_state_modification() -> None:
    state = ControlState(t=3, phase="APPROACH", signals={"dist": 0.4},
                         contact={"tip": False})
    before = copy.deepcopy(dict(state.signals))
    # frozen dataclass: attribute reassignment forbidden
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.t = 9  # type: ignore[misc]
    # mappings are read-only views
    with pytest.raises(TypeError):
        state.signals["dist"] = 99.0  # type: ignore[index]
    # a full run does not mutate observed signals
    battery.run_positive_lifecycle(toy_model(), ToyReachAdapter(), max_ticks=40)
    assert dict(state.signals) == before


# --------------------------------------------------------------------------
# 10. shared-core import isolation (no torch, no scenario, no hymeko_rl)
# --------------------------------------------------------------------------
def test_10_shared_core_import_isolation() -> None:
    package_dir = Path(hymeko_control.__file__).resolve().parent
    violations = battery.import_isolation_violations(package_dir)
    assert violations == [], f"core import isolation broken: {violations}"
