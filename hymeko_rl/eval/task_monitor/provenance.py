"""Policy-provenance guard — the identity sibling of the tensor-contract guard.

:class:`~hymeko_rl.eval.task_monitor.pipeline.PipelineSchemaLedger` proves the *tensors* have the right shape and
field order; it does NOT prove the *policy* is the one the run claims. A scripted controller, a wrong checkpoint,
or a stale anchor can all pass the schema guard while being the wrong policy — exactly the point-5 anchor bug
(a SCRIPTED expert silently used where the frozen DAgger actor was specified) that a schema check cannot catch.

`PolicyProvenanceLedger` records, for every policy entering a run, its checkpoint path + md5, architecture, seed,
DAgger stage, role, post-load parameter hash, and action checksum on a fixed canonical observation batch; and
asserts the identities the run depends on:

1. a checkpoint claimed "= frozen selected DAgger" has the frozen file md5;
2. an anchor claimed "= frozen DAgger actor" has the selected checkpoint's parameter hash;
3. at RL init, actor and anchor parameter hashes both equal the selected checkpoint's, and their action MSE on the
   fixed obs batch is ~0 (they really are the same policy);
4. a scripted controller / teacher wrapper is NEVER accepted as a learned actor checkpoint (structurally: a
   non-module has no ``state_dict`` and is rejected at registration).

Queue the expectations with ``expect_*`` and run them at RL initialization via ``verify_or_abort`` (wired into
``train_offpolicy(provenance=…)``), or call the ``assert_*`` methods directly in a launch harness.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from .submonitors import SubVerdict


class PolicyProvenanceError(RuntimeError):
    """Raised when a policy's identity/provenance does not match what the run claims."""


class PolicyRole(Enum):
    BC0 = "bc0"
    DAGGER_VAL_SELECTED = "dagger_val_selected"
    DAGGER_ORACLE_BEST = "dagger_oracle_best"
    SCRIPTED_TEACHER = "scripted_teacher"
    RL_ACTOR = "rl_actor"


LEARNED_ROLES = frozenset({PolicyRole.BC0, PolicyRole.DAGGER_VAL_SELECTED,
                           PolicyRole.DAGGER_ORACLE_BEST, PolicyRole.RL_ACTOR})


def file_md5(path: str) -> str:
    """md5 of a checkpoint file's bytes (identifies the file on disk)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_hash(module_or_state_dict: Any) -> str:
    """Order-independent md5 over a module's parameters (identifies the in-memory policy after load)."""
    sd = module_or_state_dict.state_dict() if hasattr(module_or_state_dict, "state_dict") else module_or_state_dict
    h = hashlib.md5()
    for k in sorted(sd):
        t = sd[k]
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        h.update(k.encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def action_checksum(actions: np.ndarray, *, decimals: int = 5) -> str:
    """md5 of a policy's action outputs on the fixed obs batch (a behavioural fingerprint)."""
    a = np.round(np.asarray(actions, np.float64), decimals)
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def canonical_obs_batch(make_env: Callable[[], Any], n: int = 32, seed0: int = 12345) -> np.ndarray:
    """A fixed, deterministic observation batch (n reset observations from fixed seeds) for action checksums/MSE."""
    env = make_env()
    obs = np.stack([np.asarray(env.reset(seed=seed0 + i)[0], np.float32) for i in range(n)])
    env.close()
    return obs


@dataclass(frozen=True)
class PolicyProvenance:
    """The recorded identity of one policy entering a run."""

    name: str
    role: PolicyRole
    is_scripted: bool
    arch: str
    checkpoint_path: "str | None" = None
    file_md5: "str | None" = None
    seed: "int | None" = None
    dagger_stage: "str | None" = None
    param_hash: "str | None" = None
    action_checksum: "str | None" = None

    def as_dict(self) -> dict:
        return {"name": self.name, "role": self.role.value, "is_scripted": self.is_scripted, "arch": self.arch,
                "checkpoint_path": self.checkpoint_path, "file_md5": self.file_md5, "seed": self.seed,
                "dagger_stage": self.dagger_stage, "param_hash": self.param_hash,
                "action_checksum": self.action_checksum}


class PolicyProvenanceLedger:
    """Records + validates policy identity; aborts when a run's policy is not what it claims."""

    def __init__(self, obs_batch: np.ndarray) -> None:
        self.obs_batch = np.asarray(obs_batch, np.float32)
        self.records: dict[str, PolicyProvenance] = {}
        self._actions: dict[str, np.ndarray] = {}      # cached action outputs on obs_batch (for MSE)
        self._checks: list[Callable[[], None]] = []    # queued assertions for verify_or_abort

    def _req(self, name: str) -> PolicyProvenance:
        if name not in self.records:
            raise PolicyProvenanceError(f"no policy registered under {name!r}")
        return self.records[name]

    def _actor_actions(self, actor: Any) -> np.ndarray:
        import torch
        fn = actor.action_mean if hasattr(actor, "action_mean") else actor
        with torch.no_grad():
            out = fn(torch.as_tensor(self.obs_batch, dtype=torch.float32))
        return out.detach().cpu().numpy()

    def register_checkpoint(self, name: str, role: PolicyRole, path: "str | None", actor: Any, *,
                            arch: str, seed: "int | None" = None,
                            dagger_stage: "str | None" = None) -> PolicyProvenance:
        """Register a LEARNED actor loaded from a checkpoint. Rejects a non-module (a scripted controller cannot
        masquerade as a learned checkpoint — assertion 4, enforced structurally)."""
        if role not in LEARNED_ROLES:
            raise PolicyProvenanceError(f"{role} is not a learned role; use register_scripted for teachers")
        if not hasattr(actor, "state_dict"):
            raise PolicyProvenanceError(
                f"{name}: not a learned checkpoint (no state_dict) — a scripted controller / teacher wrapper "
                "cannot be registered as a learned actor")
        acts = self._actor_actions(actor)
        rec = PolicyProvenance(
            name=name, role=role, is_scripted=False, arch=arch,
            checkpoint_path=str(path) if path is not None else None,
            file_md5=file_md5(path) if path is not None else None,
            seed=seed, dagger_stage=dagger_stage,
            param_hash=param_hash(actor), action_checksum=action_checksum(acts))
        self.records[name] = rec
        self._actions[name] = acts
        return rec

    def register_scripted(self, name: str, *, arch: str = "scripted") -> PolicyProvenance:
        """Register a scripted teacher/controller — recorded WITHOUT a parameter hash (it is not an obs→action
        module) and never usable in a learned role."""
        rec = PolicyProvenance(name=name, role=PolicyRole.SCRIPTED_TEACHER, is_scripted=True, arch=arch)
        self.records[name] = rec
        return rec

    # --- assertions (raise PolicyProvenanceError on failure) ---
    def assert_checkpoint_matches(self, name: str, expected_md5: str) -> None:
        r = self._req(name)
        if r.file_md5 != expected_md5:
            raise PolicyProvenanceError(f"{name} checkpoint md5 {r.file_md5} != expected frozen {expected_md5}")

    def assert_learned_not_scripted(self, name: str) -> None:
        r = self._req(name)
        if r.is_scripted or r.role not in LEARNED_ROLES or r.param_hash is None:
            raise PolicyProvenanceError(f"{name} is not a valid learned actor (scripted / no parameters)")

    def action_mse(self, a_name: str, b_name: str) -> float:
        if a_name not in self._actions or b_name not in self._actions:
            raise PolicyProvenanceError(f"action MSE needs two learned policies; have {list(self._actions)}")
        return float(np.mean((self._actions[a_name] - self._actions[b_name]) ** 2))

    def assert_rl_init(self, actor_name: str, anchor_name: str, selected_name: str, *, atol: float = 1e-6) -> None:
        """Assertion 3: actor and anchor parameter hashes both equal the selected checkpoint's, and their action
        MSE on the fixed obs batch is ~0."""
        sel = self._req(selected_name)
        for nm in (actor_name, anchor_name):
            r = self._req(nm)
            if r.param_hash != sel.param_hash:
                raise PolicyProvenanceError(
                    f"{nm} param hash {r.param_hash} != selected {selected_name} {sel.param_hash}")
        mse = self.action_mse(actor_name, anchor_name)
        if mse > atol:
            raise PolicyProvenanceError(
                f"actor-vs-anchor action MSE {mse:.3g} > {atol} — not identical at init")

    # --- queued-expectation API (run at RL init by verify_or_abort / train_offpolicy) ---
    def expect_checkpoint_matches(self, name: str, expected_md5: str) -> None:
        self._checks.append(lambda: self.assert_checkpoint_matches(name, expected_md5))

    def expect_learned_not_scripted(self, name: str) -> None:
        self._checks.append(lambda: self.assert_learned_not_scripted(name))

    def expect_rl_init(self, actor_name: str, anchor_name: str, selected_name: str, *, atol: float = 1e-6) -> None:
        self._checks.append(lambda: self.assert_rl_init(actor_name, anchor_name, selected_name, atol=atol))

    def verify(self) -> SubVerdict:
        violations: list[str] = []
        for chk in self._checks:
            try:
                chk()
            except PolicyProvenanceError as e:
                violations.append(str(e))
        return SubVerdict("policy_provenance", not violations, 1.0 if not violations else 0.0, violations, [], [])

    def verify_or_abort(self) -> SubVerdict:
        v = self.verify()
        if not v.passed:
            raise PolicyProvenanceError("policy provenance FAILED: " + "; ".join(v.violations))
        return v

    def report_fields(self, *, actor_name: str, anchor_name: "str | None" = None,
                      selected_name: "str | None" = None,
                      reward_file: "str | None" = None, env_file: "str | None" = None) -> dict:
        """Assertion-5 report block: provenance pass/fail + checkpoint hashes + selected DAgger stage + files."""
        a = self._req(actor_name)
        anchor = self._req(anchor_name) if anchor_name else None
        selected = self._req(selected_name) if selected_name else a
        return {
            "policy_provenance": "PASS" if self.verify().passed else "FAIL",
            "actor_checkpoint_hash": a.file_md5, "actor_param_hash": a.param_hash,
            "anchor_checkpoint_hash": anchor.file_md5 if anchor else None,
            "anchor_param_hash": anchor.param_hash if anchor else None,
            "selected_dagger_stage": selected.dagger_stage, "dagger_stage": a.dagger_stage,
            "arch": a.arch, "seed": a.seed, "reward_file": reward_file, "env_file": env_file,
        }
