"""Registry conformance for the multi-embodiment CIP-0 surface (scenarios.registry).

Torch-free by construction: this suite loads only declarative ControlModels, never an adapter/env, so it runs without
mujoco/torch and additionally *asserts* that discovery + model loading pull in neither.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hymeko_control.language.ir import ControlModel
from scenarios.registry import EmbodimentEntry, EmbodimentRegistry, EmbodimentStatus

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lists_every_embodiment() -> None:
    names = EmbodimentRegistry.list_embodiments()
    assert set(names) == {"pick_place", "aibo", "humanoid", "coin"}
    # registration order is stable
    assert names == ("pick_place", "aibo", "humanoid", "coin")


def test_get_resolves_by_short_name_and_scenario_id() -> None:
    by_name = EmbodimentRegistry.get("aibo")
    by_id = EmbodimentRegistry.get("CIP-AIBO-01")
    assert by_name is by_id
    assert by_name.scenario_id == "CIP-AIBO-01"


def test_get_unknown_raises_keyerror_listing_known() -> None:
    with pytest.raises(KeyError) as exc:
        EmbodimentRegistry.get("nonexistent")
    assert "nonexistent" in str(exc.value)


def test_integrated_includes_coin_cip00() -> None:
    # CIP-COIN-00 is now a registered consumer (adapter in the coin repo, delegates to the deployed runtime).
    integrated = {e.embodiment for e in EmbodimentRegistry.integrated()}
    assert integrated == {"pick_place", "aibo", "humanoid", "coin"}
    assert EmbodimentRegistry.get("coin").load_model is not None      # no longer PENDING


def test_certified_is_only_tagged_pick_place() -> None:
    certified = EmbodimentRegistry.certified()
    assert [e.embodiment for e in certified] == ["pick_place"]
    assert certified[0].cip_tag == "cip-pick-place-v0"


@pytest.mark.parametrize("name", ["pick_place", "aibo", "humanoid"])
def test_integrated_entry_loads_a_valid_control_model(name: str) -> None:
    model = EmbodimentRegistry.get(name).model()
    assert isinstance(model, ControlModel)


def test_coin_cip00_registered_resolves_to_adapter() -> None:
    # CIP-COIN-00 resolves through the registry to a loadable (lazy) model; its runtime lives in a separate repo, so
    # model() import may not be available in this tree's env — the REGISTRATION (id, loader, status) is what's asserted.
    coin = EmbodimentRegistry.get("coin")
    assert coin.scenario_id == "CIP-COIN-00"
    assert coin.status is EmbodimentStatus.PRESENT_UNTAGGED
    assert callable(coin.load_model)


def test_entry_invariant_tag_iff_certified() -> None:
    # a non-certified entry may not carry a tag
    with pytest.raises(AssertionError):
        EmbodimentEntry(
            embodiment="bad",
            scenario_id="CIP-BAD-00",
            title="bad",
            status=EmbodimentStatus.PRESENT_UNTAGGED,
            measured="x",
            load_model=lambda: None,  # type: ignore[return-value]
            cip_tag="cip-bad-v0",
        )


def test_entry_invariant_pending_has_no_loader() -> None:
    # a PENDING entry may not carry a loader
    with pytest.raises(AssertionError):
        EmbodimentEntry(
            embodiment="bad",
            scenario_id="CIP-BAD-00",
            title="bad",
            status=EmbodimentStatus.PENDING,
            measured="x",
            load_model=lambda: None,  # type: ignore[return-value]
        )


def test_discovery_and_model_loading_are_torch_free() -> None:
    # Order-independent isolation: in a CLEAN interpreter, importing the registry and loading every
    # integrated model must pull in neither torch nor mujoco. A subprocess is required because a
    # co-collected mujoco conformance test would otherwise have already loaded torch into this process.
    script = (
        "import sys\n"
        "from scenarios.registry import EmbodimentRegistry\n"
        # load every integrated model owned by THIS framework tree; coin's runtime is a separate repo (hymeko/rl) whose\n"
        # loader is not part of the framework's torch-free surface, so it is excluded from this framework-scoped check.\n"
        "for e in EmbodimentRegistry.integrated():\n"
        "    if e.embodiment == 'coin': continue\n"
        "    e.model()\n"
        "assert 'torch' not in sys.modules, 'registry loaded torch'\n"
        "assert 'mujoco' not in sys.modules, 'registry loaded mujoco'\n"
        "print('CLEAN')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"torch-free isolation failed: {proc.stderr}"
    assert "CLEAN" in proc.stdout
