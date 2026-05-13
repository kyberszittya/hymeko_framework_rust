"""KONECT-backed signed graphs (wikisigned, wiki_elec, wiki_conflict).

Tests avoid hitting konect.cc by either pre-filling ``<name>.txt`` under a
temporary ``DATA_DIR`` (``download`` returns immediately) or by mocking
``urlopen`` to return a minimal ``tar.bz2`` payload.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from signedkan_wip.src import datasets


def _tar_bz2_with_out_file(inner_relpath: str, body: str) -> bytes:
    buf = io.BytesIO()
    raw = body.encode("utf-8")
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        info = tarfile.TarInfo(name=inner_relpath)
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path)
    return tmp_path


def test_konect_wikisigned_load_prefilled_txt(isolated_data_dir: Path) -> None:
    (isolated_data_dir / "wikisigned.txt").write_text(
        "% src tgt sign\n"
        "100 200 1\n"
        "200 300 -1\n"
        "300 100 0.5\n",
        encoding="utf-8",
    )
    g = datasets.load("wikisigned")
    assert g.n_nodes == 3
    assert g.edges.shape[0] == 3
    assert set(int(x) for x in g.signs) == {-1, 1}


def test_konect_wiki_elec_load_prefilled_txt(isolated_data_dir: Path) -> None:
    (isolated_data_dir / "wiki_elec.txt").write_text(
        "%\n"
        "1 2 -1\n"
        "2 3 1\n",
        encoding="utf-8",
    )
    g = datasets.load("wiki_elec")
    assert g.n_nodes == 3
    assert g.edges.shape[0] == 2


def test_konect_wiki_conflict_majority_dedup(isolated_data_dir: Path) -> None:
    # Same undirected pair (0,1) after remap: four parallel ratings, majority +1.
    (isolated_data_dir / "wiki_conflict.txt").write_text(
        "%\n"
        "1 2 0.7\n"
        "1 2 0.2\n"
        "2 1 -0.1\n"
        "2 1 -0.2\n"
        "3 4 -5\n",
        encoding="utf-8",
    )
    g = datasets.load("wiki_conflict")
    assert g.n_nodes == 4
    # Pairs (0,1) and (2,3) only after dedup.
    assert g.edges.shape[0] == 2
    u01 = tuple(sorted((int(g.edges[0, 0]), int(g.edges[0, 1]))))
    u23 = tuple(sorted((int(g.edges[1, 0]), int(g.edges[1, 1]))))
    assert {u01, u23} == {(0, 1), (2, 3)}
    signs_by_pair = {
        tuple(sorted((int(g.edges[i, 0]), int(g.edges[i, 1])))): int(g.signs[i])
        for i in range(2)
    }
    assert signs_by_pair[(0, 1)] == 1
    assert signs_by_pair[(2, 3)] == -1


@patch("signedkan_wip.src.datasets.urllib.request.urlopen")
def test_konect_download_extracts_wikisigned_tar(
    mock_urlopen: object,
    isolated_data_dir: Path,
) -> None:
    txt_path = isolated_data_dir / "wikisigned.txt"
    if txt_path.exists():
        txt_path.unlink()

    payload = _tar_bz2_with_out_file(
        "wikisigned-k2/out.wikisigned-k2",
        "%\n5 6 1\n6 7 -2\n",
    )

    mock_cm = MagicMock()
    mock_body = MagicMock()
    mock_body.read.return_value = payload
    mock_cm.__enter__.return_value = mock_body
    mock_cm.__exit__.return_value = None
    mock_urlopen.return_value = mock_cm
    out = datasets.download("wikisigned")
    assert out == txt_path
    assert txt_path.is_file()
    text = txt_path.read_text(encoding="utf-8")
    assert "5" in text and "6" in text

    g = datasets.load("wikisigned")
    assert g.n_nodes == 3
    assert g.edges.shape[0] == 2


def test_gomb_smoke_cpu_one_epoch_wikisigned_prefilled(
    isolated_data_dir: Path,
) -> None:
    """End-to-end smoke: tiny KONECT-style file + short Gömb run on CPU."""
    import os
    import subprocess
    import sys

    (isolated_data_dir / "wikisigned.txt").write_text(
        "%\n"
        + "\n".join(
            f"{i} {i + 1} {1 if i % 2 == 0 else -1}"
            for i in range(1, 25)
        )
        + "\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-m",
        "signedkan_wip.src.run_gomb_smoke",
        "--dataset",
        "wikisigned",
        "--seed",
        "0",
        "--n-epochs",
        "1",
        "--device",
        "cpu",
        "--edge-split",
        "80_20",
        "--lr",
        "0.01",
        "--d-embed",
        "8",
        "--d-outer",
        "4",
        "--M-outer",
        "2",
        "--d-middle",
        "8",
        "--d-core",
        "8",
        "--topk",
        "8",
        "--n-tiers",
        "2",
        "--weight-decay",
        "0",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert '"dataset": "wikisigned"' in proc.stdout
    assert '"n_params"' in proc.stdout


def test_split_konect_prefilled_deterministic(isolated_data_dir: Path) -> None:
    (isolated_data_dir / "wikisigned.txt").write_text(
        "%\n" + "\n".join(f"{i} {i+1} 1" for i in range(50)) + "\n",
        encoding="utf-8",
    )
    g = datasets.load("wikisigned")
    tr, va, te = datasets.split(g, seed=11)
    tr2, va2, te2 = datasets.split(g, seed=11)
    assert np.array_equal(tr, tr2)
    assert len(tr) + len(va) + len(te) == g.edges.shape[0]
