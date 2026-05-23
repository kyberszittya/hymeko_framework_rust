"""Test the Reddit Hyperlinks loader against a fake TSV fixture.

Avoids network downloads — writes a synthetic 6-col TSV mimicking
the SNAP format into the data dir, then exercises the parser. The
real dataset is downloaded on first `load("reddit_body")` call;
production downloads are out of scope for unit tests.

Phase C of the Nature Comm submission plan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signedkan_wip.src.datasets import DATA_DIR, load


@pytest.fixture
def fake_reddit_tsv(tmp_path, monkeypatch):
    """Write a 5-edge synthetic Reddit Hyperlinks TSV.

    Format mirrors `soc-redditHyperlinks-{body,title}.tsv`:
      SOURCE_SUBREDDIT \t TARGET_SUBREDDIT \t POST_ID \t TIMESTAMP \t LINK_SENTIMENT \t PROPERTIES
    """
    tsv_path = tmp_path / "reddit_body.tsv"
    tsv_path.write_text(
        "SOURCE_SUBREDDIT\tTARGET_SUBREDDIT\tPOST_ID\tTIMESTAMP\tLINK_SENTIMENT\tPROPERTIES\n"
        "askscience\tphysics\tp1\t2017-01-01\t1\tx,y,z\n"
        "askreddit\tworldnews\tp2\t2017-01-02\t-1\tx,y,z\n"
        "physics\tmath\tp3\t2017-01-03\t1\tx,y,z\n"
        "worldnews\tpolitics\tp4\t2017-01-04\t-1\tx,y,z\n"
        "askscience\tmath\tp5\t2017-01-05\t1\tx,y,z\n"
        # Empty line + zero-sentiment row are skipped:
        "\n"
        "askreddit\tphysics\tp6\t2017-01-06\t0\tx,y,z\n"
    )
    monkeypatch.setattr(
        "signedkan_wip.src.datasets.DATA_DIR", tmp_path,
    )
    monkeypatch.setattr(
        "signedkan_wip.src.datasets.legacy.DATA_DIR", tmp_path,
    )
    return tsv_path


def test_reddit_loader_parses_5_signed_edges(fake_reddit_tsv):
    g = load("reddit_body")
    assert g.edges.shape == (5, 2)
    assert g.signs.shape == (5,)
    # Subreddits in the fixture: askscience, physics, askreddit,
    # worldnews, math, politics = 6 unique nodes.
    assert g.n_nodes == 6
    # Sentiments: 1, -1, 1, -1, 1 → 3 positive, 2 negative.
    assert int((g.signs == +1).sum()) == 3
    assert int((g.signs == -1).sum()) == 2
    # Zero-sentiment row was correctly skipped (no row index 5 in result).


def test_reddit_loader_node_remap_is_consecutive(fake_reddit_tsv):
    g = load("reddit_body")
    used_ids = set(g.edges.flatten().tolist())
    assert used_ids == set(range(g.n_nodes))


def test_reddit_loader_stats_shape(fake_reddit_tsv):
    g = load("reddit_body")
    s = g.stats()
    assert s["n_nodes"] == 6
    assert s["n_edges"] == 5
    assert s["n_pos"] == 3
    assert s["n_neg"] == 2
    # pos_frac = 3/5 = 0.6
    assert abs(s["pos_frac"] - 0.6) < 1e-6
