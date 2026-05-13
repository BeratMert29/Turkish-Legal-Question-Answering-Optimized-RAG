"""Tests for retrieval.graph_index.GraphIndex."""

import json
from pathlib import Path

import pytest

from retrieval.graph_index import GraphIndex

# ---------------------------------------------------------------------------
# Fixtures data
# ---------------------------------------------------------------------------

FAKE_METADATA = [
    {
        "chunk_id": "tck_5237_m12",
        "doc_id": "5237",
        "source": "Türk Ceza Kanunu",
        "text": "Madde 12 metni. Madde 13'e bakınız.",
    },
    {
        "chunk_id": "tck_5237_m13",
        "doc_id": "5237",
        "source": "Türk Ceza Kanunu",
        "text": "Madde 13 metni.",
    },
    {
        "chunk_id": "tck_5237_m14",
        "doc_id": "5237",
        "source": "Türk Ceza Kanunu",
        "text": "Madde 14 metni.",
    },
    {
        "chunk_id": "tmk_4721_m100",
        "doc_id": "4721",
        "source": "Türk Medeni Kanunu",
        "text": "5237 sayılı Türk Ceza Kanunu Madde 53 uygulanır.",
    },
]

FAKE_GRAPH: dict = {
    "tck_5237_m12": [["tck_5237_m13", "adj"], ["tck_5237_m13", "intra"]],
    "tck_5237_m13": [["tck_5237_m12", "adj"], ["tck_5237_m14", "adj"]],
    "tck_5237_m14": [["tck_5237_m13", "adj"]],
    "tmk_4721_m100": [["tck_5237_m12", "cross"]],
    "_source_madde_lookup": {"(\"Türk Ceza Kanunu\", \"12\")": ["tck_5237_m12"]},
}


# ---------------------------------------------------------------------------
# Helper to build tmp files and return a GraphIndex
# ---------------------------------------------------------------------------


def _make_index(tmp_path: Path) -> GraphIndex:
    graph_file = tmp_path / "graph.json"
    meta_file = tmp_path / "metadata.jsonl"

    graph_file.write_text(json.dumps(FAKE_GRAPH), encoding="utf-8")
    meta_file.write_text(
        "\n".join(json.dumps(row) for row in FAKE_METADATA),
        encoding="utf-8",
    )
    return GraphIndex(graph_file, meta_file)


def _chunk(chunk_id: str, score: float = 1.0) -> dict:
    """Build a minimal RetrievedChunk-compatible dict."""
    meta = next(m for m in FAKE_METADATA if m["chunk_id"] == chunk_id)
    return {
        "chunk_id": chunk_id,
        "text": meta["text"],
        "doc_id": meta["doc_id"],
        "source": meta["source"],
        "score": score,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_graph(tmp_path: Path) -> None:
    """GraphIndex yüklenir, _graph doğru okunur."""
    gi = _make_index(tmp_path)

    # underscore key must be filtered out
    assert "_source_madde_lookup" not in gi._graph

    # regular keys present
    assert "tck_5237_m12" in gi._graph
    assert "tmk_4721_m100" in gi._graph

    # edges parsed as (nb_id, kind) tuples
    edges_m12 = gi._graph["tck_5237_m12"]
    assert ("tck_5237_m13", "adj") in edges_m12
    assert ("tck_5237_m13", "intra") in edges_m12


def test_expand_adj(tmp_path: Path) -> None:
    """expand([tck_5237_m12]) adj → tck_5237_m13 eklendi, score decay ≈ original * 0.85."""
    gi = _make_index(tmp_path)
    initial_score = 1.0
    result = gi.expand(
        [_chunk("tck_5237_m12", initial_score)],
        hops=1,
        budget=5,
        kinds=("adj",),
    )

    chunk_ids = [c["chunk_id"] for c in result]
    assert "tck_5237_m13" in chunk_ids

    neighbor = next(c for c in result if c["chunk_id"] == "tck_5237_m13")
    assert abs(neighbor["score"] - initial_score * 0.85) < 1e-9


def test_expand_intra(tmp_path: Path) -> None:
    """kinds=('intra',) → adj komşusu eklenmez, intra komşusu eklenir."""
    gi = _make_index(tmp_path)
    # m12 has both adj and intra edges to m13
    result = gi.expand(
        [_chunk("tck_5237_m12", 1.0)],
        hops=1,
        budget=5,
        kinds=("intra",),
    )

    chunk_ids = [c["chunk_id"] for c in result]
    # m13 reached via intra
    assert "tck_5237_m13" in chunk_ids

    # score must use intra decay (0.70), not adj (0.85)
    neighbor = next(c for c in result if c["chunk_id"] == "tck_5237_m13")
    assert abs(neighbor["score"] - 1.0 * 0.70) < 1e-9

    # m14 is only reachable via adj from m13; must NOT appear (kinds excludes adj)
    assert "tck_5237_m14" not in chunk_ids


def test_expand_cross(tmp_path: Path) -> None:
    """expand([tmk_4721_m100], kinds=('cross',)) → tck_5237_m12 eklendi."""
    gi = _make_index(tmp_path)
    result = gi.expand(
        [_chunk("tmk_4721_m100", 1.0)],
        hops=1,
        budget=5,
        kinds=("cross",),
    )

    chunk_ids = [c["chunk_id"] for c in result]
    assert "tck_5237_m12" in chunk_ids

    neighbor = next(c for c in result if c["chunk_id"] == "tck_5237_m12")
    assert abs(neighbor["score"] - 1.0 * 0.60) < 1e-9


def test_expand_budget(tmp_path: Path) -> None:
    """budget=1 → en fazla 1 yeni komşu eklenir."""
    gi = _make_index(tmp_path)
    result = gi.expand(
        [_chunk("tck_5237_m12", 1.0)],
        hops=1,
        budget=1,
        kinds=("adj", "intra", "cross"),
    )

    added = [c for c in result if c["chunk_id"] != "tck_5237_m12"]
    assert len(added) == 1


def test_expand_no_duplicate(tmp_path: Path) -> None:
    """Aynı chunk_id iki kez eklenmez (seen kontrolü)."""
    gi = _make_index(tmp_path)
    # m12 and m13 are mutual adj neighbors; m13 must not appear twice
    result = gi.expand(
        [_chunk("tck_5237_m12", 1.0), _chunk("tck_5237_m13", 0.9)],
        hops=1,
        budget=10,
        kinds=("adj", "intra", "cross"),
    )

    chunk_ids = [c["chunk_id"] for c in result]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_id found in result"
