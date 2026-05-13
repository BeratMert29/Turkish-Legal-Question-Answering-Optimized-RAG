"""In-memory cross-reference graph for expanding retrieved chunks with neighbors."""

import json
import logging
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_DECAY: dict[str, float] = {"adj": 0.85, "intra": 0.70, "cross": 0.60}


class GraphIndex:
    """In-memory cross-reference graph; expands retrieved chunks with neighbors."""

    def __init__(self, graph_path: str | Path, metadata_path: str | Path) -> None:
        """Load graph and chunk metadata from disk.

        Args:
            graph_path: Path to ``index/graph.json``.
            metadata_path: Path to ``index/metadata.jsonl``.
        """
        self._graph: dict[str, list[tuple[str, str]]] = {}
        self._chunk_meta: dict[str, dict] = {}

        self._load_graph(Path(graph_path))
        self._load_metadata(Path(metadata_path))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_graph(self, path: Path) -> None:
        """Parse graph.json; skip underscore-prefixed keys."""
        with path.open(encoding="utf-8") as fh:
            raw: dict = json.load(fh)
        for key, edges in raw.items():
            if key.startswith("_"):
                continue
            self._graph[key] = [(nb_id, kind) for nb_id, kind in edges]

    def _load_metadata(self, path: Path) -> None:
        """Build chunk_id → {text, doc_id, source} mapping from metadata.jsonl."""
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record: dict = json.loads(line)
                cid = record.get("chunk_id")
                if cid is None:
                    continue
                self._chunk_meta[cid] = {
                    "text": record.get("text", ""),
                    "doc_id": record.get("doc_id", ""),
                    "source": record.get("source", ""),
                }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(
        self,
        chunks: list[dict],
        hops: int = 1,
        budget: int = 3,
        kinds: tuple[str, ...] = ("adj", "intra", "cross"),
        decay: dict[str, float] | None = None,
    ) -> list[dict]:
        """Top-K retrieval sonucunu komşularla genişlet; yeni RetrievedChunk'lar ekler.

        Args:
            chunks: Initial ``RetrievedChunk`` list (highest score first preferred).
            hops: BFS depth limit for multi-hop expansion.
            budget: Maximum number of new neighbor chunks to add.
            kinds: Which edge types to follow.
            decay: Score multiplier per edge kind; falls back to ``_DEFAULT_DECAY``.

        Returns:
            Original chunks plus added neighbor chunks.
        """
        if decay is None:
            decay = _DEFAULT_DECAY

        seen: set[str] = {c["chunk_id"] for c in chunks}
        added: list[dict] = []
        remaining_budget = budget

        # BFS queue: (chunk_id, parent_score, current_depth)
        sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
        queue: deque[tuple[str, float, int]] = deque(
            (c["chunk_id"], c["score"], 0) for c in sorted_chunks
        )

        while queue and remaining_budget > 0:
            current_id, parent_score, depth = queue.popleft()

            neighbors = self._graph.get(current_id, [])
            for nb_id, kind in neighbors:
                if remaining_budget <= 0:
                    break
                if kind not in kinds:
                    continue
                if nb_id in seen:
                    continue

                meta = self._chunk_meta.get(nb_id)
                if meta is None:
                    log.debug("chunk_id '%s' not found in metadata; skipping.", nb_id)
                    seen.add(nb_id)  # avoid repeated lookups
                    continue

                nb_score = parent_score * decay.get(kind, 0.7)
                new_chunk: dict = {
                    "chunk_id": nb_id,
                    "text": meta["text"],
                    "doc_id": meta["doc_id"],
                    "source": meta["source"],
                    "score": nb_score,
                }
                added.append(new_chunk)
                seen.add(nb_id)
                remaining_budget -= 1

                if depth + 1 < hops:
                    queue.append((nb_id, nb_score, depth + 1))

        return chunks + added

    def expand_batch(
        self,
        batch: list[list[dict]],
        hops: int = 1,
        budget: int = 3,
        kinds: tuple[str, ...] = ("adj", "intra", "cross"),
        decay: dict[str, float] | None = None,
    ) -> list[list[dict]]:
        """expand() her query için ayrı ayrı uygular.

        Args:
            batch: List of per-query ``RetrievedChunk`` lists.
            hops: BFS depth limit forwarded to :meth:`expand`.
            budget: Per-query neighbor budget forwarded to :meth:`expand`.
            kinds: Edge kinds to follow.
            decay: Score decay mapping.

        Returns:
            List of expanded chunk lists, one per query.
        """
        return [
            self.expand(chunks, hops=hops, budget=budget, kinds=kinds, decay=decay)
            for chunks in batch
        ]

    @classmethod
    def from_config(cls) -> "GraphIndex":
        """config.INDEX_DIR/GRAPH_FILE + METADATA_FILE yollarından yükle."""
        import config  # local import to avoid hard dependency at module level

        return cls(
            config.INDEX_DIR / config.GRAPH_FILE,
            config.INDEX_DIR / config.METADATA_FILE,
        )
