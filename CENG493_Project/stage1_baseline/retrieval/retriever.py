import json
import numpy as np
import faiss
from pathlib import Path
from typing import TypedDict
import config

class RetrievedChunk(TypedDict):
    text: str
    doc_id: str
    source: str
    score: float
    chunk_id: str

class Retriever:
    def __init__(self, embedder, index_path=None, metadata_path=None):
        self.embedder = embedder
        self.index = None
        self.metadata: list[dict] = []
        if index_path and metadata_path:
            self.load_index(index_path, metadata_path)

    def build_index(self, texts: list[str], metadata: list[dict]) -> None:
        """Encode texts and build FAISS IndexFlatIP."""
        if len(texts) != len(metadata):
            raise ValueError(f"texts and metadata must have same length: {len(texts)} vs {len(metadata)}")
        embeddings = self.embedder.encode(texts, is_query=False)
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

    def save_index(self, index_path, metadata_path) -> None:
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_index(self, index_path, metadata_path) -> None:
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f if line.strip()]
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index/metadata mismatch: {self.index.ntotal} vectors vs {len(self.metadata)} metadata entries"
            )

    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> list[RetrievedChunk]:
        """Retrieve top_k chunks for a single query."""
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        q_emb = self.embedder.encode([query], is_query=True, show_progress=False)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(RetrievedChunk(
                text=meta.get("text", ""),
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                score=float(score),
                chunk_id=meta.get("chunk_id", ""),
            ))
        return results

    def batch_retrieve(self, queries: list[str],
                       top_k: int = config.TOP_K_RETRIEVAL) -> list[list[RetrievedChunk]]:
        """Retrieve top_k chunks for all queries in one embedding call."""
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        q_embs = self.embedder.encode(queries, is_query=True)
        scores_all, indices_all = self.index.search(q_embs.astype(np.float32), top_k)
        results = []
        for scores, indices in zip(scores_all, indices_all):
            chunks = []
            for score, idx in zip(scores, indices):
                if idx == -1:
                    continue
                meta = self.metadata[idx]
                chunks.append(RetrievedChunk(
                    text=meta.get("text", ""),
                    doc_id=meta.get("doc_id", ""),
                    source=meta.get("source", ""),
                    score=float(score),
                    chunk_id=meta.get("chunk_id", ""),
                ))
            results.append(chunks)
        return results

    def hybrid_retrieve(self, query: str, bm25_index, alpha: float = 0.7,
                        top_k: int = None,
                        candidate_pool: int = None) -> list[RetrievedChunk]:
        """Hybrid dense+sparse retrieval for a single query.
        final_score = alpha * dense_score + (1 - alpha) * bm25_score
        Only searches candidate_pool candidates from each source — O(candidate_pool) not O(N).
        """
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        if candidate_pool is None:
            candidate_pool = config.RERANKER_CANDIDATES

        q_emb = self.embedder.encode([query], is_query=True, show_progress=False)
        dense_scores_raw, dense_indices_raw = self.index.search(q_emb.astype(np.float32), candidate_pool)

        # Build dense score dict: corpus_idx → score
        dense_scores: dict[int, float] = {}
        for score, idx in zip(dense_scores_raw[0], dense_indices_raw[0]):
            if idx != -1:
                dense_scores[int(idx)] = float(score)

        # Get BM25 top candidates: list of (corpus_idx, normalized_score)
        bm25_top = bm25_index.get_top_k(query, k=candidate_pool)
        # get_top_k returns raw scores — normalize to [0,1] for blending
        bm25_raw = {int(idx): float(score) for idx, score in bm25_top}
        bm25_max = max(bm25_raw.values()) if bm25_raw else 1.0
        bm25_scores: dict[int, float] = {
            idx: score / bm25_max for idx, score in bm25_raw.items()
        } if bm25_max > 0 else bm25_raw

        # Union and blend
        candidates = set(dense_scores) | set(bm25_scores)
        final_scores: dict[int, float] = {}
        for idx in candidates:
            d = dense_scores.get(idx, 0.0)
            b = bm25_scores.get(idx, 0.0)
            final_scores[idx] = alpha * d + (1.0 - alpha) * b

        top_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]
        return [RetrievedChunk(
            text=self.metadata[i].get("text", ""),
            doc_id=self.metadata[i].get("doc_id", ""),
            source=self.metadata[i].get("source", ""),
            score=float(final_scores[i]),
            chunk_id=self.metadata[i].get("chunk_id", ""),
        ) for i in top_indices]

    def batch_hybrid_retrieve(self, queries: list[str], bm25_index,
                              alpha: float = 0.7,
                              top_k: int = None,
                              candidate_pool: int = None) -> list[list[RetrievedChunk]]:
        """Hybrid dense+sparse retrieval for a batch of queries.
        final_score = alpha * dense_score + (1 - alpha) * bm25_score
        Only searches candidate_pool candidates — O(candidate_pool) not O(N).
        """
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        if candidate_pool is None:
            candidate_pool = config.RERANKER_CANDIDATES

        q_embs = self.embedder.encode(queries, is_query=True)
        dense_scores_all, dense_indices_all = self.index.search(q_embs.astype(np.float32), candidate_pool)

        results = []
        for q_idx, query in enumerate(queries):
            # Dense scores for this query
            dense_scores: dict[int, float] = {}
            for score, idx in zip(dense_scores_all[q_idx], dense_indices_all[q_idx]):
                if idx != -1:
                    dense_scores[int(idx)] = float(score)

            # BM25 top candidates
            bm25_top = bm25_index.get_top_k(query, k=candidate_pool)
            bm25_raw = {int(idx): float(score) for idx, score in bm25_top}
            bm25_max = max(bm25_raw.values()) if bm25_raw else 1.0
            bm25_scores: dict[int, float] = {
                idx: score / bm25_max for idx, score in bm25_raw.items()
            } if bm25_max > 0 else bm25_raw

            # Union and blend
            candidates = set(dense_scores) | set(bm25_scores)
            final_scores: dict[int, float] = {}
            for idx in candidates:
                d = dense_scores.get(idx, 0.0)
                b = bm25_scores.get(idx, 0.0)
                final_scores[idx] = alpha * d + (1.0 - alpha) * b

            top_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]
            results.append([RetrievedChunk(
                text=self.metadata[i].get("text", ""),
                doc_id=self.metadata[i].get("doc_id", ""),
                source=self.metadata[i].get("source", ""),
                score=float(final_scores[i]),
                chunk_id=self.metadata[i].get("chunk_id", ""),
            ) for i in top_indices])
        return results

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────

    def batch_rrf_retrieve(self, queries: list[str], bm25_index,
                           top_k: int = None,
                           rrf_k: int = None,
                           candidate_pool: int = None) -> list[list[RetrievedChunk]]:
        """Reciprocal Rank Fusion of dense + BM25 rankings.

        RRF is rank-based so it avoids the score-calibration issues of
        linear blending. Each document's fused score is:
            sum_over_lists( 1 / (rrf_k + rank) )
        """
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        if rrf_k is None:
            rrf_k = config.RRF_K
        if candidate_pool is None:
            candidate_pool = config.RERANKER_CANDIDATES

        q_embs = self.embedder.encode(queries, is_query=True)
        dense_scores_all, dense_indices_all = self.index.search(
            q_embs.astype(np.float32), candidate_pool,
        )

        results: list[list[RetrievedChunk]] = []
        for q_idx, query in enumerate(queries):
            dense_ranking: dict[int, int] = {}
            for rank, idx in enumerate(dense_indices_all[q_idx]):
                if idx != -1:
                    dense_ranking[int(idx)] = rank

            bm25_top = bm25_index.get_top_k(query, k=candidate_pool)
            bm25_ranking = {idx: rank for rank, (idx, _score) in enumerate(bm25_top)}

            candidates = set(dense_ranking) | set(bm25_ranking)
            rrf_scores: dict[int, float] = {}
            for idx in candidates:
                score = 0.0
                if idx in dense_ranking:
                    score += 1.0 / (rrf_k + dense_ranking[idx])
                if idx in bm25_ranking:
                    score += 1.0 / (rrf_k + bm25_ranking[idx])
                rrf_scores[idx] = score

            top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
            results.append([RetrievedChunk(
                text=self.metadata[i].get("text", ""),
                doc_id=self.metadata[i].get("doc_id", ""),
                source=self.metadata[i].get("source", ""),
                score=float(rrf_scores[i]),
                chunk_id=self.metadata[i].get("chunk_id", ""),
            ) for i in top_indices])
        return results
