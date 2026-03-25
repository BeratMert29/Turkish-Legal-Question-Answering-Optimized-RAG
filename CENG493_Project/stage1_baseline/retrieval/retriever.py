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
