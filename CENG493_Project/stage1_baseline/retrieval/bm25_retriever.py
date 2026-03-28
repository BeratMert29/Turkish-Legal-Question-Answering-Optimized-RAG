"""BM25 retrieval for hybrid dense+sparse search."""
import unicodedata
import numpy as np
from rank_bm25 import BM25Okapi


def normalize_turkish(text: str) -> str:
    text = text.replace('İ', 'i').replace('I', 'ı')
    return unicodedata.normalize('NFC', text).lower()


def tokenize(text: str) -> list[str]:
    return [w for w in normalize_turkish(text).split() if len(w) >= 2]


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.metadata: list[dict] = []

    def build(self, metadata: list[dict]) -> None:
        self.metadata = metadata
        corpus = [tokenize(m['text']) for m in metadata]
        self.bm25 = BM25Okapi(corpus)

    def get_scores(self, query: str) -> np.ndarray:
        """Return min-max normalized BM25 scores for all documents."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        max_s = scores.max()
        if max_s > 0:
            scores = scores / max_s
        return scores.astype(np.float32)

    def get_top_k(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        """Return top-k (index, raw_score) pairs sorted by BM25 score."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
