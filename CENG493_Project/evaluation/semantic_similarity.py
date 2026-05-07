"""
evaluation/semantic_similarity.py — Semantic similarity between predicted and
expected answers using multilingual sentence embeddings.

Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 by default,
which covers Turkish well and is fast enough for eval batches.
"""

from __future__ import annotations

import numpy as np
import torch

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_semantic_similarity(
    predictions: list[dict],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> dict:
    """
    Compute cosine similarity between predicted and expected answer embeddings.

    Args:
        predictions: list of dicts with keys "predicted", "expected", optionally "query_id"
        model_name:  sentence-transformers model to use

    Returns:
        {
            "mean_similarity": float,
            "per_sample": [{"query_id": ..., "similarity": float}, ...]
        }
    """
    if not predictions:
        return {"mean_similarity": 0.0, "per_sample": []}

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # Graceful degradation: return 0.0 scores rather than crashing the pipeline
        per_sample = [
            {"query_id": p.get("query_id", i), "similarity": 0.0}
            for i, p in enumerate(predictions)
        ]
        return {"mean_similarity": 0.0, "per_sample": per_sample}

    model = SentenceTransformer(model_name, device=_DEVICE)

    predicted_texts = [p.get("predicted", "") for p in predictions]
    expected_texts  = [p.get("expected",  "") for p in predictions]

    # Encode in two batches (cheaper than interleaving)
    pred_embs = model.encode(predicted_texts, batch_size=64, show_progress_bar=False,
                             normalize_embeddings=True)
    exp_embs  = model.encode(expected_texts,  batch_size=64, show_progress_bar=False,
                             normalize_embeddings=True)

    per_sample = []
    similarities = []
    for i, p in enumerate(predictions):
        # With normalize_embeddings=True, dot product == cosine similarity
        sim = float(np.dot(pred_embs[i], exp_embs[i]))
        similarities.append(sim)
        per_sample.append({
            "query_id": p.get("query_id", i),
            "similarity": sim,
        })

    mean_sim = float(np.mean(similarities)) if similarities else 0.0
    return {"mean_similarity": mean_sim, "per_sample": per_sample}
