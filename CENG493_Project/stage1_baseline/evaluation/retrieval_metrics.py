import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of rank of first relevant doc."""
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    relevant_set = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_set
    )
    ideal_k = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


def compute_query_metrics(result: dict) -> dict:
    retrieved = result["retrieved"]  # list of doc_id strings
    relevant = result["relevant"]    # list of doc_id strings
    return {
        "recall@5": recall_at_k(retrieved, relevant, 5),
        "recall@10": recall_at_k(retrieved, relevant, 10),
        "mrr": mrr(retrieved, relevant),
        "ndcg@10": ndcg_at_k(retrieved, relevant, 10),
    }


def compute_all_metrics(results: list[dict]) -> dict:
    if not results:
        return {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0, "num_queries": 0}
    valid_results = []
    for i, r in enumerate(results):
        if "retrieved" not in r or "relevant" not in r:
            print(f"[WARNING] Skipping result at index {i}: missing 'retrieved' or 'relevant' key.")
            continue
        valid_results.append(r)
    if not valid_results:
        return {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0, "num_queries": 0}
    per_query = [compute_query_metrics(r) for r in valid_results]
    keys = ["recall@5", "recall@10", "mrr", "ndcg@10"]
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys} | {"num_queries": len(valid_results)}
