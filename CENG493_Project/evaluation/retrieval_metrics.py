from ranx import Qrels, Run, evaluate as ranx_evaluate


def compute_all_metrics(results: list[dict]) -> dict:
    """
    Compute retrieval metrics using ranx.

    Args:
        results: list of {"query_id": str, "retrieved": [chunk_id, ...], "relevant": [chunk_id, ...]}
                 Queries with empty relevant sets are excluded from metric computation.

    Returns:
        {"recall_at_5": float, "recall_at_10": float, "mrr": float, "ndcg_at_10": float, "num_queries": int}
    """
    # Build qrels: only include queries that have at least one relevant doc
    qrels_dict = {}
    run_dict = {}

    for r in results:
        qid = str(r["query_id"])
        relevant = r.get("relevant", [])
        retrieved = r.get("retrieved", [])

        if not relevant:
            continue  # skip queries with no ground-truth relevant docs

        qrels_dict[qid] = {str(doc_id): 1 for doc_id in relevant}
        # Score by inverse rank so ranx sorts correctly
        run_dict[qid] = {str(doc_id): 1.0 / (rank + 1) for rank, doc_id in enumerate(retrieved)}

    if not qrels_dict:
        return {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0, "num_queries": 0}

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    raw = ranx_evaluate(qrels, run, ["recall@5", "recall@10", "mrr", "ndcg@10"])

    return {
        "recall_at_5":  float(raw["recall@5"]),
        "recall_at_10": float(raw["recall@10"]),
        "mrr":          float(raw["mrr"]),
        "ndcg_at_10":   float(raw["ndcg@10"]),
        "num_queries":  len(qrels_dict),
    }
