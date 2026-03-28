import random
import config


def _classify_result(result: dict) -> str:
    """
    Classify a result as hit/partial/miss based on top-1 retrieved chunk score.
    Uses result["retrieved_chunks"] — a list of RetrievedChunk dicts.
    """
    chunks = result.get("retrieved_chunks", [])
    if not chunks:
        return "miss"
    top_score = chunks[0]["score"]  # dict access, not attribute
    if top_score > 0.7:
        return "hit"
    elif top_score >= 0.4:
        return "partial"
    else:
        return "miss"


def stratified_sample(results: list[dict], sample_size: int = config.HALLUCINATION_SAMPLE_SIZE) -> dict:
    """
    Stratify results into hits/partial/misses and sample up to sample_size//3 from each.
    Compensates if a category is smaller than target.
    Returns: {"hits": [...], "partial": [...], "misses": [...]}
    """
    hits, partial, misses = [], [], []
    for r in results:
        cat = _classify_result(r)
        if cat == "hit":
            hits.append(r)
        elif cat == "partial":
            partial.append(r)
        else:
            misses.append(r)

    target = sample_size // 3
    h = random.sample(hits, min(target, len(hits)))
    p = random.sample(partial, min(target, len(partial)))
    m = random.sample(misses, min(target, len(misses)))

    # Compensate: if a category is short, pull from others
    total = len(h) + len(p) + len(m)
    if total < sample_size:
        sampled_ids = {r.get("query_id") for r in h + p + m if r.get("query_id") is not None}
        pool = [x for x in hits + partial + misses if x.get("query_id") not in sampled_ids]
        extra = random.sample(pool, min(sample_size - total, len(pool)))
        # distribute extra evenly across h, p, m
        for i, item in enumerate(extra):
            if i % 3 == 0:
                h.append(item)
            elif i % 3 == 1:
                p.append(item)
            else:
                m.append(item)

    return {"hits": h, "partial": p, "misses": m}


def evaluate_faithfulness(answer: str, context: str, nli_model) -> dict:
    """
    Evaluate faithfulness using NLI CrossEncoder.
    cross-encoder/nli-deberta-v3-small returns logits shape (n_pairs, 3)
    Label order: [contradiction, neutral, entailment] — index 2 is entailment.
    """
    import numpy as np
    logits = nli_model.predict([(context, answer)])  # shape (1, 3)
    logit_vec = logits[0]  # shape (3,)
    # Softmax to get probabilities
    exp_logits = np.exp(logit_vec - np.max(logit_vec))  # numerically stable
    probs = exp_logits / exp_logits.sum()
    entailment_prob = float(probs[2])  # index 2 = entailment
    return {"faithful": entailment_prob >= 0.5, "score": entailment_prob}


def run_hallucination_analysis(
    sample_dict: dict,
    retrieved_results: dict,
    nli_model,
) -> dict:
    """
    Run faithfulness analysis on stratified sample.

    Args:
        sample_dict: {"hits": [...], "partial": [...], "misses": [...]} from stratified_sample()
        retrieved_results: {query_id: [RetrievedChunk, ...]} — full chunk list per query
        nli_model: pre-loaded CrossEncoder instance

    Returns: {"summary": {...}, "per_sample": [...]}
    """
    per_sample = []
    faithful_count = 0
    by_category = {"hits": {"total": 0, "faithful": 0},
                   "partial": {"total": 0, "faithful": 0},
                   "misses": {"total": 0, "faithful": 0}}

    for category, items in sample_dict.items():
        for item in items:
            query_id = item.get("query_id", "")
            answer = item.get("predicted", "")
            # Build context from full retrieved chunks
            chunks = retrieved_results.get(query_id, [])
            context = "\n\n".join(c["text"] for c in chunks[:5]) if chunks else ""
            result = evaluate_faithfulness(answer, context, nli_model)
            if result["faithful"]:
                faithful_count += 1
                by_category[category]["faithful"] += 1
            by_category[category]["total"] += 1
            per_sample.append({
                "query_id": query_id,
                "category": category,
                "answer": answer,
                "faithful": result["faithful"],
                "score": result["score"],
            })

    total = sum(c["total"] for c in by_category.values())
    return {
        "summary": {
            "total": total,
            "faithful_count": faithful_count,
            "faithful_rate": faithful_count / total if total > 0 else 0.0,
            "by_category": by_category,
        },
        "per_sample": per_sample,
    }
