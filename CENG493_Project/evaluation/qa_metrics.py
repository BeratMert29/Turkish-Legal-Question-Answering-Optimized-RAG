from collections import Counter
import evaluate as hf_evaluate
from utils import normalize_turkish

try:
    _BLEU_METRIC = hf_evaluate.load("bleu")
    _ROUGE_METRIC = hf_evaluate.load("rouge")
    _USE_HF_EVALUATE = True
except Exception:
    _USE_HF_EVALUATE = False


def exact_match(predicted: str, expected: str) -> float:
    return 1.0 if normalize_turkish(predicted.strip()) == normalize_turkish(expected.strip()) else 0.0


def token_f1(predicted: str, expected: str) -> float:
    pred_tokens = normalize_turkish(predicted).split()
    exp_tokens = normalize_turkish(expected).split()
    if not pred_tokens and not exp_tokens:
        return 1.0
    if not pred_tokens or not exp_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    exp_counter = Counter(exp_tokens)
    common = sum((pred_counter & exp_counter).values())
    precision = common / len(pred_tokens)
    recall = common / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_score(predicted: str, expected: str) -> float:
    pred_norm = normalize_turkish(predicted)
    ref_norm = normalize_turkish(expected)
    if not pred_norm or not ref_norm:
        return 0.0
    if _USE_HF_EVALUATE:
        result = _BLEU_METRIC.compute(predictions=[pred_norm], references=[[ref_norm]])
        return float(result["bleu"])
    # fallback: simple unigram overlap
    pred_tokens = set(pred_norm.split())
    ref_tokens = set(ref_norm.split())
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def rouge_l_score(predicted: str, expected: str) -> float:
    pred_norm = normalize_turkish(predicted)
    exp_norm = normalize_turkish(expected)
    if _USE_HF_EVALUATE:
        result = _ROUGE_METRIC.compute(
            predictions=[pred_norm], references=[exp_norm], rouge_types=["rougeL"]
        )
        return float(result["rougeL"])
    return 0.0


def compute_qa_metrics(predicted: str, expected: str) -> dict:
    return {
        "em": exact_match(predicted, expected),
        "f1": token_f1(predicted, expected),
        "bleu": bleu_score(predicted, expected),
        "rouge_l": rouge_l_score(predicted, expected),
    }


def compute_all_qa_metrics(predictions: list[dict]) -> dict:
    """
    predictions: list of {"predicted": str, "expected": str}
    Returns: {"em", "f1", "bleu", "rouge_l", "num_samples"}
    """
    if not predictions:
        return {"em": 0.0, "f1": 0.0, "bleu": 0.0, "rouge_l": 0.0, "num_samples": 0}
    metrics = [compute_qa_metrics(p["predicted"], p["expected"]) for p in predictions]
    keys = ["em", "f1", "rouge_l"]
    result = {k: sum(m[k] for m in metrics) / len(metrics) for k in keys}
    # Corpus-level BLEU via evaluate
    if _USE_HF_EVALUATE:
        preds_norm = [normalize_turkish(p["predicted"]) for p in predictions]
        refs_norm = [[normalize_turkish(p["expected"])] for p in predictions]
        bleu_result = _BLEU_METRIC.compute(predictions=preds_norm, references=refs_norm)
        result["bleu"] = float(bleu_result["bleu"])
    else:
        result["bleu"] = sum(m["bleu"] for m in metrics) / len(metrics)
    result["num_samples"] = len(predictions)
    return result


def citation_accuracy(retrieved_sources: list[str], expected_source: str) -> float:
    """Returns 1.0 if expected_source matches any retrieved source (normalized exact match)."""
    if not expected_source:
        return 0.0
    exp_norm = normalize_turkish(expected_source.strip())
    for s in retrieved_sources:
        if not s:
            continue
        if normalize_turkish(s.strip()) == exp_norm:
            return 1.0
    return 0.0


def compute_all_qa_metrics_with_citation(predictions: list[dict]) -> dict:
    """
    predictions: list of {"predicted": str, "expected": str,
                           "retrieved_sources": list[str], "expected_source": str}
    Returns: averaged em, f1, bleu, rouge_l, citation_accuracy, num_samples
    """
    if not predictions:
        return {"em": 0.0, "f1": 0.0, "bleu": 0.0, "rouge_l": 0.0,
                "citation_accuracy": 0.0, "num_samples": 0}
    qa_metrics = [compute_qa_metrics(p["predicted"], p["expected"]) for p in predictions]
    cite_scores = [
        citation_accuracy(p.get("retrieved_sources", []), p.get("expected_source", ""))
        for p in predictions
    ]
    n = len(predictions)
    keys = ["em", "f1", "rouge_l"]
    result = {k: sum(m[k] for m in qa_metrics) / n for k in keys}
    # Corpus-level BLEU via evaluate
    if _USE_HF_EVALUATE:
        preds_norm = [normalize_turkish(p["predicted"]) for p in predictions]
        refs_norm = [[normalize_turkish(p["expected"])] for p in predictions]
        bleu_result = _BLEU_METRIC.compute(predictions=preds_norm, references=refs_norm)
        result["bleu"] = float(bleu_result["bleu"])
    else:
        result["bleu"] = sum(m["bleu"] for m in qa_metrics) / n
    result["citation_accuracy"] = sum(cite_scores) / n
    result["num_samples"] = n
    return result
