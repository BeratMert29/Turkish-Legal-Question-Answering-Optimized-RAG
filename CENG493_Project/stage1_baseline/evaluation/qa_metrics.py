import unicodedata
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Download punkt tokenizer if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

_ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def normalize_turkish(text: str) -> str:
    """
    Turkish-aware normalization.
    CRITICAL: must replace 'I' → 'ı' and 'İ' → 'i' BEFORE calling .lower(),
    because Python's .lower() maps 'I' → 'i' (not Turkish dotless 'ı').
    """
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    return text


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
    pred_tokens = normalize_turkish(predicted).split()
    ref_tokens = normalize_turkish(expected).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)


def rouge_l_score(predicted: str, expected: str) -> float:
    pred_norm = normalize_turkish(predicted)
    exp_norm = normalize_turkish(expected)
    scores = _ROUGE_SCORER.score(exp_norm, pred_norm)
    return scores['rougeL'].fmeasure


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
    keys = ["em", "f1", "bleu", "rouge_l"]
    return {k: sum(m[k] for m in metrics) / len(metrics) for k in keys} | {"num_samples": len(predictions)}


def citation_accuracy(retrieved_sources: list[str], expected_source: str) -> float:
    """Returns 1.0 if expected_source appears in any of the retrieved sources, else 0.0."""
    return 1.0 if any(expected_source in s or s in expected_source for s in retrieved_sources) else 0.0


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
    keys = ["em", "f1", "bleu", "rouge_l"]
    result = {k: sum(m[k] for m in qa_metrics) / n for k in keys}
    result["citation_accuracy"] = sum(cite_scores) / n
    result["num_samples"] = n
    return result
