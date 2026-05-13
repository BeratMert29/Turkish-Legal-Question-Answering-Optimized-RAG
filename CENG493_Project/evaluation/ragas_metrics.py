"""RAGAS metrics via Ollama (LLM + embeddings)."""
import logging
from typing import Optional

log = logging.getLogger(__name__)


def compute_ragas_metrics(
    predictions: list[dict],
    llm_model: str,
    embedding_model: str = "nomic-embed-text",
    ollama_base: str = "http://localhost:11434",
    sample_size: int = 50,
) -> Optional[dict]:
    """Faithfulness, answer_relevancy, context_precision/recall; keys question, predicted, expected, retrieved_chunks. None if ragas missing."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        try:
            from ragas.metrics import answer_correctness
            _has_answer_correctness = True
        except ImportError:
            _has_answer_correctness = False
    except ImportError as e:
        log.warning("RAGAS dependencies not installed (%s). Run: pip install ragas langchain-ollama", e)
        return None

    sampled = [p for p in predictions[:sample_size] if p.get("predicted")]

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for pred in sampled:
        chunks = pred.get("retrieved_chunks", [])
        contexts = [c.get("text", "") for c in chunks[:5] if c.get("text")]
        if not contexts:
            continue
        data["question"].append(pred.get("question", ""))
        data["answer"].append(pred.get("predicted", ""))
        data["contexts"].append(contexts)
        data["ground_truth"].append(pred.get("expected", ""))

    if not data["question"]:
        log.warning("No valid predictions for RAGAS evaluation.")
        return None

    dataset = Dataset.from_dict(data)

    try:
        llm = ChatOllama(model=llm_model, base_url=ollama_base, temperature=0)
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base)

        metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall]
        if _has_answer_correctness:
            metrics_list.append(answer_correctness)

        result = evaluate(
            dataset=dataset,
            metrics=metrics_list,
            llm=LangchainLLMWrapper(llm),
            embeddings=LangchainEmbeddingsWrapper(embeddings),
            raise_exceptions=False,
        )

        def _to_float(val):
            """Convert a scalar or per-sample list to a rounded float, or None."""
            import math
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                finite = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
                if not finite:
                    return None
                val = sum(finite) / len(finite)
            try:
                f = float(val)
            except (TypeError, ValueError):
                return None
            if math.isnan(f):
                return None
            return round(f, 4)

        out = {
            "ragas_faithfulness": _to_float(result.get("faithfulness")),
            "ragas_answer_relevancy": _to_float(result.get("answer_relevancy")),
            "ragas_context_precision": _to_float(result.get("context_precision")),
            "ragas_context_recall": _to_float(result.get("context_recall")),
        }
        if _has_answer_correctness:
            out["ragas_answer_correctness"] = _to_float(result.get("answer_correctness"))
        return out

    except Exception as exc:
        log.warning("RAGAS evaluation failed: %s", exc)
        return None
