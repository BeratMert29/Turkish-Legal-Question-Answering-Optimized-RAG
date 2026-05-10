"""Perplexity computation for generated answers via Ollama native API."""
import math
import requests
import logging

log = logging.getLogger(__name__)


def compute_perplexity(
    predictions: list[dict],
    model: str,
    ollama_base: str = "http://localhost:11434",
    sample_size: int = 50,
) -> float | None:
    """
    Compute mean perplexity of generated answers given their contexts.

    predictions: list of dicts with keys 'question', 'predicted', 'retrieved_chunks'
    Returns mean perplexity, or None if Ollama does not support logprobs.
    """
    sampled = predictions[:sample_size]
    perplexities = []

    for pred in sampled:
        question = pred.get("question", "")
        answer = pred.get("predicted", "").strip()
        chunks = pred.get("retrieved_chunks", [])

        if not answer or not chunks:
            continue

        context = "\n\n".join(
            f"[Kaynak {i+1}] {c.get('text','')}" for i, c in enumerate(chunks[:5])
        )
        prompt = f"Bağlam:\n{context}\n\nSoru: {question}\n\nCevap: {answer}"

        try:
            resp = requests.post(
                f"{ollama_base}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 0, "temperature": 0},
                    "logprobs": True,
                },
                timeout=30,
            )
            data = resp.json()
            token_logprobs = data.get("logprobs", {}).get("token_logprobs", [])
            if not token_logprobs:
                log.warning("Ollama logprobs not available for model %s", model)
                return None

            valid = [lp for lp in token_logprobs if lp is not None]
            if valid:
                nll = -sum(valid) / len(valid)
                perplexities.append(math.exp(nll))

        except Exception as exc:
            log.debug("Perplexity request failed: %s", exc)
            continue

    if not perplexities:
        return None
    return round(sum(perplexities) / len(perplexities), 4)
