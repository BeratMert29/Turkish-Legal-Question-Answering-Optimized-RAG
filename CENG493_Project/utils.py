"""Shared utilities for the Turkish Legal RAG pipeline."""
import unicodedata
import random
import numpy as np


def normalize_turkish(text: str) -> str:
    """
    Turkish-aware text normalization.

    CRITICAL: replaces 'I' → 'ı' and 'İ' → 'i' BEFORE calling .lower(),
    because Python's built-in .lower() maps 'I' → 'i' (not Turkish dotless 'ı').
    """
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    return text


def set_seeds(seed: int = 42) -> None:
    """Set all RNG seeds for reproducibility (Python, NumPy, PyTorch CPU+GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_ollama(base_url: str, model_name: str) -> bool:
    """Return True if Ollama is running and model_name is available."""
    import httpx
    import logging
    log = logging.getLogger(__name__)
    try:
        root = base_url.rstrip("/").removesuffix("/v1")
        resp = httpx.get(f"{root}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        tags = resp.json().get("models", [])
        model_names = [m.get("name", "") for m in tags]
        if not any(model_name in name for name in model_names):
            log.warning(
                "Ollama is running but model '%s' is not pulled. Run: ollama pull %s",
                model_name, model_name,
            )
            return False
        return True
    except Exception:
        return False
