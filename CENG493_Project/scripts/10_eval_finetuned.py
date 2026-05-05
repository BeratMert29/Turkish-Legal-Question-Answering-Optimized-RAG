"""
Stage 1 Fine-tuned LLM Evaluation
Runs QA evaluation using the fine-tuned LoRA model (Transformers/PEFT, not Ollama)
and saves results to results/stage1_ft/baseline_metrics.json.

Usage:
    python scripts/10_eval_finetuned.py
"""

import os, sys
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# TRL reads Jinja templates without explicit encoding; on Windows with Turkish locale
# (cp1254) this crashes. Force UTF-8 before any trl/transformers import.
os.environ.setdefault("PYTHONUTF8", "1")

import importlib.util
import logging
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from data.data_processor import DataProcessor
from retrieval.embedder import Embedder
from run_baseline import load_index, run_generation_eval, run_hallucination_eval, save_results
from utils import set_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import FinetunedRAGPipeline from scripts/09_load_finetuned_model.py
# (filename starts with a digit so standard import is not possible)
# ---------------------------------------------------------------------------
_script_09_path = _PROJECT_ROOT / "scripts" / "09_load_finetuned_model.py"
_spec = importlib.util.spec_from_file_location("_load_finetuned_model", _script_09_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FinetunedRAGPipeline = _mod.FinetunedRAGPipeline

RESULTS_DIR = config.BASE_DIR / "results" / "stage1_ft"
RETRIEVAL_MODE = "dense_finetuned_llm"


def main() -> None:
    set_seeds(42)

    # ------------------------------------------------------------------
    # 1. Embedder + index
    # ------------------------------------------------------------------
    log.info("Loading embedder …")
    embedder = Embedder()
    embedder.load_model()

    retriever = load_index(embedder)

    # ------------------------------------------------------------------
    # 2. QA eval set
    # ------------------------------------------------------------------
    log.info("Loading data from %s …", config.RAW_DATA_PATH)
    processor = DataProcessor(config.RAW_DATA_PATH)
    summary = processor.load_and_validate()
    log.info("Dataset summary: %s", summary)

    qa_examples = processor.build_qa_eval_set()
    log.info("QA eval set: %d examples", len(qa_examples))

    # ------------------------------------------------------------------
    # 3. Fine-tuned pipeline
    # ------------------------------------------------------------------
    log.info("Instantiating FinetunedRAGPipeline …")
    pipeline = FinetunedRAGPipeline(retriever=retriever)

    log.info("Loading fine-tuned model (base Qwen2.5-7B + LoRA adapter in 4-bit) …")
    pipeline.load_model()
    log.info("Model loaded.")

    # ------------------------------------------------------------------
    # 4. Generation eval
    # ------------------------------------------------------------------
    qa_metrics, predictions = run_generation_eval(pipeline, qa_examples)

    # ------------------------------------------------------------------
    # 5. Hallucination eval
    # ------------------------------------------------------------------
    hallucination = run_hallucination_eval(predictions)

    # ------------------------------------------------------------------
    # 6. Assemble final_results (same structure as run_baseline.py main())
    # ------------------------------------------------------------------
    final_results = {
        "hyperparameters": {
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_for_generation": config.TOP_K_FOR_GENERATION,
            "llm_model": _mod.HF_MODEL_ID,
            "llm_temperature": 0.0,
            "llm_max_tokens": config.LLM_MAX_TOKENS,
            "hallucination_sample_size": config.HALLUCINATION_SAMPLE_SIZE,
            "retrieval_mode": RETRIEVAL_MODE,
            "device": embedder.device,
            "index_build_time_s": None,
        },
        "retrieval_metrics": {},
        "qa_metrics": qa_metrics,
        "hallucination_summary": hallucination.get("summary", {}),
        "faithfulness_rate": hallucination.get("summary", {}).get("faithful_rate"),
    }

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    save_results(final_results, RESULTS_DIR)
    log.info("Done. Results written to %s/baseline_metrics.json", RESULTS_DIR)


if __name__ == "__main__":
    main()
