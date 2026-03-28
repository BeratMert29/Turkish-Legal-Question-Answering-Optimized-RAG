"""
Stage 1 Baseline Runner
Usage:
    python run_baseline.py --build-index --eval
    python run_baseline.py --eval --results-dir results/stage1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure parent dir (where config.py lives) and this dir are on the path
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (_PARENT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config
from data.data_processor import DataProcessor, CorpusChunk, QAExample
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from generation.rag_pipeline import RAGPipeline
from evaluation.retrieval_metrics import compute_all_metrics
from evaluation.qa_metrics import compute_all_qa_metrics_with_citation
from evaluation.hallucination import run_hallucination_analysis, stratified_sample
from utils import set_seeds, check_ollama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_index(processor: DataProcessor, embedder: Embedder, chunks: list[CorpusChunk] = None) -> Retriever:
    if chunks is None:
        log.info("Building corpus chunks …")
        chunks = list(processor.build_corpus_chunks())
    log.info("  %d chunks total", len(chunks))

    texts = [c.text for c in chunks]
    metadata = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "source": c.source}
        for c in chunks
    ]

    retriever = Retriever(embedder)
    log.info("Encoding corpus (this may take a while) …")
    retriever.build_index(texts, metadata)

    index_path = config.INDEX_DIR / config.INDEX_FILE
    meta_path = config.INDEX_DIR / config.METADATA_FILE
    retriever.save_index(index_path, meta_path)
    log.info("Index saved → %s", index_path)
    return retriever


def load_index(embedder: Embedder) -> Retriever:
    index_path = config.INDEX_DIR / config.INDEX_FILE
    meta_path = config.INDEX_DIR / config.METADATA_FILE
    log.info("Loading index from %s …", index_path)
    retriever = Retriever(embedder, index_path=index_path, metadata_path=meta_path)
    log.info("  %d vectors loaded", retriever.index.ntotal)
    return retriever


def run_retrieval_eval(
    retriever: Retriever,
    qa_examples: list[QAExample],
    corpus_chunks: list[CorpusChunk],
) -> dict:
    log.info("Building ground-truth relevance map (semantic) …")
    relevant_map = DataProcessor.build_relevant_chunk_map(
        corpus_chunks, qa_examples, retriever=retriever,
    )

    log.info("Running retrieval on %d queries …", len(qa_examples))
    questions = [qa.question for qa in qa_examples]
    all_retrieved = retriever.batch_retrieve(questions, top_k=config.TOP_K_RETRIEVAL)

    results = []
    for qa, retrieved_chunks in zip(qa_examples, all_retrieved):
        results.append({
            "query_id": qa.query_id,
            "retrieved": [c["chunk_id"] for c in retrieved_chunks],
            "relevant": relevant_map.get(qa.query_id, []),
            "retrieved_chunks": [dict(c) for c in retrieved_chunks],
        })

    metrics = compute_all_metrics(results)
    log.info("Retrieval metrics: %s", metrics)
    return metrics, results


def run_generation_eval(
    pipeline: RAGPipeline,
    qa_examples: list[QAExample],
) -> tuple[dict, list[dict]]:
    log.info("Batch-retrieving %d queries …", len(qa_examples))
    questions = [qa.question for qa in qa_examples]
    all_retrieved = pipeline.retriever.batch_retrieve(questions, top_k=config.TOP_K_RETRIEVAL)

    log.info("Running generation on %d examples …", len(qa_examples))
    predictions = []
    from tqdm import tqdm
    for i, (qa, retrieved_chunks) in enumerate(tqdm(zip(qa_examples, all_retrieved), total=len(qa_examples), desc="Generating")):
        try:
            context_used, context_chunks = pipeline.assemble_context(retrieved_chunks)
            answer = pipeline.generate(qa.question, context_used)
            retrieved_sources = [c["source"] for c in context_chunks]
            predictions.append({
                "query_id": qa.query_id,
                "predicted": answer,
                "expected": qa.answer,
                "retrieved_sources": retrieved_sources,
                "expected_source": qa.source,
                "retrieved_chunks": [dict(c) for c in context_chunks],
            })
        except Exception as exc:
            log.warning("Generation failed for query %s: %s", qa.query_id, exc)
            predictions.append({
                "query_id": qa.query_id,
                "predicted": "",
                "expected": qa.answer,
                "retrieved_sources": [],
                "expected_source": qa.source,
                "retrieved_chunks": [],
            })
    metrics = compute_all_qa_metrics_with_citation(predictions)
    log.info("QA metrics: %s", metrics)
    return metrics, predictions


def run_hallucination_eval(predictions: list[dict]) -> dict:
    try:
        from sentence_transformers import CrossEncoder
        log.info("Loading NLI model …")
        nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    except Exception as exc:
        log.warning("NLI model unavailable (%s); skipping hallucination analysis", exc)
        return {"summary": {"faithful_rate": None, "skipped": True}, "per_sample": []}

    # Build sample_dict format expected by run_hallucination_analysis
    retrieval_results_dict = {
        p["query_id"]: p.get("retrieved_chunks", [])
        for p in predictions
    }
    result_list = [
        {
            "query_id": p["query_id"],
            "predicted": p["predicted"],
            "retrieved_chunks": p.get("retrieved_chunks", []),
        }
        for p in predictions
    ]
    sample_dict = stratified_sample(result_list, sample_size=config.HALLUCINATION_SAMPLE_SIZE)
    hallucination_result = run_hallucination_analysis(sample_dict, retrieval_results_dict, nli_model)
    log.info("Hallucination analysis: %s", hallucination_result["summary"])
    return hallucination_result


def save_results(results: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baseline_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Results saved → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 Baseline Runner")
    parser.add_argument("--build-index", action="store_true",
                        help="Build FAISS index from corpus (slow; skipped if index exists)")
    parser.add_argument("--eval", action="store_true",
                        help="Run retrieval + generation + hallucination evaluation")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Run only retrieval metrics (no LLM required)")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR,
                        help="Directory to write baseline_metrics.json")
    args = parser.parse_args()

    set_seeds(42)

    log.info("Loading data from %s …", config.RAW_DATA_PATH)
    processor = DataProcessor(config.RAW_DATA_PATH)
    summary = processor.load_and_validate()
    log.info("Dataset summary: %s", summary)

    embedder = Embedder()
    embedder.load_model()

    log.info("Building corpus chunks for reuse …")
    corpus_chunks: list[CorpusChunk] = list(processor.build_corpus_chunks())

    if args.build_index:
        retriever = build_index(processor, embedder, chunks=corpus_chunks)
    else:
        retriever = load_index(embedder)

    if not args.eval and not args.retrieval_only:
        log.info("--eval not specified; exiting after index step.")
        return

    qa_examples: list[QAExample] = processor.build_qa_eval_set()
    # corpus_chunks already built above — no second call needed

    # --- Retrieval metrics ---
    retrieval_metrics, retrieval_results = run_retrieval_eval(retriever, qa_examples, corpus_chunks)

    qa_metrics: dict = {}
    hallucination: dict = {}

    if args.eval:
        # --- Ollama connectivity check ---
        if not check_ollama(config.LLM_BASE_URL, config.LLM_MODEL):
            log.warning(
                "Ollama not reachable at %s — skipping generation eval",
                config.LLM_BASE_URL,
            )
        else:
            # --- Generation + QA metrics ---
            pipeline = RAGPipeline(retriever)
            qa_metrics, predictions = run_generation_eval(pipeline, qa_examples)

            # --- Hallucination analysis ---
            hallucination = run_hallucination_eval(predictions)
    else:
        log.info("Skipping generation/hallucination (--retrieval-only mode).")

    # --- Merge and save ---
    final_results = {
        "hyperparameters": {
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_for_generation": config.TOP_K_FOR_GENERATION,
            "llm_model": config.LLM_MODEL,
            "llm_temperature": config.LLM_TEMPERATURE,
            "llm_max_tokens": config.LLM_MAX_TOKENS,
            "hallucination_sample_size": config.HALLUCINATION_SAMPLE_SIZE,
        },
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "hallucination_summary": hallucination.get("summary", {}),
        "faithfulness_rate": hallucination.get("summary", {}).get("faithful_rate"),
    }
    save_results(final_results, args.results_dir)


if __name__ == "__main__":
    main()
