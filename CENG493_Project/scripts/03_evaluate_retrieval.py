"""Evaluate retrieval quality using context-hash ground truth + BM25 hybrid."""
import json
import time
import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)

import config
from data.data_processor import DataProcessor
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Index
from evaluation.retrieval_metrics import compute_all_metrics


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load index ──────────────────────────────────────────────────────────
    index_path    = config.INDEX_DIR / config.INDEX_FILE
    metadata_path = config.INDEX_DIR / config.METADATA_FILE
    print(f"Loading index from {index_path}")
    embedder = Embedder()
    embedder.load_model()
    retriever = Retriever(embedder)
    retriever.load_index(index_path, metadata_path)
    n_corpus = retriever.index.ntotal
    print(f"  Index loaded: {n_corpus} vectors")

    # ── Load data via DataProcessor (same approach as run_baseline.py) ──────
    print(f"\nLoading data from {config.RAW_DATA_PATH}")
    processor = DataProcessor(config.RAW_DATA_PATH)
    processor.load_and_validate()
    corpus_chunks = list(processor.build_corpus_chunks())
    qa_examples   = processor.build_qa_eval_set()
    print(f"  Corpus chunks: {len(corpus_chunks)}")
    print(f"  QA examples:   {len(qa_examples)}")

    # ── Build ground-truth relevance map ────────────────────────────────────
    print("\nBuilding ground-truth relevance map (context-hash)...")
    relevant_map = DataProcessor.build_relevant_chunk_map(corpus_chunks, qa_examples)
    matched = sum(1 for v in relevant_map.values() if v)
    print(f"  Queries with at least one relevant chunk: {matched}/{len(qa_examples)}")

    questions = [qa.question for qa in qa_examples]

    # ── Dense retrieval ──────────────────────────────────────────────────────
    print(f"\nDense retrieval for {len(qa_examples)} queries...")
    t0 = time.time()
    all_dense = retriever.batch_retrieve(questions, top_k=config.TOP_K_RETRIEVAL)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── BM25 index + hybrid retrieval ───────────────────────────────────────
    print(f"\nBuilding BM25 index over {len(corpus_chunks)} chunks...")
    t0 = time.time()
    bm25_index = BM25Index()
    bm25_index.build([{"text": c.text, "chunk_id": c.chunk_id} for c in corpus_chunks])
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nHybrid retrieval for {len(qa_examples)} queries (alpha=0.7)...")
    t0 = time.time()
    all_hybrid = retriever.batch_hybrid_retrieve(
        questions, bm25_index, alpha=0.7, top_k=config.TOP_K_RETRIEVAL
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Build metric results ─────────────────────────────────────────────────
    def build_results(all_retrieved):
        metric_results = []
        full_results   = {}
        for qa, chunks in zip(qa_examples, all_retrieved):
            retrieved_chunk_ids = []
            seen: set[str] = set()
            for c in chunks:
                cid = c["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    retrieved_chunk_ids.append(cid)
            metric_results.append({
                "query_id":  qa.query_id,
                "relevant":  relevant_map.get(qa.query_id, []),
                "retrieved": retrieved_chunk_ids,
            })
            full_results[qa.query_id] = chunks
        return metric_results, full_results

    dense_results,  dense_full  = build_results(all_dense)
    hybrid_results, hybrid_full = build_results(all_hybrid)

    dense_metrics  = compute_all_metrics(dense_results)
    hybrid_metrics = compute_all_metrics(hybrid_results)

    print(f"\n=== Dense Retrieval Metrics ===")
    for k, v in dense_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n=== Hybrid Retrieval Metrics (alpha=0.7) ===")
    for k, v in hybrid_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n=== Dense vs Hybrid ===")
    for k in sorted(set(dense_metrics) | set(hybrid_metrics)):
        d = dense_metrics.get(k, "N/A")
        h = hybrid_metrics.get(k, "N/A")
        d_s = f"{d:.4f}" if isinstance(d, float) else str(d)
        h_s = f"{h:.4f}" if isinstance(h, float) else str(h)
        print(f"  {k:20s}  dense={d_s}  hybrid={h_s}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "metrics":           dense_metrics,    # legacy key for script 05
        "hybrid_metrics":    hybrid_metrics,
        "per_query_results": dense_results,
    }
    results_path = config.RESULTS_DIR / "retrieval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {results_path}")

    full_path = config.RESULTS_DIR / "retrieval_full_results.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(dense_full, f, ensure_ascii=False)
    print(f"  Saved: {full_path}")

    hybrid_full_path = config.RESULTS_DIR / "retrieval_hybrid_full_results.json"
    with open(hybrid_full_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_full, f, ensure_ascii=False)
    print(f"  Saved: {hybrid_full_path}")

    print("\nRetrieval evaluation complete")


if __name__ == '__main__':
    main()
