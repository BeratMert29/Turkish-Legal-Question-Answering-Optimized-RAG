"""Evaluate retrieval quality using embedding-based oracle relevance + BM25 hybrid."""
import json
import time
import sys
import numpy as np
from pathlib import Path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)
import config
from stage1_baseline.data.data_processor import DataProcessor
from stage1_baseline.retrieval.embedder import Embedder
from stage1_baseline.retrieval.retriever import Retriever
from stage1_baseline.retrieval.bm25_retriever import BM25Index
from stage1_baseline.evaluation.retrieval_metrics import compute_all_metrics

ORACLE_NOTE = (
    "Oracle relevance: for each query the top-5 corpus chunks most similar "
    "to the answer embedding are marked relevant. Answer texts encoded as "
    "passages (is_query=False). Corpus vectors reconstructed from FAISS via "
    "reconstruct_n. Relative top-K avoids absolute threshold issues caused by "
    "domain clustering (all Turkish legal text has cosine sim > 0.80)."
)


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

    # ── Load eval set ───────────────────────────────────────────────────────
    eval_path = config.PROCESSED_DIR / config.QA_GOLD_FILE
    eval_set  = DataProcessor.load_jsonl(eval_path)
    print(f"  Eval set: {len(eval_set)} examples")

    # ── Reconstruct corpus vectors from FAISS (N x 1024) ───────────────────
    print(f"\nReconstructing {n_corpus} corpus vectors from FAISS index...")
    t0 = time.time()
    corpus_embs  = np.array(retriever.index.reconstruct_n(0, n_corpus), dtype=np.float32)
    doc_ids_list = [meta['doc_id'] for meta in retriever.metadata]
    print(f"  Done in {time.time()-t0:.1f}s  shape={corpus_embs.shape}")

    # ── Encode all answers as passages ──────────────────────────────────────
    answers = [e['answer'] for e in eval_set]
    print(f"\nEncoding {len(answers)} answers as passages...")
    t0 = time.time()
    answer_embs = np.array(embedder.encode(answers, is_query=False), dtype=np.float32)
    print(f"  Done in {time.time()-t0:.1f}s  shape={answer_embs.shape}")

    # ── Cosine similarity matrix (Q x N) ────────────────────────────────────
    print("\nComputing answer×corpus similarity matrix...")
    t0 = time.time()
    sim_matrix = answer_embs @ corpus_embs.T          # (Q, N)
    print(f"  Done in {time.time()-t0:.1f}s")

    questions = [e['question'] for e in eval_set]

    # ── Dense retrieval ──────────────────────────────────────────────────────
    print(f"\nDense retrieval for {len(eval_set)} queries...")
    t0 = time.time()
    all_dense = retriever.batch_retrieve(questions, top_k=config.TOP_K_RETRIEVAL)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── BM25 index + hybrid retrieval ───────────────────────────────────────
    print(f"\nBuilding BM25 index over {n_corpus} chunks...")
    t0 = time.time()
    bm25_index = BM25Index()
    bm25_index.build(retriever.metadata)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nHybrid retrieval for {len(eval_set)} queries (alpha=0.7)...")
    t0 = time.time()
    all_hybrid = retriever.batch_hybrid_retrieve(
        questions, bm25_index, alpha=0.7, top_k=config.TOP_K_RETRIEVAL
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Build metric results ─────────────────────────────────────────────────
    def build_results(all_retrieved):
        metric_results, full_results = [], {}
        for i, (example, chunks) in enumerate(zip(eval_set, all_retrieved)):
            qid  = example['query_id']
            sims = sim_matrix[i]
            top_idx = np.argsort(sims)[::-1][:config.TOP_K_ORACLE]
            relevant = [doc_ids_list[j] for j in top_idx]
            seen, retrieved_doc_ids = set(), []
            for c in chunks:
                if c['doc_id'] not in seen:
                    seen.add(c['doc_id'])
                    retrieved_doc_ids.append(c['doc_id'])
            metric_results.append({"query_id": qid, "relevant": relevant, "retrieved": retrieved_doc_ids})
            full_results[qid] = chunks
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
        "metrics":        dense_metrics,          # legacy key for script 05
        "hybrid_metrics": hybrid_metrics,
        "per_query_results": dense_results,
        "oracle_relevance_note": ORACLE_NOTE,
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

    print("\n✓ Retrieval evaluation complete")


if __name__ == '__main__':
    main()
