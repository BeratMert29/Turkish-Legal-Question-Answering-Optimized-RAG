#!/usr/bin/env python3
"""
14_eval_all_stages.py — Full Ablation Evaluation (All Stages)

Runs every pipeline configuration through Ollama (no Transformers inference)
and prints a comparative ablation table at the end.

Prerequisites:
  - Ollama running: ollama serve
  - Base model pulled: ollama pull qwen2.5:7b
  - Fine-tuned LLM (optional): python scripts/13_export_lora_to_ollama.py
  - Fine-tuned embedding (optional): python scripts/12_finetune_embeddings.py

Usage:
    python scripts/14_eval_all_stages.py                           # all available stages
    python scripts/14_eval_all_stages.py --stages base,rrf_rerank,llm_ft
    python scripts/14_eval_all_stages.py --stages base --dataset hmgs
    python scripts/14_eval_all_stages.py --list-stages

Available stages:
    base         BGE-M3 base    + dense          + qwen2.5:7b
    hybrid       BGE-M3 base    + hybrid BM25    + qwen2.5:7b
    rrf          BGE-M3 base    + RRF            + qwen2.5:7b
    rrf_rerank   BGE-M3 base    + RRF+rerank     + qwen2.5:7b   ← best retrieval
    llm_ft       BGE-M3 base    + dense          + qwen25-legal-ft (fine-tuned)
    emb_ft       BGE-M3 ft*     + RRF+rerank     + qwen2.5:7b   ← requires emb training
    full         BGE-M3 ft*     + RRF+rerank     + qwen25-legal-ft  ← best overall
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from data.data_processor import DataProcessor
from evaluation.hallucination import run_hallucination_analysis, stratified_sample
from evaluation.qa_metrics import compute_all_qa_metrics_with_citation
from evaluation.retrieval_metrics import compute_all_metrics
from generation.rag_pipeline import RAGPipeline, TURKISH_PROMPT, SHORT_ANSWER_PROMPT
from retrieval.bm25_retriever import BM25Index
from retrieval.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from utils import check_ollama, set_seeds


# ─────────────────────────────────────────────────────────────────────────────
# Stage configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageConfig:
    name: str                        # human-readable label for the ablation table
    embedding: str                   # "base" | "finetuned"
    retrieval: str                   # "dense" | "hybrid" | "rrf"
    use_rerank: bool                 # apply cross-encoder reranker
    llm: str                         # "base" | "finetuned"
    results_dir: Path
    inject_citations: bool = False   # post-hoc citation injection (for ft LLM)
    requires_emb_ft: bool = False    # skip automatically if emb model dir is empty


STAGE_REGISTRY: dict[str, StageConfig] = {
    "base": StageConfig(
        name="Stage 1 — Base RAG",
        embedding="base",
        retrieval="dense",
        use_rerank=False,
        llm="base",
        results_dir=config.RESULTS_DIR_BASE,
    ),
    "hybrid": StageConfig(
        name="Stage 1b — Hybrid BM25+Dense",
        embedding="base",
        retrieval="hybrid",
        use_rerank=False,
        llm="base",
        results_dir=config.RESULTS_DIR_BASE / "hybrid",
    ),
    "rrf": StageConfig(
        name="Stage 1c — RRF",
        embedding="base",
        retrieval="rrf",
        use_rerank=False,
        llm="base",
        results_dir=config.RESULTS_DIR_BASE / "rrf",
    ),
    "rrf_rerank": StageConfig(
        name="Stage 3 — RRF + Rerank",
        embedding="base",
        retrieval="rrf",
        use_rerank=True,
        llm="base",
        results_dir=config.RESULTS_DIR_RERANK,
    ),
    "llm_ft": StageConfig(
        name="Stage 4 — Fine-tuned LLM",
        embedding="base",
        retrieval="dense",
        use_rerank=False,
        llm="finetuned",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_LLM_FT,
    ),
    "emb_ft": StageConfig(
        name="Stage 2 — Fine-tuned Embedding",
        embedding="finetuned",
        retrieval="rrf",
        use_rerank=True,
        llm="base",
        results_dir=config.RESULTS_DIR_EMB_FT,
        requires_emb_ft=True,
    ),
    "full": StageConfig(
        name="Stage 5 — Full Optimized",
        embedding="finetuned",
        retrieval="rrf",
        use_rerank=True,
        llm="finetuned",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_FULL,
        requires_emb_ft=True,
    ),
}

# ordered list for display / default run
DEFAULT_STAGE_ORDER = ["base", "hybrid", "rrf", "rrf_rerank", "llm_ft", "emb_ft", "full"]


# ─────────────────────────────────────────────────────────────────────────────
# Citation injection (post-hoc, overlap-based)
# ─────────────────────────────────────────────────────────────────────────────

def _inject_citations(answer: str, chunks: list) -> str:
    """Append [Kaynak N] markers using token-overlap heuristic."""
    THRESHOLD = 0.15

    def _tok(text: str) -> set:
        return {t.lower() for t in re.split(r"[\s\.,;:!?()\[\]{}'\"]+", text) if t}

    def _overlap(a: set, b: set) -> float:
        return len(a & b) / min(len(a), len(b)) if a and b else 0.0

    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sents or not chunks:
        return answer

    sent_toks = [_tok(s) for s in sents]
    pending: list[tuple[int, int, float]] = []

    for ci, chunk in enumerate(chunks):
        chunk_toks = _tok(chunk.get("text", ""))
        best_score, best_si = 0.0, -1
        for si, st in enumerate(sent_toks):
            sc = _overlap(st, chunk_toks)
            if sc > best_score:
                best_score, best_si = sc, si
        if best_score >= THRESHOLD and best_si >= 0:
            pending.append((best_si, ci + 1, best_score))

    if not pending:
        return answer

    from collections import defaultdict
    s2l: dict = defaultdict(list)
    for si, lbl, _ in pending:
        s2l[si].append(lbl)

    parts = []
    for i, sent in enumerate(sents):
        if i in s2l:
            tags = " ".join(f"[Kaynak {lbl}]" for lbl in sorted(s2l[i]))
            parts.append(f"{sent} {tags}")
        else:
            parts.append(sent)
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval helpers
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve(
    retriever: Retriever,
    questions: list[str],
    stage: StageConfig,
    bm25: Optional[BM25Index],
    reranker: Optional[Reranker],
) -> list[list[dict]]:
    """Dispatch to the correct retrieval method and optionally rerank."""
    initial_k = config.RERANKER_CANDIDATES if stage.use_rerank else config.TOP_K_RETRIEVAL

    t0 = time.time()
    if stage.retrieval == "rrf" and bm25 is not None:
        chunks = retriever.batch_rrf_retrieve(questions, bm25, top_k=initial_k)
    elif stage.retrieval == "hybrid" and bm25 is not None:
        chunks = retriever.batch_hybrid_retrieve(questions, bm25, top_k=initial_k)
    else:
        chunks = retriever.batch_retrieve(questions, top_k=initial_k)

    if stage.use_rerank and reranker is not None:
        chunks = reranker.batch_rerank(questions, chunks, top_k=config.TOP_K_RETRIEVAL)

    print(f"    Retrieval done in {time.time()-t0:.1f}s")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Run one stage
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(
    stage_key: str,
    stage: StageConfig,
    qa_examples,
    corpus_chunks,
    *,
    embedder_cache: dict,
    retriever_cache: dict,
    bm25_cache: dict,
    reranker_cache: dict,
    relevant_map: dict,
    short_answer_mode: bool,
) -> dict:
    """Run a single stage. Returns the final_results dict (same schema as run_baseline)."""

    print(f"\n{'━'*66}")
    print(f"  {stage.name}")
    print(f"{'━'*66}")

    # ── Embedding model ────────────────────────────────────────────────────
    emb_key = stage.embedding
    if emb_key not in embedder_cache:
        if emb_key == "finetuned":
            model_name = config.FINETUNED_EMBEDDING_MODEL
        else:
            model_name = config.EMBEDDING_MODEL
        print(f"  Loading embedding model: {model_name}")
        emb = Embedder(model_name=model_name) if "model_name" in Embedder.__init__.__code__.co_varnames else Embedder()
        emb.load_model()
        embedder_cache[emb_key] = emb

    embedder: Embedder = embedder_cache[emb_key]

    # ── FAISS index (rebuild when embedding changes) ───────────────────────
    idx_key = emb_key
    if idx_key not in retriever_cache:
        print(f"  Building FAISS index ({len(corpus_chunks)} chunks) …")
        retriever = Retriever(embedder)
        texts = [c.text for c in corpus_chunks]
        metadata = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "source": c.source}
            for c in corpus_chunks
        ]
        t0 = time.time()
        retriever.build_index(texts, metadata)
        print(f"    Index built in {time.time()-t0:.1f}s")
        retriever_cache[idx_key] = retriever

    retriever: Retriever = retriever_cache[idx_key]

    # ── BM25 index (shared across all stages that need it) ─────────────────
    needs_bm25 = stage.retrieval in ("hybrid", "rrf")
    bm25: Optional[BM25Index] = None
    if needs_bm25:
        if "bm25" not in bm25_cache:
            print(f"  Building BM25 index …")
            b = BM25Index()
            b.build([{"text": c.text, "chunk_id": c.chunk_id} for c in corpus_chunks])
            bm25_cache["bm25"] = b
        bm25 = bm25_cache["bm25"]

    # ── Reranker (shared) ──────────────────────────────────────────────────
    reranker: Optional[Reranker] = None
    if stage.use_rerank:
        if "reranker" not in reranker_cache:
            print(f"  Loading reranker: {config.RERANKER_MODEL}")
            r = Reranker()
            r.load_model()
            reranker_cache["reranker"] = r
        reranker = reranker_cache["reranker"]

    # ── LLM selection ──────────────────────────────────────────────────────
    llm_model = config.LLM_FINETUNED_MODEL if stage.llm == "finetuned" else config.LLM_MODEL

    # ── Retrieval metrics ──────────────────────────────────────────────────
    print(f"  Retrieval ({stage.retrieval}, rerank={stage.use_rerank}) …")
    questions = [qa.question for qa in qa_examples]

    retrieved_all = _retrieve(retriever, questions, stage, bm25, reranker)

    metric_input = []
    full_retrieved: dict[str, list] = {}
    for qa, chunks in zip(qa_examples, retrieved_all):
        seen: set[str] = set()
        deduped = []
        for c in chunks:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                deduped.append(c["chunk_id"])
        metric_input.append({
            "query_id": qa.query_id,
            "relevant": relevant_map.get(qa.query_id, []),
            "retrieved": deduped,
        })
        full_retrieved[qa.query_id] = chunks

    retrieval_metrics = compute_all_metrics(metric_input)
    print(f"    R@5={retrieval_metrics.get('recall_at_5',0):.4f}  "
          f"R@10={retrieval_metrics.get('recall_at_10',0):.4f}  "
          f"MRR={retrieval_metrics.get('mrr',0):.4f}  "
          f"nDCG@10={retrieval_metrics.get('ndcg_at_10',0):.4f}")

    # ── Generation ─────────────────────────────────────────────────────────
    print(f"  Generation with {llm_model} …")
    pipeline = RAGPipeline(
        retriever,
        model=llm_model,
        short_answer_mode=short_answer_mode,
    )

    predictions = []
    from tqdm import tqdm
    for qa, chunks in tqdm(zip(qa_examples, retrieved_all),
                           total=len(qa_examples), desc=f"  [{stage_key}]"):
        try:
            ctx, ctx_chunks = pipeline.assemble_context(chunks)
            answer = pipeline.generate(qa.question, ctx)
            if stage.inject_citations:
                answer = _inject_citations(answer, ctx_chunks)
            predictions.append({
                "query_id": qa.query_id,
                "predicted": answer,
                "expected": qa.answer,
                "retrieved_sources": [c["source"] for c in ctx_chunks],
                "expected_source": qa.source,
                "retrieved_chunks": [dict(c) for c in ctx_chunks],
            })
        except Exception as exc:
            print(f"\n    ERROR on {qa.query_id}: {exc}")
            predictions.append({
                "query_id": qa.query_id,
                "predicted": "",
                "expected": qa.answer,
                "retrieved_sources": [],
                "expected_source": qa.source,
                "retrieved_chunks": [],
            })

    qa_metrics = compute_all_qa_metrics_with_citation(predictions)
    print(f"    F1={qa_metrics.get('f1',0):.4f}  "
          f"ROUGE-L={qa_metrics.get('rouge_l',0):.4f}  "
          f"Citation={qa_metrics.get('citation_accuracy',0):.4f}")

    # ── Hallucination ──────────────────────────────────────────────────────
    print(f"  Hallucination analysis …")
    from sentence_transformers import CrossEncoder
    if not hasattr(run_stage, "_nli_model"):
        print("    Loading NLI model …")
        run_stage._nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    nli_model = run_stage._nli_model  # reuse across stages

    sample = stratified_sample(predictions, config.HALLUCINATION_SAMPLE_SIZE)
    hall = run_hallucination_analysis(sample, full_retrieved, nli_model)
    faithful_rate = hall["summary"].get("faithful_rate", 0.0)
    print(f"    Faithfulness={faithful_rate:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────
    stage.results_dir.mkdir(parents=True, exist_ok=True)

    final = {
        "hyperparameters": {
            "stage": stage_key,
            "stage_name": stage.name,
            "embedding_model": (config.FINETUNED_EMBEDDING_MODEL
                                if stage.embedding == "finetuned"
                                else config.EMBEDDING_MODEL),
            "retrieval_mode": stage.retrieval + ("_rerank" if stage.use_rerank else ""),
            "llm_model": llm_model,
            "inject_citations": stage.inject_citations,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_for_generation": config.TOP_K_FOR_GENERATION,
        },
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "hallucination_summary": hall.get("summary", {}),
        "faithfulness_rate": faithful_rate,
    }

    out_path = stage.results_dir / "baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    pred_path = stage.results_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"  ✓ Results → {out_path}")
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Ablation table
# ─────────────────────────────────────────────────────────────────────────────

def print_ablation_table(results: dict[str, dict]) -> None:
    """Print a markdown-compatible ablation table to stdout."""

    def _pct(v) -> str:
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "N/A"

    def _f4(v) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"

    header = (
        f"| {'Stage':<26} | {'R@5':>6} | {'R@10':>6} | {'MRR':>6} | "
        f"{'nDCG@10':>7} | {'F1':>6} | {'ROUGE-L':>7} | {'Citation':>8} | {'Faith.':>7} |"
    )
    sep = "|" + "|".join(["-"*w for w in [28, 8, 8, 8, 9, 8, 9, 10, 9]]) + "|"

    print("\n\n" + "="*90)
    print("  ABLATION TABLE")
    print("="*90)
    print(header)
    print(sep)

    for stage_key in DEFAULT_STAGE_ORDER:
        if stage_key not in results:
            continue
        r = results[stage_key]
        ret = r.get("retrieval_metrics", {})
        qa = r.get("qa_metrics", {})
        stage_name = r.get("hyperparameters", {}).get("stage_name", stage_key)
        print(
            f"| {stage_name:<26} | {_f4(ret.get('recall_at_5')):>6} | "
            f"{_f4(ret.get('recall_at_10')):>6} | {_f4(ret.get('mrr')):>6} | "
            f"{_f4(ret.get('ndcg_at_10')):>7} | {_pct(qa.get('f1')):>6} | "
            f"{_pct(qa.get('rouge_l')):>7} | {_pct(qa.get('citation_accuracy')):>8} | "
            f"{_pct(r.get('faithfulness_rate')):>7} |"
        )
    print("="*90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all ablation stages and print a comparison table."
    )
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGE_ORDER),
        help=f"Comma-separated stages to run. Default: all. "
             f"Options: {', '.join(DEFAULT_STAGE_ORDER)}",
    )
    parser.add_argument(
        "--dataset", choices=["kaggle", "hmgs"], default="kaggle",
        help="Evaluation dataset (default: kaggle, 300 questions).",
    )
    parser.add_argument(
        "--list-stages", action="store_true",
        help="Print available stages and exit.",
    )
    args = parser.parse_args()

    if args.list_stages:
        print("\nAvailable stages:")
        for key, cfg in STAGE_REGISTRY.items():
            print(f"  {key:<14} {cfg.name}")
        return

    set_seeds(42)

    # ── Validate requested stages ──────────────────────────────────────────
    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    valid = []
    for key in requested:
        if key not in STAGE_REGISTRY:
            print(f"WARNING: Unknown stage '{key}' — skipping.")
            continue
        stage = STAGE_REGISTRY[key]
        if stage.requires_emb_ft:
            emb_dir = Path(config.FINETUNED_EMBEDDING_MODEL)
            if not emb_dir.exists() or not any(emb_dir.iterdir()):
                print(f"INFO: Stage '{key}' skipped — "
                      f"fine-tuned embedding model not found at {emb_dir}\n"
                      f"  Run: python scripts/12_finetune_embeddings.py first.")
                continue
        if stage.llm == "finetuned":
            # Check Ollama has the model
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if config.LLM_FINETUNED_MODEL not in result.stdout:
                print(f"INFO: Stage '{key}' skipped — "
                      f"Ollama model '{config.LLM_FINETUNED_MODEL}' not found.\n"
                      f"  Run: python scripts/13_export_lora_to_ollama.py first.")
                continue
        valid.append(key)

    if not valid:
        sys.exit("ERROR: No valid stages to run.")

    print(f"\n🚀  Stages to run: {', '.join(valid)}")
    print(f"   Dataset: {args.dataset}\n")

    # ── Check Ollama ───────────────────────────────────────────────────────
    if not check_ollama(config.LLM_BASE_URL, config.LLM_MODEL):
        sys.exit(
            f"ERROR: Ollama not reachable at {config.LLM_BASE_URL}.\n"
            f"  Start with: ollama serve\n"
            f"  Pull model: ollama pull {config.LLM_MODEL}"
        )

    # ── Load data (shared) ────────────────────────────────────────────────
    short_answer_mode = (args.dataset == "hmgs")
    print("Loading data …")
    processor = DataProcessor(config.RAW_DATA_PATH)
    processor.load_and_validate()
    corpus_chunks = list(processor.build_corpus_chunks())

    if args.dataset == "hmgs":
        qa_examples = DataProcessor.build_gold_eval_set()
    else:
        qa_examples = processor.build_qa_eval_set()

    print(f"  Corpus: {len(corpus_chunks)} chunks  |  QA: {len(qa_examples)} examples")

    # ── Ground-truth relevance map (shared) ───────────────────────────────
    relevant_map = DataProcessor.build_relevant_chunk_map(corpus_chunks, qa_examples)

    # ── Shared caches (avoid reloading models between stages) ─────────────
    embedder_cache: dict = {}
    retriever_cache: dict = {}
    bm25_cache: dict = {}
    reranker_cache: dict = {}

    # ── Run stages ─────────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for key in valid:
        stage = STAGE_REGISTRY[key]
        try:
            result = run_stage(
                key, stage, qa_examples, corpus_chunks,
                embedder_cache=embedder_cache,
                retriever_cache=retriever_cache,
                bm25_cache=bm25_cache,
                reranker_cache=reranker_cache,
                relevant_map=relevant_map,
                short_answer_mode=short_answer_mode,
            )
            all_results[key] = result
        except KeyboardInterrupt:
            print(f"\n  ⚠ Interrupted during stage '{key}'. Saving partial results …")
            break
        except Exception as exc:
            print(f"\n  ERROR in stage '{key}': {exc}")
            import traceback
            traceback.print_exc()
            print("  Continuing with next stage …")

    # ── Ablation table ─────────────────────────────────────────────────────
    if all_results:
        print_ablation_table(all_results)

        summary_path = config.BASE_DIR / "results" / "ablation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Full results saved to: {summary_path}")
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()
