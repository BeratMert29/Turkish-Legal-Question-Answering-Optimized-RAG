> RAG pipeline for Turkish legal question answering with hybrid retrieval, reranking, and fine-tuned components.

# Turkish Legal QA with RAG

A retrieval-augmented generation (RAG) pipeline for answering Turkish legal questions, built for CENG493 (Information Retrieval). The system combines dense retrieval with BM25, reciprocal rank fusion, cross-encoder reranking, and fine-tuned embedding/LLM components.

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### Basic Pipeline
```bash
# Build corpus index (one-time setup)
python scripts/01_prepare_data.py
python scripts/02_build_index.py

# Evaluate on HMGS 2025 benchmark
$env:PYTHONUTF8="1"
python scripts/14_eval_all_stages.py --dataset hmgs
```

## Architecture

The system includes six evaluation stages that progressively add capabilities:

| Stage | Components | Key Addition |
|-------|-----------|--------------|
| **1 — Base RAG** | Dense retrieval (BGE-M3) + Qwen2.5 | Baseline system |
| **2 — Hybrid** | BM25 + dense fusion (RRF) | Sparse + dense retrieval |
| **3 — RRF + Rerank** | Reciprocal rank fusion + BGE-reranker-v2-m3 | Cross-encoder reranking |
| **4 — Fine-tuned Embedding** | BGE-M3 fine-tuned on Turkish legal data | Task-specific embeddings |
| **5 — Fine-tuned LLM** | Qwen2.5-7B + QLoRA fine-tuning | Legal QA instruction tuning |
| **6 — Full Optimized** | All components combined | Best performance |

### Retrieval Pipeline

1. **Dual Retrieval**: BM25 (sparse) and FAISS dense search (default: top-50 each)
2. **Rank Fusion**: Reciprocal Rank Fusion (RRF) combines rankings
3. **Reranking**: BGE-reranker-v2-m3 cross-encoder reranks top candidates
4. **Context Assembly**: Top-K passages fed to LLM with question and instructions

### Models

- **Embedding**: BAAI/bge-m3 (base and fine-tuned variants)
- **Reranker**: BAAI/bge-reranker-v2-m3
- **LLM**: Qwen/Qwen2.5-7B-Instruct via Ollama (local inference, no API key required)

## Evaluation

### Metrics

**Retrieval**: Recall@K, MRR, nDCG@10, Source Hit@K, Precision@K

**QA**: F1, ROUGE-L, BLEU, Exact Match, Citation Accuracy

**Faithfulness**: NLI-based hallucination detection, RAGAS (faithfulness, answer relevancy, context precision/recall), semantic similarity, perplexity

**Overall**: 3 composite scenario scores, LLM judge (quality, faithfulness, relevancy, coherence)

### Datasets

- `qa_eval.jsonl` — 300 synthetic questions (development/ablation)
- `qa_hmgs.jsonl` — 161 HMGS 2025 Turkish legal exam questions (primary benchmark)

## Project Structure

```
CENG493_Project/
├── config.py                  # all hyperparameters and paths
├── requirements.txt
├── utils.py
├── data/
│   ├── processed/             # chunked corpus, metadata, train/eval jsonl files
│   └── extra_laws.jsonl       # supplementary laws added to corpus
├── index/                     # FAISS vector index
├── models/
│   ├── bge-m3-turkish-legal/  # fine-tuned embedding model
│   └── qwen25_lora/           # QLoRA adapter weights
├── results/                   # per-stage eval output (metrics, predictions)
├── evaluation/                # metric modules (F1, RAGAS, hallucination, etc.)
├── generation/
│   └── rag_pipeline.py        # RAG pipeline: retrieve → assemble context → generate
├── retrieval/                 # dense, BM25, RRF, reranker modules
└── scripts/
    ├── 00_validate_raw_data.py
    ├── 01_prepare_data.py
    ├── 02_build_index.py
    ├── 03_evaluate_retrieval.py
    ├── 04_generate_answers.py
    ├── 05_evaluate_qa.py
    ├── 06_generate_stage1_report.py
    ├── 07_merge_finetune_dataset.py
    ├── 08_finetune_llm.py
    ├── 09_load_finetuned_model.py
    ├── 10_eval_finetuned.py
    ├── 11_build_embedding_triplets.py
    ├── 12_finetune_embeddings.py
    ├── 13_export_lora_to_ollama.py
    └── 14_eval_all_stages.py
```

## Fine-tuning

### LLM Fine-tuning (QLoRA)

```bash
# Prepare fine-tuning dataset
python scripts/07_prepare_finetuning.py

# Fine-tune Qwen2.5-7B (max_length=1536, RAG-formatted)
python scripts/08_finetune_llm.py

# Export LoRA adapter to Ollama
python scripts/13_export_lora_to_ollama.py
```

Requirements: CUDA-capable GPU (tested on RTX 5070 Ti 16GB).

### Embedding Fine-tuning

```bash
# Build triplet training data
python scripts/11_build_embedding_triplets.py

# Fine-tune BGE-M3 + reranker
python scripts/12_finetune_embeddings.py
```

## Running Full Ablation

```bash
# Evaluate all 6 stages on both datasets
$env:PYTHONUTF8="1"
python scripts/14_eval_all_stages.py --dataset hmgs
python scripts/14_eval_all_stages.py --dataset qa
```

Output includes per-stage metrics, composite scores, and comparison tables.

## Configuration

All settings are in `config.py`. Key values to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `qwen2.5:14b` | Ollama model for generation |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `TOP_K_RETRIEVAL` | 10 | Number of chunks retrieved |
| `TOP_K_FOR_GENERATION` | 5 | Chunks passed to LLM |
| `CHUNK_SIZE` | 1400 | Characters per chunk |
| `HF_PERPLEXITY_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Reference model for perplexity scoring |

## Tech Stack

- **Python 3.11**, PyTorch, HuggingFace (transformers, PEFT, TRL)
- **Retrieval**: FAISS, rank_bm25
- **Inference**: Ollama (local)
- **Evaluation**: RAGAS, langchain-ollama, sentence-transformers
- **GPU**: CUDA (tested on RTX 5070 Ti 16GB)

