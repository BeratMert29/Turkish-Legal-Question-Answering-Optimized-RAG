# BGE-M3 Embedding Fine-Tuning for Turkish Legal RAG

## 1. Approach Overview

### Why Fine-Tune?

Off-the-shelf BGE-M3 was trained on general multilingual data. Turkish legal text has domain-specific vocabulary (e.g., "haksiz fiil", "munfesih", "tekerrur") and structural patterns (article references, cross-references between laws) that the base model has limited exposure to. Contrastive fine-tuning teaches the model that:

- A question about "Is Kanunu madde 17" should be close to the corresponding article chunk, not a similarly-worded chunk from Ceza Kanunu.
- Legal synonyms (e.g., "fesih" and "sozlesmenin sona ermesi") should map to nearby embeddings.

**Current baseline (off-the-shelf BGE-M3):**

| Mode            | R@5   | R@10  | MRR   |
|-----------------|-------|-------|-------|
| Dense           | ~0.38 | ~0.48 | ~0.35 |
| RRF+Rerank      | 0.46  | 0.56  | 0.91  |

**Expected after fine-tuning (conservative estimates):**

| Mode            | R@5   | R@10  | MRR   |
|-----------------|-------|-------|-------|
| Dense           | ~0.50 | ~0.60 | ~0.45 |
| RRF+Rerank      | ~0.55 | ~0.65 | ~0.93 |

Fine-tuning the dense embedder lifts the floor -- the reranker can only reorder what the first stage retrieves, so a better first-stage recall propagates through the entire pipeline.

### BGE-M3 Specifics

BGE-M3 supports three retrieval modes: dense, sparse (lexical weights), and ColBERT (multi-vector). This guide focuses on **dense-only fine-tuning** because:

1. The FAISS index (`IndexFlatIP`) operates on single dense vectors.
2. Dense fine-tuning gives the biggest ROI for the existing pipeline.
3. Sparse and ColBERT modes can be added later without changing the dense training.

The model architecture is XLM-RoBERTa-Large (568M params) with a projection head. Fine-tuning uses **InfoNCE contrastive loss**: for each query, pull the positive passage closer and push negatives away.

---

## 2. Training Data Preparation

### Data Sources

- `data/processed/qa_train.jsonl` -- 13,354 Turkish legal QA pairs
- `data/processed/corpus_chunks.jsonl` -- 1,246 law article chunks (the retrieval corpus)

### Step 2a: Map QA Pairs to Corpus Chunks (Positive Mining)

Each QA pair has a `question` and `answer`. We need to find which corpus chunk(s) contain the answer. Strategy: embed all chunks, embed each answer, take the top-1 chunk by cosine similarity as the positive.

```python
#!/usr/bin/env python3
"""scripts/11_build_embedding_triplets.py -- Build contrastive training triplets."""
import json
import sys
import random
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import config
from retrieval.embedder import Embedder

# ── Config ──────────────────────────────────────────────────────────────
NUM_HARD_NEGATIVES = 7       # hard negatives per query
HARD_NEG_TOP_K = 30          # search pool for hard negative mining
OUTPUT_PATH = config.PROCESSED_DIR / "embedding_triplets.jsonl"
QA_PATH = config.PROCESSED_DIR / "qa_train.jsonl"
CORPUS_PATH = config.PROCESSED_DIR / "corpus_chunks.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # Load data
    print("Loading QA pairs and corpus chunks...")
    qa_pairs = load_jsonl(QA_PATH)
    corpus = load_jsonl(CORPUS_PATH)
    chunk_texts = [c["text"] for c in corpus]
    chunk_ids = [c["chunk_id"] for c in corpus]

    # Load embedder
    print(f"Loading {config.EMBEDDING_MODEL}...")
    embedder = Embedder()
    embedder.load_model()

    # Embed corpus once
    print(f"Embedding {len(chunk_texts)} corpus chunks...")
    corpus_embs = embedder.encode(chunk_texts, is_query=False)  # (N, 1024)

    # Build FAISS index for fast search
    import faiss
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(corpus_embs.astype(np.float32))

    # Process each QA pair
    print(f"Mining positives and hard negatives for {len(qa_pairs)} queries...")
    triplets = []
    skipped = 0

    questions = [qa["question"] for qa in qa_pairs]

    # Batch encode all questions
    print("Encoding all questions...")
    q_embs = embedder.encode(questions, is_query=True)

    # Batch search
    scores_all, indices_all = index.search(q_embs.astype(np.float32), HARD_NEG_TOP_K)

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        answer = qa["answer"]

        # Top-1 chunk = positive (the chunk most similar to the question)
        pos_idx = int(indices_all[i][0])
        pos_chunk_id = chunk_ids[pos_idx]
        pos_text = chunk_texts[pos_idx]
        pos_score = float(scores_all[i][0])

        # Sanity: skip if positive score is very low (question has no relevant chunk)
        if pos_score < 0.3:
            skipped += 1
            continue

        # Hard negatives: chunks that rank high but are NOT the positive
        # Take ranks 5-30 (skip top few to avoid near-duplicates of positive)
        hard_negs = []
        for rank in range(4, HARD_NEG_TOP_K):
            neg_idx = int(indices_all[i][rank])
            if neg_idx == pos_idx:
                continue
            # Extra filter: skip chunks from the same source document
            if corpus[neg_idx].get("source") == corpus[pos_idx].get("source"):
                continue
            hard_negs.append(chunk_texts[neg_idx])
            if len(hard_negs) >= NUM_HARD_NEGATIVES:
                break

        # Pad with random negatives if not enough hard negatives
        while len(hard_negs) < NUM_HARD_NEGATIVES:
            rand_idx = random.randint(0, len(chunk_texts) - 1)
            if rand_idx != pos_idx:
                hard_negs.append(chunk_texts[rand_idx])

        triplets.append({
            "query": question,
            "pos": [pos_text],
            "neg": hard_negs,
        })

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(triplets)} triplets saved to {OUTPUT_PATH}")
    print(f"Skipped {skipped} queries with low positive score (<0.3)")


if __name__ == "__main__":
    main()
```

### FlagEmbedding Data Format

The output format above matches what FlagEmbedding expects. Each line is:

```json
{"query": "...", "pos": ["positive passage"], "neg": ["hard neg 1", "hard neg 2", ...]}
```

- `pos` is a list (usually length 1).
- `neg` is a list of hard negatives. More is better -- 7 is a good default.

### Time Estimate

- Embedding 1,246 chunks: ~10 seconds
- Embedding 13,354 questions: ~2 minutes
- FAISS search: ~1 second
- Total: **~3 minutes** on CPU, faster on GPU

---

## 3. Training Script

### Option A: FlagEmbedding CLI (Recommended)

Install:

```bash
pip install FlagEmbedding
```

Training command:

```bash
torchrun --nproc_per_node 1 \
  -m FlagEmbedding.finetune.embedder.encoder_only.base \
  --model_name_or_path BAAI/bge-m3 \
  --train_data data/processed/embedding_triplets.jsonl \
  --output_dir models/bge-m3-turkish-legal \
  --cache_dir .cache \
  --train_group_size 8 \
  --query_max_len 128 \
  --passage_max_len 512 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --dataloader_drop_last true \
  --normlized true \
  --temperature 0.02 \
  --query_instruction_for_retrieval "" \
  --query_instruction_format "{}{}" \
  --logging_steps 10 \
  --save_steps 500 \
  --warmup_ratio 0.1 \
  --fp16 \
  --gradient_checkpointing
```

**Key hyperparameters explained:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `learning_rate` | 1e-5 | Conservative -- BGE-M3 is already well-trained, we just adapt |
| `temperature` | 0.02 | Standard for InfoNCE; lower = sharper contrast |
| `train_group_size` | 8 | 1 positive + 7 negatives per query (matches our triplet format) |
| `query_max_len` | 128 | Turkish legal questions are short |
| `passage_max_len` | 512 | Chunks are ~1400 chars but 512 tokens captures most content |
| `num_train_epochs` | 2 | Small corpus -- 2 epochs avoids overfitting |
| `per_device_train_batch_size` | 4 | With grad accum 4 = effective batch 16 |

### Option B: LoRA Fine-Tuning (Memory Efficient)

If GPU VRAM is limited (<16GB), use LoRA via the `--use_lora` flag:

```bash
torchrun --nproc_per_node 1 \
  -m FlagEmbedding.finetune.embedder.encoder_only.base \
  --model_name_or_path BAAI/bge-m3 \
  --train_data data/processed/embedding_triplets.jsonl \
  --output_dir models/bge-m3-turkish-legal-lora \
  --cache_dir .cache \
  --train_group_size 8 \
  --query_max_len 128 \
  --passage_max_len 512 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --dataloader_drop_last true \
  --normlized true \
  --temperature 0.02 \
  --query_instruction_for_retrieval "" \
  --query_instruction_format "{}{}" \
  --logging_steps 10 \
  --save_steps 500 \
  --warmup_ratio 0.1 \
  --fp16 \
  --gradient_checkpointing \
  --use_lora true \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules q_proj k_proj v_proj o_proj
```

LoRA differences from full fine-tuning:
- Higher learning rate (2e-5 vs 1e-5) since fewer parameters update.
- 3 epochs instead of 2 since LoRA converges slower.
- Only ~2-4% of parameters trained.

### Option C: Python Script (Full Control)

If the CLI does not work or you need custom logic:

```python
#!/usr/bin/env python3
"""scripts/12_finetune_embeddings.py -- Fine-tune BGE-M3 with sentence-transformers."""
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

MODEL_NAME = "BAAI/bge-m3"
OUTPUT_DIR = config.BASE_DIR / "models" / "bge-m3-turkish-legal"
TRIPLET_FILE = config.PROCESSED_DIR / "embedding_triplets.jsonl"

TRAINING_CONFIG = {
    "learning_rate": 1e-5,
    "epochs": 2,
    "batch_size": 4,
    "warmup_ratio": 0.1,
    "temperature": 0.02,
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from datasets import Dataset

    print(f"Loading triplets from {TRIPLET_FILE}...")
    raw = load_jsonl(TRIPLET_FILE)
    print(f"  {len(raw)} triplets loaded.")

    # Build dataset: anchor, positive, negative
    # sentence-transformers MultipleNegativesRankingLoss expects (anchor, positive) pairs.
    # Hard negatives are included as additional columns.
    records = []
    for t in raw:
        record = {
            "anchor": t["query"],
            "positive": t["pos"][0],
        }
        # Add hard negatives as separate columns
        for j, neg in enumerate(t["neg"]):
            record[f"negative_{j}"] = neg
        records.append(record)

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    print(f"\nLoading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # MultipleNegativesRankingLoss: InfoNCE-style contrastive loss
    # Uses in-batch negatives + the hard negatives from extra columns
    loss = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=TRAINING_CONFIG["epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save(str(OUTPUT_DIR))

    # Also save training config for reproducibility
    config_path = OUTPUT_DIR / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(TRAINING_CONFIG, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
```

### GPU Memory Requirements

| Configuration | VRAM Required | Training Time (est.) |
|--------------|--------------|---------------------|
| Full fine-tune, batch_size=4 + grad_accum=4 | ~14 GB | ~30 min |
| Full fine-tune, batch_size=2 + grad_accum=8 | ~10 GB | ~45 min |
| LoRA, batch_size=4 + grad_accum=4 | ~8 GB | ~20 min |
| CPU (not recommended) | 8+ GB RAM | ~8 hours |

The corpus is small (1,246 chunks, 13K queries), so training is fast. A single T4 (16 GB) or an M1 Mac with 16GB unified memory is sufficient.

**Apple Silicon note:** Replace `--fp16` with `--bf16` on M1/M2/M3 Macs. PyTorch MPS backend does not support fp16 training well.

---

## 4. Integration with Existing Pipeline

### Step 4a: Update `config.py`

Add a config entry for the fine-tuned model path:

```python
# In config.py, add after EMBEDDING_MODEL line:
FINETUNED_EMBEDDING_MODEL = str(BASE_DIR / "models" / "bge-m3-turkish-legal")

# To switch, change EMBEDDING_MODEL:
# EMBEDDING_MODEL = FINETUNED_EMBEDDING_MODEL   # uncomment after fine-tuning
```

### Step 4b: Update `retrieval/embedder.py`

The existing `Embedder` class already works with local model paths because `SentenceTransformer` accepts both HuggingFace model IDs and local directories. No code change is needed if you update `config.EMBEDDING_MODEL` to point to the local path.

However, for easy A/B comparison, you can add a flag:

```python
# In retrieval/embedder.py, modify __init__:
class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL,
                 batch_size: int = config.EMBEDDING_BATCH_SIZE,
                 device: str = None):
        self.model_name = model_name
        # ... rest unchanged

# Usage for comparison:
embedder_base = Embedder(model_name="BAAI/bge-m3")
embedder_ft = Embedder(model_name="models/bge-m3-turkish-legal")
```

This already works -- the constructor accepts `model_name` as a parameter.

### Step 4c: Rebuild the FAISS Index

After fine-tuning, the embeddings change, so the index must be rebuilt:

```bash
# Point config to fine-tuned model, then:
python scripts/02_build_index.py
```

Or, for a clean comparison, save the fine-tuned index separately:

```python
# Temporary override in 02_build_index.py or via environment variable
import config
config.EMBEDDING_MODEL = str(config.BASE_DIR / "models" / "bge-m3-turkish-legal")
config.INDEX_DIR = config.BASE_DIR / "index_finetuned"
```

### Step 4d: If Using LoRA Adapter

If you trained with `--use_lora`, the output directory contains only the adapter weights. To use with `SentenceTransformer`:

```python
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Load base model
model = SentenceTransformer("BAAI/bge-m3")

# Apply LoRA adapter
model[0].auto_model = PeftModel.from_pretrained(
    model[0].auto_model,
    "models/bge-m3-turkish-legal-lora"
)

# Merge for inference speed (optional but recommended)
model[0].auto_model = model[0].auto_model.merge_and_unload()

# Save merged model for easy loading
model.save("models/bge-m3-turkish-legal-merged")
```

After merging, the model loads like any other `SentenceTransformer` -- no PEFT dependency needed at inference time.

---

## 5. Evaluation Plan

### Before vs. After Comparison

Run the retrieval evaluation script with both the base and fine-tuned model:

```bash
# 1. Evaluate base model (already done -- save results if not saved)
python scripts/03_evaluate_retrieval.py
cp results/stage1/retrieval_results.json results/stage1/retrieval_results_base.json

# 2. Switch to fine-tuned model
# Edit config.py: EMBEDDING_MODEL = FINETUNED_EMBEDDING_MODEL

# 3. Rebuild index with fine-tuned embeddings
python scripts/02_build_index.py

# 4. Evaluate fine-tuned model
python scripts/03_evaluate_retrieval.py
cp results/stage1/retrieval_results.json results/stage1/retrieval_results_finetuned.json
```

### Expected Ablation Table for Report

```
| Retrieval Mode     | R@5 (base) | R@5 (ft) | Delta | R@10 (base) | R@10 (ft) | Delta |
|--------------------|------------|----------|-------|-------------|-----------|-------|
| Dense              | 0.38       | 0.50     | +0.12 | 0.48        | 0.60      | +0.12 |
| Hybrid (a=0.7)     | 0.40       | 0.48     | +0.08 | 0.52        | 0.60      | +0.08 |
| RRF                | 0.42       | 0.50     | +0.08 | 0.53        | 0.61      | +0.08 |
| Dense+Rerank       | 0.44       | 0.52     | +0.08 | 0.54        | 0.63      | +0.09 |
| Hybrid+Rerank      | 0.44       | 0.52     | +0.08 | 0.55        | 0.63      | +0.08 |
| RRF+Rerank         | 0.46       | 0.55     | +0.09 | 0.56        | 0.65      | +0.09 |
```

**Note:** These are estimates. Actual gains depend on how well the QA pairs align with the corpus chunks. The dense-only mode should see the largest relative improvement since it benefits most directly from better embeddings. Reranked modes see smaller relative gains because the cross-encoder already compensates for embedding weaknesses.

### HMGS Gold Set Evaluation

Also run on the HMGS gold test set for an independent evaluation:

```bash
# The evaluate script already handles both qa_eval.jsonl and qa_hmgs.jsonl
# Check scripts/03_evaluate_retrieval.py for the HMGS evaluation path
```

---

## 6. Implementation Checklist

Follow these steps in order:

### Phase 1: Data Preparation (~5 minutes)

- [ ] **Step 1.** Verify data exists: `wc -l data/processed/qa_train.jsonl data/processed/corpus_chunks.jsonl`
- [ ] **Step 2.** Create `scripts/11_build_embedding_triplets.py` using the code from Section 2
- [ ] **Step 3.** Run it: `python scripts/11_build_embedding_triplets.py`
- [ ] **Step 4.** Verify output: `wc -l data/processed/embedding_triplets.jsonl` (should be ~13K lines)
- [ ] **Step 5.** Spot-check: `head -1 data/processed/embedding_triplets.jsonl | python -m json.tool`

### Phase 2: Environment Setup (~5 minutes)

- [ ] **Step 6.** Install FlagEmbedding: `pip install FlagEmbedding`
- [ ] **Step 7.** Or for sentence-transformers approach: `pip install sentence-transformers>=3.0`
- [ ] **Step 8.** Check GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"`

### Phase 3: Training (~30-60 minutes)

- [ ] **Step 9.** Save baseline retrieval results: `cp results/stage1/retrieval_results.json results/stage1/retrieval_results_base.json`
- [ ] **Step 10.** Run training using Option A (FlagEmbedding CLI), Option B (LoRA), or Option C (Python script) from Section 3
- [ ] **Step 11.** Verify model saved: `ls models/bge-m3-turkish-legal/`
- [ ] **Step 12.** If LoRA, merge adapter weights (see Section 4d)

### Phase 4: Integration (~10 minutes)

- [ ] **Step 13.** Update `config.py`: set `EMBEDDING_MODEL` to fine-tuned model path
- [ ] **Step 14.** Rebuild FAISS index: `python scripts/02_build_index.py`
- [ ] **Step 15.** Verify index: check that `index/faiss.index` was updated

### Phase 5: Evaluation (~10 minutes)

- [ ] **Step 16.** Run retrieval evaluation: `python scripts/03_evaluate_retrieval.py`
- [ ] **Step 17.** Compare R@5, R@10, MRR with baseline results
- [ ] **Step 18.** If gains are too small (<2% R@5), try: increase epochs to 3, lower temperature to 0.01, add more hard negatives
- [ ] **Step 19.** If overfitting (train loss drops but eval metrics stagnate): reduce epochs, increase temperature, use LoRA

### Phase 6: Downstream Impact (~15 minutes)

- [ ] **Step 20.** Run answer generation: `python scripts/04_generate_answers.py`
- [ ] **Step 21.** Run QA evaluation: `python scripts/05_evaluate_qa.py`
- [ ] **Step 22.** Compare end-to-end QA accuracy with baseline

---

## Appendix: Turkish Legal Domain Notes

### Hard Negative Mining Considerations

Turkish law texts share a lot of structural boilerplate ("Bu Kanunun uygulanmasinda...", "Madde X - ..."). The hard negative mining script in Section 2 filters out chunks from the same source document to avoid trivially similar negatives. For this corpus, this is important because:

- Turk Ceza Kanunu and Ceza Muhakemesi Kanunu have many similarly-phrased articles.
- Anayasa has preamble text that is semantically similar to many law articles.
- Is Kanunu and Borclar Kanunu overlap on employment-related provisions.

### Tokenization

BGE-M3 uses XLM-RoBERTa's SentencePiece tokenizer, which handles Turkish characters (i, I, dotted/dotless i, etc.) well. No special preprocessing is needed. Do NOT lowercase Turkish text before feeding to the model -- the tokenizer is case-sensitive and Turkish casing rules (I/i vs dotted-I/dotless-i) are handled internally.

### Passage Length

Corpus chunks are ~1400 characters (~350-400 Turkish tokens). Setting `passage_max_len=512` in the training config captures the full chunk in most cases. If you see truncation warnings during training, increase to 768.

### What NOT to Do

- Do NOT freeze the first N layers -- BGE-M3's lower layers already capture multilingual representations that are useful; the upper layers need adaptation for legal terminology.
- Do NOT use machine-translated English legal data as augmentation -- Turkish legal concepts do not map 1:1 to English common law.
- Do NOT train for more than 3 epochs on this small corpus -- the model will overfit to the 1,246 chunks.
