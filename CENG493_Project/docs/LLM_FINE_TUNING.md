# LLM fine-tuning (Stage 4 first)

This document describes **QLoRA supervised fine-tuning** of the generation model for the Turkish legal RAG project, using the code already in this repository. You are running this **before** embedding and reranker stages; that order is fine for getting a trained adapter early, but for the **final report** you should still compare ablations (baseline vs. fine-tuned LLM with the **same** retrieval setup).

---

## 1. What gets trained

| Item | Value |
|------|--------|
| **Base model** | `Qwen/Qwen2.5-7B-Instruct` (Hugging Face) |
| **Method** | QLoRA (4-bit NF4 + LoRA adapters) |
| **Script** | `scripts/08_finetune_llm.py` |
| **Adapter output** | `models/qwen25_lora/` (under `CENG493_Project/`) |
| **Training records** | `qa_train_merged.jsonl` if present, else `qa_train.jsonl` |

Each JSONL row must include at least **`question`** and **`answer`** (chat template: system → user → assistant).

---

## 2. Prerequisites

### 2.1 Hardware

- **NVIDIA GPU with CUDA** (project baseline: RTX 4070 SUPER, 12 GB VRAM).
- QLoRA 7B typically needs on the order of **~8–12 GB** VRAM depending on batch size and context length. If you hit OOM, reduce `per_device_train_batch_size` to `1` and increase `gradient_accumulation_steps` to keep effective batch size 16 (see section 6).

### 2.2 Python dependencies

`requirements.txt` covers inference/indexing. Training needs extra packages (versions should match your CUDA / PyTorch install):

```text
peft
trl
bitsandbytes
accelerate
datasets
```

Install PyTorch with CUDA from the [official matrix](https://pytorch.org/get-started/locally/) first, then the packages above.

### 2.3 Training data

1. Ensure `data/processed/qa_train.jsonl` exists (from your data prep pipeline, e.g. `scripts/01_prepare_data.py`).
2. Build the merged set (recommended):

   ```powershell
   cd CENG493_Project
   python scripts/07_merge_finetune_dataset.py
   ```

   Output: `data/processed/qa_train_merged.jsonl` and `merge_config.json`.

3. **Do not** mix HMGS gold questions into this file if you use HMGS as a held-out benchmark.

### 2.4 Windows / Turkish locale

`08_finetune_llm.py` sets `PYTHONUTF8=1` so TRL can read templates under UTF-8. If you still see encoding errors, run from a terminal with UTF-8 code page or set the environment variable globally.

---

## 3. Hyperparameters (as in code)

These values live in `TRAINING_CONFIG` inside `scripts/08_finetune_llm.py`. Copy the table into your report and note any changes you make.

### 3.1 LoRA

| Parameter | Value |
|-----------|--------|
| `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `bias` | `none` |
| `task_type` | `CAUSAL_LM` |

### 3.2 Quantization (bitsandbytes)

| Parameter | Value |
|-----------|--------|
| `load_in_4bit` | True |
| `bnb_4bit_quant_type` | `nf4` |
| `bnb_4bit_use_double_quant` | True |
| `bnb_4bit_compute_dtype` | `bfloat16` |

### 3.3 Training (TRL `SFTTrainer`)

| Parameter | Value |
|-----------|--------|
| `num_train_epochs` | 3 |
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 |
| **Effective batch size** | 16 |
| `learning_rate` | 2e-4 |
| `warmup_ratio` | 0.03 |
| `lr_scheduler_type` | `cosine` |
| `max_length` | 320 |
| `logging_steps` | 10 |
| `save_strategy` | `epoch` |
| `fp16` | False |
| `bf16` | True |
| `optim` | `paged_adamw_32bit` |
| `report_to` | `none` |
| `gradient_checkpointing` | True |
| `dataloader_num_workers` | 0 |

### 3.4 Prompting during SFT

- **System prompt** (training):  
  `Sen bir Türk hukuku konusunda uzman hukuki asistansın. Soruyu Türkçe yanıtla.`
- **User**: `question` field  
- **Assistant**: `answer` field  

RAG at inference time uses a longer retrieval-aware system prompt in `generation/rag_pipeline.py` / `scripts/09_load_finetuned_model.py`. That is expected: SFT teaches Turkish legal style; RAG still injects `[Kaynak N]` context at run time.

---

## 4. Commands

Run from the **`CENG493_Project`** directory (the folder that contains `config.py`).

### 4.1 Dry run (recommended first)

Loads the 4-bit model, tokenizes five examples, exits without training:

```powershell
python scripts/08_finetune_llm.py --dry-run
```

### 4.2 Full training

```powershell
python scripts/08_finetune_llm.py
```

Artifacts:

- LoRA weights and tokenizer under `models/qwen25_lora/`
- `models/qwen25_lora/training_config.json` (frozen copy of `TRAINING_CONFIG`)
- Checkpoints per epoch: `models/qwen25_lora/checkpoint-*` (if enabled by save strategy)

### 4.3 Resume after interruption

```powershell
python scripts/08_finetune_llm.py --resume
```

Resumes from the latest `checkpoint-*` under `models/qwen25_lora/`.

---

## 5. After training: merge, GGUF, Ollama

Your pipeline for **live demos** uses **Ollama** (`config.LLM_MODEL`, `LLM_BASE_URL`). To serve the fine-tuned weights the same way:

1. Print exact steps (merge HF → GGUF → Modelfile → `ollama create`):

   ```powershell
   python scripts/09_load_finetuned_model.py --show-export-steps
   ```

2. Follow the printed blocks (merge LoRA into base, convert with `llama.cpp`, create `Modelfile`, `ollama create qwen25-legal-ft -f Modelfile`).

3. Point generation at the new model, e.g. in `config.py`:

   ```python
   LLM_MODEL = "qwen25-legal-ft"
   ```

4. Re-run answer generation and QA eval (`scripts/04_generate_answers.py`, `scripts/05_evaluate_qa.py`, or your orchestrator) and save metrics under a new results folder (e.g. `results/llm_ft/`) for the report.

### 5.1 Quick HF smoke test (optional)

```powershell
python scripts/09_load_finetuned_model.py --demo "YOUR_QUESTION_HERE"
```

Loads base + LoRA in 4-bit via Transformers (requires GPU RAM similar to training).

---

## 6. Troubleshooting

| Symptom | Action |
|---------|--------|
| CUDA OOM | Set `per_device_train_batch_size` to `1` and `gradient_accumulation_steps` to `16` in `08_finetune_llm.py`. Optionally lower `max_length` (e.g. 768) if answers are short. |
| `bitsandbytes` / CUDA errors | Reinstall torch + bitsandbytes matching your CUDA version. |
| Very slow training on CPU | Training on CPU is not practical; use a CUDA GPU. |
| Dataset not found | Run `07_merge_finetune_dataset.py` or ensure `data/processed/qa_train.jsonl` exists. |

---

## 7. Reproducibility and report checklist

- [ ] GPU model name and VRAM (e.g. `nvidia-smi`).
- [ ] Exact commit hash of the repo and any edits to `TRAINING_CONFIG`.
- [ ] Path to training JSONL and row count (`merge_config.json` if merged).
- [ ] Training wall-clock and peak VRAM if logged.
- [ ] Final adapter path and Ollama model name if exported.
- [ ] Before/after metrics on the **same** eval set (e.g. HMGS subset, 116 questions) with **identical retrieval** if you are isolating LLM impact.

---

## 8. Relation to other project stages

Doing **LLM fine-tuning first** gives you a strong checkpoint for demos and for measuring answer-quality gains early. For the assignment’s ablation table, you will still want:

- **Same retrieval** when comparing “baseline LLM” vs. “fine-tuned LLM”.
- Later: “fine-tuned LLM + improved retrieval + reranker” as the fully optimized row.

This file only covers the **LLM QLoRA** track; embedding and reranker fine-tuning are separate stages.
