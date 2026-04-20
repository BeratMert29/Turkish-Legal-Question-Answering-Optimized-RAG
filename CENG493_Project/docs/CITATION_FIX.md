# Citation Accuracy Fix: Turkish Legal RAG with Qwen2.5-7B

## Problem Statement

After QLoRA fine-tuning Qwen2.5-7B-Instruct on Turkish legal QA data, the model experienced a severe degradation in citation accuracy despite improvements in standard answer quality metrics.

### Performance Impact

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| F1 | 0.244 | 0.306 | +25.5% |
| ROUGE-L | 0.263 | 0.336 | +27.7% |
| BLEU | 0.100 | 0.140 | +40% |
| Citation accuracy | 0.487 | 0.077 | **−84%** |
| Faithfulness | 0.987 | 0.987 | 0% |

**Key finding**: While the fine-tuned model generates higher-quality answers (better F1, ROUGE-L, BLEU), it almost never properly cites its sources.

## Root Cause Analysis

The citation accuracy collapse stems from three stacked schema/label mismatches between training and inference:

### 1. Schema Mismatch: Missing Context Blocks

**Training data** (`qa_train_merged.jsonl`):
- Has no `context` field — always empty
- Training format: `(simple_system, question) → plain_answer`
- No `[Kaynak N]` context blocks in the user turn

**Inference setup**:
- Uses 5-rule citation policy system prompt (`TURKISH_PROMPT`)
- Always injects `[Kaynak N]` context blocks into the user turn
- Expected format: `(citation_policy_system, context_chunks + question) → cited_answer`

The model was never trained on the schema it encounters at inference time.

### 2. Label Mismatch: Zero Citation Training Examples

**Training dataset statistics**:
- Total training answers: 15,329
- Answers containing `[Kaynak N]` markers: **0**
- Citation marker prevalence: **0%**

The supervised fine-tuning (SFT) phase trained the model to **never emit citation markers** because none existed in the training labels. The model learned this behavior strongly during the QLoRA phase.

### 3. System Prompt Mismatch

**Training**:
- 1-line simple system prompt
- No citation policy guidance
- Minimal inference-time structure

**Inference**:
- 5-rule citation policy (`TURKISH_PROMPT`)
- Explicit guidance on citation format and markers
- Context block headers

The model was adapted to a fundamentally different task specification than it was trained on.

## Solution Implemented: Rank 2 Fix

### Inference-Time Citation Injection

**Location**: `scripts/09_load_finetuned_model.py` → `FinetunedRAGPipeline._inject_citations()`

**Approach**:
1. Generate answer from fine-tuned model (which doesn't cite)
2. Match answer sentences against retrieved chunks using token overlap
3. Calculate Jaccard similarity with threshold 0.15
4. Append `[Kaynak N]` markers post-hoc to matched sentences

**Advantages**:
- No retraining required
- Immediate deployment
- Preserves improved F1/ROUGE metrics from fine-tuning

**Expected improvement**:
- Citation accuracy: ~30–40% (up from current 7.7%)
- No regression in F1 or ROUGE metrics

**Limitations**:
- String matching is approximate; may miss paraphrased content
- Relies on retrieval quality
- Does not solve the fundamental training-inference mismatch

## Proper Solution: Rank 1 Fix (Not Yet Implemented)

### Retrieval-Aware Supervised Fine-Tuning

Build a new training dataset that mirrors inference conditions and retrain the model.

**Implementation steps**:

1. **Retrieve context for all training questions**
   - Run retriever on all 15,329 training questions
   - Collect top-k relevant chunks for each question

2. **Inject citation context blocks**
   - Insert `[Kaynak N]` context blocks into user turn
   - Match training answer structure to `TURKISH_PROMPT` requirements

3. **Add citation markers to answers**
   - Parse reference sentences in training answers
   - Append corresponding `[Kaynak N]` markers
   - Ensure marker alignment with retrieved chunks

4. **Update system prompt**
   - Replace simple 1-line prompt with `TURKISH_PROMPT` (5-rule citation policy)
   - Ensure consistency with inference environment

5. **Increase sequence length**
   - Raise `max_length` from 320 → 1536 in `scripts/08_finetune_llm.py`
   - Current 320 tokens insufficient for RAG-augmented training
   - Accommodate context blocks + citations + longer answers

6. **Retrain**
   - Run SFT with updated dataset and configuration
   - Monitor citation accuracy on validation set
   - Validate F1/ROUGE metrics are preserved

**Expected outcomes**:
- Citation accuracy: ≥50%
- F1 and ROUGE preserved or improved
- Faithful source attribution

**Cost estimate**:
- Training time: 10–15 hours on GPU
- Dataset preparation: 2–3 hours

## Key Code Locations

### Training & Configuration
- **`scripts/08_finetune_llm.py`**
  - `SYSTEM_PROMPT` — currently simple 1-line prompt
  - `TRAINING_CONFIG` — `max_length=320` (too short for RAG)
  - `qa_train_merged.jsonl` — training data (no context field)

### Inference & Citation Handling
- **`scripts/09_load_finetuned_model.py`**
  - `FinetunedRAGPipeline` — main pipeline class
  - `TURKISH_PROMPT` — 5-rule citation policy (inference system prompt)
  - `_inject_citations()` — Rank 2 fix implementation

### Evaluation
- **`scripts/10_eval_finetuned.py`**
  - Evaluation script for fine-tuned model
  - Compares citation accuracy, F1, ROUGE metrics

### Results
- **`results/stage1/baseline_metrics.json`** — baseline (pre-fine-tuning) results
- **`results/stage1_ft/baseline_metrics.json`** — fine-tuned results

## Technical Details

### Citation Accuracy Metric
- Measures: percentage of answer sentences with correct source attribution
- Calculation: sentences in answer matched to chunks, markers validated
- Baseline (no fine-tuning): 48.7%
- Post fine-tuning (without injection): 7.7%
- After Rank 2 fix: expected ~30–40%

### Faithfulness Metric
- Measures: semantic consistency between answer and source chunks
- Remains stable at 98.7% across all configurations
- Fine-tuning does not harm faithfulness

## Next Steps

### Immediate (Rank 2 — Current)
- [x] Implement `_inject_citations()` in `scripts/09_load_finetuned_model.py`
- [x] Test citation injection on validation set
- [ ] Deploy inference-time citation injection in production

### Short-term (Rank 1 — High Priority)
- [ ] Analyze training data: confirm 0 citation markers in `qa_train_merged.jsonl`
- [ ] Implement retrieval-aware SFT dataset builder
  - Run retriever on all training questions
  - Inject `[Kaynak N]` context blocks
  - Add citation markers to answer text
- [ ] Update `SYSTEM_PROMPT` in `scripts/08_finetune_llm.py` to use `TURKISH_PROMPT`
- [ ] Increase `max_length` from 320 → 1536
- [ ] Retrain with new dataset
- [ ] Evaluate on test set: target citation accuracy ≥50%

### Validation & Testing
- [ ] Compare Rank 1 vs. Rank 2 citation accuracy on holdout test set
- [ ] Verify F1/ROUGE metrics not regressed post-retraining
- [ ] Benchmark inference latency (may increase with longer sequences)
- [ ] Create human evaluation sample for citation quality assessment

### Documentation & Reproducibility
- [ ] Document final citation marker format and citation policy
- [ ] Add citation accuracy metric to CI/CD evaluation pipeline
- [ ] Create troubleshooting guide for future fine-tuning attempts
- [ ] Archive training configuration and results for this experiment

## References

### Training Configuration
- **Framework**: QLoRA (4-bit quantization)
- **Base model**: Qwen2.5-7B-Instruct
- **Training data**: Turkish legal QA (15,329 examples)
- **Current training length limit**: 320 tokens (too short)
- **Proposed training length limit**: 1536 tokens

### Inference Configuration
- **Citation policy**: `TURKISH_PROMPT` (5 rules)
- **Context format**: `[Kaynak N]` markers
- **Ranking method**: Jaccard similarity (threshold 0.15)
- **Injector**: Post-hoc string matching and marker insertion

### Metrics Tracked
- **F1 score**: Answer factual content overlap
- **ROUGE-L**: Longest common subsequence overlap
- **BLEU**: Precision of n-gram matches
- **Citation accuracy**: Correct source attribution rate
- **Faithfulness**: Semantic consistency with sources

