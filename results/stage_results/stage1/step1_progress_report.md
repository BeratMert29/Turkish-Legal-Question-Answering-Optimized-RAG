# Step 1 Progress Report

Source metrics: [baseline_metrics.json](./baseline_metrics.json)

## Experiment Setup

- Dataset: HMGS gold evaluation set
- Number of evaluation questions: 300
- Retrieval mode: `rrf_rerank`
- Embedding model: `BAAI/bge-m3`
- LLM model: `qwen2.5:7b`
- Device used by the run: `auto`
- Chunk size / overlap: `1400` / `180`

## Retrieval Results

| Metric | Value |
| --- | ---: |
| Recall@5 | 42.84% |
| Recall@10 | 48.19% |
| MRR | 0.8825 |
| nDCG@10 | 0.5469 |
| Retrieval time | N/A s |
| Avg. retrieval time / query | N/A ms |

### Retrieval Bars

```text
Recall@5   ##########-------------- 42.8%
Recall@10  ############------------ 48.2%
MRR        #####################--- 88.3%
nDCG@10    #############----------- 54.7%
```

## QA Results

| Metric | Value |
| --- | ---: |
| Exact Match | 0.00% |
| F1 | 17.63% |
| ROUGE-L | 19.58% |
| BLEU | 4.78% |
| Citation Accuracy | 6.00% |
| Samples | 300 |

### QA Bars

```text
EM         ------------------------ 0.0%
F1         ####-------------------- 17.6%
ROUGE-L    #####------------------- 19.6%
BLEU       #----------------------- 4.8%
Citation   #----------------------- 6.0%
```

## Faithfulness

| Metric | Value |
| --- | ---: |
| Faithful answers | 149 / 150 |
| Faithfulness rate | 99.33% |

### By Retrieval Category

| Category | Faithful | Total | Rate |
| --- | ---: | ---: | ---: |
| Hits | 50 | 50 | 100.00% |
| Partial | 49 | 50 | 98.00% |
| Misses | 50 | 50 | 100.00% |

```text
Faithfulness  ######################## 99.3%
```

## Hardware Snapshot

- Current GPU snapshot: NVIDIA GeForce RTX 5070 Ti
- Current VRAM usage: 251 / 16303 MiB
- Current GPU utilization: 0%
- Current GPU temperature: 39 C
- Historical GPU utilization during the original run: not logged in `baseline_metrics.json`
- Index build time during the original run: not recorded

## Reporting Notes

- This file summarizes the saved Step 1 baseline artifact already present in the repo.
- Retrieval quality is mixed: `MRR` and `nDCG@10` are strong, while `Recall@5/10` is low under the project's strict chunk-level relevance definition.
- Answer quality is still limited (`EM` and `F1` are low), but citation accuracy (6.00%) and faithfulness (99.33%) are strong enough to report that grounding behavior is working.
- For future runs, log `nvidia-smi` samples during evaluation if you need report-grade GPU utilization curves.
