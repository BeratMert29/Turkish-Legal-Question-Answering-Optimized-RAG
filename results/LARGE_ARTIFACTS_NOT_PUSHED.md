# Large Artifacts Not Pushed

These files were generated locally but are too large for normal GitHub storage.

| Artifact | Local path | Approx. size |
| --- | --- | ---: |
| Merged Qwen model | `CENG493_Project/models/qwen25_merged/model.safetensors` | 14.5 GB |
| Ollama GGUF model | `CENG493_Project/models/qwen25_merged.gguf` | 7.7 GB |
| Fine-tuned BGE model | `CENG493_Project/models/bge-m3-turkish-legal/model.safetensors` | 2.1 GB |
| BGE optimizer checkpoints | `CENG493_Project/models/bge-m3-turkish-legal/checkpoint-*/optimizer.pt` | 4.3 GB each |
| BGE checkpoint model files | `CENG493_Project/models/bge-m3-turkish-legal/checkpoint-*/model.safetensors` | 2.1 GB each |
| Qwen LoRA checkpoint zip | `CENG493_Project/models/qwen25_lora/checkpoint-5175.zip` | 397 MB |
| Embedding triplets dataset | `CENG493_Project/data/processed/embedding_triplets.jsonl` | 120 MB |

The important lightweight files were copied into `results/`. The Qwen LoRA adapter itself is included at:

`results/model_configs/qwen25_lora/adapter_model.safetensors`

To recreate the largest artifacts, rerun the relevant scripts:

- `12_finetune_embeddings.py` for `models/bge-m3-turkish-legal/`
- `13_export_lora_to_ollama.py --skip-merge` for `qwen25_merged.gguf`
