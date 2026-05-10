"""QLoRA fine-tuning of Qwen/Qwen2.5-7B-Instruct on merged Turkish legal QA data."""
import argparse
import json
import os
import sys
from pathlib import Path

# TRL reads Jinja templates without explicit encoding; on Windows with Turkish locale
# (cp1254) this crashes. Force UTF-8 before any trl import.
os.environ.setdefault("PYTHONUTF8", "1")

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = config.BASE_DIR / "models" / "qwen25_lora"

RAG_DATASET    = config.PROCESSED_DIR / "qa_train_rag.jsonl"
MERGED_DATASET = config.PROCESSED_DIR / "qa_train_merged.jsonl"
FALLBACK_DATASET = config.PROCESSED_DIR / "qa_train.jsonl"

SYSTEM_PROMPT = (
    "Sen Türk hukuku alanında uzman bir hukuki asistansın. "
    "Yanıtını yalnızca verilen bağlama dayandır. "
    "Kısa, doğrudan ve kaynaklı cevap ver. "
    "Yeni soru üretme, bağlamı tekrar yazma."
)

TRAINING_CONFIG = {
    "base_model": HF_MODEL_ID,
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "qlora": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    },
    "training": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 8,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_length": 512,
        "dataset_text_field": "text",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "fp16": False,
        "bf16": True,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
    },
    "adapter_output_dir": str(ADAPTER_DIR),
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_as_chat(example: dict, tokenizer) -> str:
    question = example.get("question", "")
    ctx = example.get("context_str", "") or example.get("context", "")
    context = ctx
    if context and context.strip():
        user_content = f"Bağlam:\n{context.strip()}\n\nSoru: {question}"
    else:
        user_content = question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["answer"]},
    ]
    # add_generation_prompt=False because the assistant turn is already included
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def print_gpu_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): "
                      f"{reserved:.1f} GB reserved / {total:.1f} GB total")
        else:
            print("  No CUDA GPU detected — training will run on CPU (very slow).")
    except Exception as e:
        print(f"  Could not query GPU memory: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-7B-Instruct")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and tokenize 5 examples, then exit without training.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume training from the latest checkpoint in {ADAPTER_DIR}.",
    )
    return p


def find_last_checkpoint(adapter_dir: Path):
    """Return path to the latest checkpoint subdir, or None."""
    checkpoints = sorted(adapter_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def main() -> None:
    args = build_arg_parser().parse_args()

    # Heavy imports deferred so --help and import-time errors are readable
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    dataset_path = RAG_DATASET if RAG_DATASET.exists() else (MERGED_DATASET if MERGED_DATASET.exists() else FALLBACK_DATASET)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}")
        sys.exit(1)
    print(f"Loading dataset from {dataset_path} ...")
    raw_records = load_jsonl(dataset_path)
    print(f"  {len(raw_records)} examples loaded.")

    print("\nGPU memory at startup:")
    print_gpu_memory()

    print(f"\nLoading tokenizer from {HF_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    # Qwen2.5 has no pad token by default; reuse eos so padding doesn't break training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dry_run:
        print("\n[dry-run] Tokenizing 5 examples ...")
        for i, rec in enumerate(raw_records[:5]):
            text = format_as_chat(rec, tokenizer)
            tokens = tokenizer(text, return_tensors="pt")
            print(f"  Example {i+1}: {tokens['input_ids'].shape[1]} tokens")

        print("\n[dry-run] Loading model in 4-bit ...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"  Model loaded: {model.__class__.__name__}")
        print("\n[dry-run] Done. Exiting without training.")
        return

    print("\nPreparing formatted dataset ...")
    formatted_texts = [format_as_chat(r, tokenizer) for r in raw_records]
    hf_dataset = Dataset.from_dict({"text": formatted_texts})
    split = hf_dataset.train_test_split(test_size=0.1, seed=42)
    hf_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(hf_dataset)}, Val: {len(eval_dataset)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=TRAINING_CONFIG["qlora"]["load_in_4bit"],
        bnb_4bit_quant_type=TRAINING_CONFIG["qlora"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=TRAINING_CONFIG["qlora"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"\nLoading base model {HF_MODEL_ID} in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lc = TRAINING_CONFIG["lora"]
    lora_config = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        target_modules=lc["target_modules"],
        bias=lc["bias"],
        task_type=lc["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nGPU memory after model load:")
    print_gpu_memory()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    tc = TRAINING_CONFIG["training"]
    resume_from = None
    if args.resume:
        resume_from = find_last_checkpoint(ADAPTER_DIR)
        if resume_from:
            print(f"\nResuming from checkpoint: {resume_from}")
        else:
            print("\nNo checkpoint found in adapter dir — starting from scratch.")

    training_args = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        warmup_ratio=tc["warmup_ratio"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        logging_steps=tc["logging_steps"],
        save_strategy=tc["save_strategy"],
        eval_strategy="steps",
        eval_steps=50,
        fp16=tc["fp16"],
        bf16=tc["bf16"],
        optim=tc["optim"],
        report_to=tc["report_to"],
        remove_unused_columns=True,
        gradient_checkpointing=tc["gradient_checkpointing"],
        dataloader_num_workers=tc["dataloader_num_workers"],
        # SFT-specific fields live here in TRL 1.x
        dataset_text_field=tc["dataset_text_field"],
        max_length=tc["max_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training ...")
    trainer.train(resume_from_checkpoint=resume_from)

    print(f"\nSaving LoRA adapter to {ADAPTER_DIR} ...")
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    config_out_path = ADAPTER_DIR / "training_config.json"
    with open(config_out_path, "w", encoding="utf-8") as f:
        json.dump(TRAINING_CONFIG, f, ensure_ascii=False, indent=2)

    print(f"\nTraining complete.")
    print(f"  Adapter saved to       : {ADAPTER_DIR}")
    print(f"  Training config saved  : {config_out_path}")
    print("\nFinal GPU memory:")
    print_gpu_memory()


if __name__ == "__main__":
    main()
