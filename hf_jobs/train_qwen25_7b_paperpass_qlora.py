# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=4.51.0",
#     "datasets>=3.0.0",
#     "accelerate>=1.0.0",
#     "peft>=0.14.0",
#     "trl>=0.15.0",
#     "bitsandbytes>=0.45.0",
#     "huggingface_hub>=0.30.0",
# ]
# ///
"""
Qwen2.5-7B-Instruct QLoRA SFT on paperpass rewrite data.
Designed for HF Jobs on l4x1 (preferred) or t4-medium.
"""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DATASET_ID = "k86574/paperpass-rewrite"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HUB_MODEL_ID = "k86574/qwen25-7b-paperpass-rewrite-lora"
OUTPUT_DIR = "qwen25-7b-paperpass-rewrite-lora"


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to load private dataset and push the adapter.")
    login(token=token)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this Jobs run.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

    if vram_gb < 20:
        max_length = 1536
        batch_size = 1
        grad_accum = 16
        print("Using T4-safe memory profile")
    else:
        max_length = 2048
        batch_size = 1
        grad_accum = 8
        print("Using L4 memory profile")

    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")
    if "messages" not in dataset.column_names:
        raise ValueError(f"Expected 'messages' column, got: {dataset.column_names}")

    split = dataset.train_test_split(test_size=0.08, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    print(f"Loading base model (4-bit): {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="epoch",
        save_total_limit=1,
        max_length=max_length,
        packing=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=[],
        push_to_hub=False,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting QLoRA SFT...")
    trainer.train()
    print("Saving adapter locally then pushing to Hub...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    api = HfApi(token=token)
    api.create_repo(HUB_MODEL_ID, private=True, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        commit_message="QLoRA adapter: Qwen2.5-7B paperpass rewrite",
    )
    print(f"Done. Adapter: https://huggingface.co/{HUB_MODEL_ID}")


if __name__ == "__main__":
    main()
