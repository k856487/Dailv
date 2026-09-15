# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=4.51.0",
#     "peft>=0.14.0",
#     "accelerate>=1.0.0",
#     "huggingface_hub>=0.30.0",
#     "safetensors",
# ]
# ///
"""Merge QLoRA/LoRA adapter into Qwen2.5-7B-Instruct and push merged FP16 model."""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_ID = "k86574/qwen25-7b-paperpass-rewrite-lora"
MERGED_ID = "k86574/qwen25-7b-paperpass-rewrite-merged"
OUT_DIR = "merged-model"


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    login(token=token)

    print(f"Loading base: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading adapter: {ADAPTER_ID}")
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    print("Merging and unloading...")
    model = model.merge_and_unload()

    print(f"Saving locally to {OUT_DIR}")
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"Uploading merged model: {MERGED_ID}")
    api = HfApi(token=token)
    api.create_repo(MERGED_ID, private=True, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=OUT_DIR,
        repo_id=MERGED_ID,
        repo_type="model",
        commit_message="Merged Qwen2.5-7B + paperpass rewrite LoRA",
    )
    print(f"Done: https://huggingface.co/{MERGED_ID}")


if __name__ == "__main__":
    main()
