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
"""Merge Qwen2.5-3B paperpass LoRA into base and push FP16 merged model."""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_ID = "k86574/qwen25-3b-paperpass-rewrite-lora"
MERGED_ID = "k86574/qwen25-3b-paperpass-rewrite-merged"


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    login(token=token)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print(f"Loading base: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading adapter: {ADAPTER_ID}")
    model = PeftModel.from_pretrained(base, ADAPTER_ID, token=token)
    print("Merging...")
    model = model.merge_and_unload()
    out = "qwen25-3b-paperpass-rewrite-merged"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(MERGED_ID, private=True, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=out,
        repo_id=MERGED_ID,
        repo_type="model",
        commit_message="Merged 3B paperpass distill LoRA",
    )
    print(f"Done: https://huggingface.co/{MERGED_ID}")


if __name__ == "__main__":
    main()
