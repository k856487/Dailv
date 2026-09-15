# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=4.51.0",
#     "datasets>=3.0.0",
#     "accelerate>=1.0.0",
#     "peft>=0.14.0",
#     "bitsandbytes>=0.45.0",
#     "huggingface_hub>=0.30.0",
# ]
# ///
"""
Hard distillation data: use paperpass 7B teacher to rewrite each user paragraph
multiple times, then push a student SFT dataset for Qwen2.5-3B.
"""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SOURCE_DATASET = "k86574/paperpass-rewrite"
TEACHER_MERGED = "k86574/qwen25-7b-paperpass-rewrite-merged"
TEACHER_BASE = "Qwen/Qwen2.5-7B-Instruct"
TEACHER_LORA = "k86574/qwen25-7b-paperpass-rewrite-lora"
OUT_DATASET = "k86574/paperpass-rewrite-distill-3b"

SYSTEM_PROMPT = (
    "你是一名专业的中文学术文本润色助手。"
    "请将用户给出的论文段落改写得更加自然、流畅、易读，"
    "保持原意与专业术语准确，不编造事实，不增删核心论点。"
    "只输出改写后的段落正文，不要解释、不要加标题或多余说明。"
)

# Multiple samples per prompt: transfer style + expand tiny dataset.
# 2 teacher rewrites + 1 human label ≈ 3x rows (~2k) for 3B SFT.
N_SAMPLES = 2
GEN_MAX_NEW_TOKENS = 640
TEMPERATURES = (0.55, 0.8)


def extract_user_text(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return None


def load_teacher(token: str):
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    api = HfApi(token=token)
    use_merged = True
    try:
        api.repo_info(TEACHER_MERGED, repo_type="model")
    except Exception:
        use_merged = False

    if use_merged:
        print(f"Loading merged teacher: {TEACHER_MERGED}")
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MERGED,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
            token=token,
        )
    else:
        print(f"Loading base+LoRA teacher: {TEACHER_BASE} + {TEACHER_LORA}")
        base = AutoModelForCausalLM.from_pretrained(
            TEACHER_BASE,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
            token=token,
        )
        model = PeftModel.from_pretrained(base, TEACHER_LORA, token=token)

    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_one(tokenizer, model, user_text: str, temperature: float) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    # Drop accidental chat chrome
    for stop in ("\nassistant", "\nuser", "\nsystem"):
        if stop in text:
            text = text.split(stop)[0].strip()
    return text


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    login(token=token)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    src = load_dataset(SOURCE_DATASET, split="train", token=token)
    print(f"Source rows: {len(src)}")

    tokenizer, model = load_teacher(token)

    rows: list[dict] = []
    kept_human = 0
    for i, ex in enumerate(src):
        messages = ex.get("messages") or []
        user_text = extract_user_text(messages)
        if not user_text:
            continue

        # Keep original human label once (anchors meaning)
        human_asst = None
        for m in messages:
            if m.get("role") == "assistant":
                human_asst = (m.get("content") or "").strip()
                break
        if human_asst:
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": human_asst},
                    ],
                    "source": "human",
                }
            )
            kept_human += 1

        for s in range(N_SAMPLES):
            temp = TEMPERATURES[s % len(TEMPERATURES)]
            try:
                rewritten = generate_one(tokenizer, model, user_text, temp)
            except Exception as e:
                print(f"[{i}] gen failed: {e}")
                continue
            if len(rewritten) < 20:
                print(f"[{i}] skip short gen ({len(rewritten)})")
                continue
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": rewritten},
                    ],
                    "source": f"teacher_t{temp}",
                }
            )

        if (i + 1) % 20 == 0 or i == 0:
            print(f"progress {i + 1}/{len(src)} | distill_rows={len(rows)}")

    print(f"Done generating. human={kept_human} total_rows={len(rows)}")
    ds = Dataset.from_list(rows)
    api = HfApi(token=token)
    api.create_repo(OUT_DATASET, repo_type="dataset", private=True, exist_ok=True)
    ds.push_to_hub(OUT_DATASET, private=True, token=token)
    print(f"Pushed dataset: https://huggingface.co/datasets/{OUT_DATASET}")


if __name__ == "__main__":
    main()
