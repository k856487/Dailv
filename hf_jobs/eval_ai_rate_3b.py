#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eval 3B rewrite vs AI-rate detector. Pass if mean AI-prob drops >= 5pp."""

from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

DET_ROOT = Path(r"D:\1code\cursor\ai率检测")
# Keep detector weights under ai率检测; put large LLM downloads on D:\hf_models
MODEL_CACHE = Path(r"D:\hf_models\cache")
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE / "transformers"))
os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
DET_CACHE = DET_ROOT / ".hf_cache" / "hub"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.bert import BertForSequenceClassification, BertTokenizer

REPORT = Path(r"D:\hf_models\tools\dailv\unattended_eval_report.json")
ADAPTER_ID = "k86574/qwen25-3b-paperpass-rewrite-lora"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DET_REPO = "yuchuantian/AIGC_detector_zhv3"
SYSTEM_PROMPT = (
    "你是中文学术论文降AI率改写助手。"
    "将用户段落改写成更像人工撰写的学术表述：避免空泛套话、排比堆砌、模板化起承转合；"
    "句式长短错落，用具体、克制的学术语言重述原意，保留术语与论点，不编造事实。"
    "只输出改写后的段落正文，不要解释。"
)

SAMPLES = [
    "随着数字技术的飞速发展，人工智能技术在各个领域得到了广泛应用。本文旨在探讨人工智能技术在教育领域的应用现状及其发展趋势，通过分析相关文献和案例，总结出人工智能技术对教育模式创新的重要意义，并提出相应的对策建议，以期为相关研究和实践提供参考。",
    "在全球化背景下，跨文化交流日益频繁，语言能力成为衡量人才综合素质的重要指标。本文从理论与实践相结合的角度出发，系统分析了当前大学生英语学习中存在的主要问题，深入探讨了提升英语综合应用能力的有效路径，具有一定的理论价值和实践意义。",
    "新媒体时代的到来深刻改变了信息传播的方式与格局。短视频、社交平台等新兴媒介形式不仅重塑了大众的阅读习惯，也对青年群体的价值观念产生了深远影响。因此，加强青年网络素养教育显得尤为重要和紧迫。",
    "绿色低碳发展是实现经济社会可持续发展的必然要求。本文基于相关理论框架，构建了评价指标体系，运用实证分析方法对典型区域的绿色发展水平进行测度，并根据研究结果提出了有针对性的政策建议。",
    "乡村振兴战略的实施为农村经济社会发展注入了新动能。本文围绕产业兴旺、生态宜居、乡风文明、治理有效、生活富裕总要求，分析了当前乡村振兴实践中面临的挑战，并探索了高质量发展的实现路径。",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_detector():
    print(f"Loading detector {DET_REPO} on {DEVICE} ...")
    # Prefer local detector cache under ai率检测
    kwargs = {"cache_dir": str(DET_CACHE)} if DET_CACHE.exists() else {}
    tokenizer = BertTokenizer.from_pretrained(DET_REPO, **kwargs)
    model = BertForSequenceClassification.from_pretrained(DET_REPO, **kwargs)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def ai_prob(text: str) -> float:
    tokenizer, model = load_detector()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        scores = model(**inputs).logits[0].softmax(0)
    # labels: 人类=0, AI=1
    return float(scores[1].item())


def load_rewrite_model(token: str | None):
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else torch.float32)
    print(f"Rewrite backend: {'cuda' if use_cuda else 'cpu'} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Prefer local merged model when fully present (post-download)
    merged_local = Path(r"D:\hf_models\paperpass\3b-merged")
    if (merged_local / "config.json").exists() and (merged_local / "model.safetensors").exists():
        print(f"Loading local merged model: {merged_local}")
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_local),
            torch_dtype=dtype,
            device_map="auto" if use_cuda else "cpu",
            trust_remote_code=True,
        )
        model.eval()
        return tok, model
    if (merged_local / "config.json").exists():
        print(f"Local merged incomplete (no model.safetensors); using LoRA path")


    if use_cuda:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            token=token,
        )
    else:
        # CPU path: full weights (no bitsandbytes). Slow but works without CUDA torch.
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=token,
        )
    model = PeftModel.from_pretrained(base, ADAPTER_ID, token=token)
    model.eval()
    return tok, model


@torch.inference_mode()
def rewrite(tok, model, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[-1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok_file = Path(r"D:\hf_models\tools\dailv\.hf_token")
    if not token and tok_file.exists():
        token = tok_file.read_text(encoding="utf-8").strip()
        os.environ["HF_TOKEN"] = token

    print("Loading rewrite model (3B+LoRA)...")
    tok, model = load_rewrite_model(token)
    rows = []
    for i, src in enumerate(SAMPLES, 1):
        print(f"[{i}/{len(SAMPLES)}] detect baseline...")
        before = ai_prob(src)
        print(f"  AI_before={before:.4f}; rewriting...")
        t0 = time.time()
        dst = rewrite(tok, model, src)
        print(f"  rewritten in {time.time()-t0:.1f}s; detect after...")
        after = ai_prob(dst)
        drop = before - after
        print(f"  AI_after={after:.4f} drop={drop*100:.2f}pp")
        rows.append(
            {
                "source": src,
                "rewritten": dst,
                "ai_before": before,
                "ai_after": after,
                "drop": drop,
            }
        )

    mean_before = sum(r["ai_before"] for r in rows) / len(rows)
    mean_after = sum(r["ai_after"] for r in rows) / len(rows)
    mean_drop = mean_before - mean_after
    passed = mean_drop >= 0.05
    report = {
        "mean_ai_before": mean_before,
        "mean_ai_after": mean_after,
        "mean_drop": mean_drop,
        "threshold": 0.05,
        "passed": passed,
        "rows": rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("mean_ai_before", "mean_ai_after", "mean_drop", "passed")},
            ensure_ascii=False,
        )
    )
    print(f"WROTE {REPORT}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
