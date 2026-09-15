#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-rewrite academic paragraphs via Dailv / llama-cli."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# Reuse Dailv entry
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dailv  # noqa: E402

PARAS = [
    (
        "摘要",
        "随着移动互联网、短视频、社交平台等新媒体业态的快速普及，网络空间已成为当代青年学习生活、社交娱乐、价值塑造的核心场域。海量网络信息在为青年成长提供便利资源的同时，也带来了信息泛滥、网络谣言、不良思潮、网络沉迷等诸多问题。青年作为网络使用的核心群体，其网络素养水平直接关系个人成长成才，也关乎网络生态建设与社会长治久安。本文结合新媒体时代网络传播特征，分析当代青年网络素养的现存问题，探索科学有效的培育路径，为新时代青年网络素养教育提供参考。",
    ),
    (
        "一、引言",
        "当前，新媒体技术飞速迭代，信息传播打破了时间与空间的限制，呈现出碎片化、即时化、多元化的传播特征。青年群体思维活跃、接受能力强，是新媒体最主要的使用者和传播者。网络不仅拓宽了青年的认知视野、丰富了其精神文化生活，也为青年表达自我、参与社会公共事务提供了平台。但与此同时，网络信息良莠不齐、虚拟空间监管存在漏洞、青年价值体系尚未完全成熟等因素，导致部分青年出现信息辨别能力薄弱、网络行为失范、价值认知偏差等问题。因此，系统开展青年网络素养培育工作，引导青年树立正确的网络观、规范网络行为、提升网络认知能力，具有重要的现实意义。",
    ),
    (
        "二、问题（信息甄别）",
        "首先，信息甄别与思辨能力不足。新媒体时代，短视频、自媒体文章等轻量化信息成为主流，碎片化的传播方式弱化了深度思考。面对海量繁杂的网络信息，部分青年缺乏独立思辨能力，难以区分真实信息与虚假谣言、优质内容与低俗信息，容易盲目跟风转发、被动接受片面观点，甚至被网络谣言、极端思潮误导。",
    ),
    (
        "二、问题（自律法治）",
        "其次，网络自律与法治意识淡薄。部分青年对网络空间的公共属性认知不足，错误认为网络是法外之地，在网络发言随意偏激，存在网络吐槽、恶意评论、跟风网暴、泄露他人信息等不规范行为，缺乏网络法治思维和责任意识。",
    ),
    (
        "二、问题（价值认知）",
        "最后，网络价值认知存在偏差。部分青年过度沉迷短视频、网络游戏，耗费大量时间精力，影响学习生活；同时，流量至上、享乐主义、拜金主义等不良网络风气，容易扭曲青年的价值取向，导致其功利化心态严重，缺乏正向的理想信念。",
    ),
    (
        "三、路径（校园教育）",
        "第一，构建常态化校园教育体系，夯实素养基础。高校和中小学应将网络素养教育纳入常态化德育教学体系，结合青年认知特点，开设网络信息辨别、网络法律法规、网络安全防护、网络文明素养等专题课程。摒弃枯燥的理论灌输模式，结合网络热点案例、真实网络乱象事件开展情景化教学，引导青年学会理性辨析网络信息、规范网络行为，树立正确的网络价值观。",
    ),
    (
        "三、路径（家庭引导）",
        "第二，强化家庭引导与陪伴，筑牢成长防线。家长作为青年成长的第一责任人，应主动学习网络知识，了解青少年网络使用习惯，摒弃“一刀切”的禁止式管理。通过良性沟通陪伴，引导青年合理规划上网时间，规避网络沉迷问题；同时主动关注孩子的网络社交动态，及时纠正不良上网行为，引导孩子树立自律、文明的上网习惯。",
    ),
    (
        "三、路径（平台责任）",
        "第三，压实平台主体责任，净化网络生态。各大新媒体平台应坚守社会责任，完善内容审核机制，加大对低俗、虚假、极端不良信息的筛查和清理力度，优化青年网络使用环境。同时依托算法推荐优化内容推送，多推送正能量、知识性、励志性的优质内容，以正向内容浸润青年思想，引导青年树立积极向上的生活态度和价值追求。",
    ),
    (
        "三、路径（自我赋能）",
        "第四，引导青年自我赋能，提升综合素养。青年自身应主动强化自我约束，树立终身学习意识，主动学习网络安全、网络法治相关知识，培养独立思考、理性判断的能力。自觉抵制网络低俗文化和不良思潮，规范自身网络言行，主动参与网络文明建设，争做文明上网、理性用网的新时代青年。",
    ),
    (
        "四、结语",
        "新媒体时代为青年成长带来了全新机遇，也带来了全新挑战。青年网络素养培育是一项系统性、长期性的工程，需要学校、家庭、平台、社会多方协同发力。唯有多方联动、多措并举，才能持续提升青年的信息辨别能力、网络法治意识、自我约束能力和价值塑造能力，引导青年正确认识网络、合理使用网络，让网络真正成为青年成长成才的助力载体，助力新时代青年健康全面发展。",
    ),
]


def rewrite_one(text: str) -> str:
    llama = dailv.find_llama_cli()
    threads = int(__import__("os").environ.get("DAILV_THREADS", str(dailv.DEFAULT_THREADS)))
    device = __import__("os").environ.get("DAILV_DEVICE", dailv.DEFAULT_DEVICE)
    cmd = [
        llama,
        "-m",
        dailv.MODEL_PATH,
        "-dev",
        device,
        "-ngl",
        str(dailv.DEFAULT_NGL),
        "-t",
        str(threads),
        "-c",
        str(dailv.DEFAULT_CTX),
        "-n",
        str(min(dailv.DEFAULT_N_PREDICT, max(256, dailv.DEFAULT_CTX))),
        "-sys",
        dailv.SYSTEM_PROMPT,
        "-p",
        text,
        "-st",
        "--no-display-prompt",
        "--simple-io",
        "--log-disable",
    ]
    # Capture full output without typewriter (batch)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    out = (proc.stdout or "").strip()
    # Strip llama banners / prompts if any leaked
    for marker in dailv.STOP_MARKERS:
        if marker in out:
            out = out[: out.find(marker)].rstrip()
    # Drop common CLI chrome lines
    lines = []
    skip_prefixes = ("Loading model", "build ", "model ", "ftype ", "modalities", "available commands", "▄", "█", "▀")
    for ln in out.splitlines():
        s = ln.strip()
        if not s:
            if lines:
                lines.append("")
            continue
        if any(s.startswith(p) or p in s[:3] for p in ("▄", "█", "▀")):
            continue
        if s.startswith(skip_prefixes):
            continue
        if s.startswith("/exit") or s.startswith("/regen"):
            continue
        lines.append(ln)
    cleaned = "\n".join(lines).strip()
    # Sometimes model echoes; take last substantial block
    if not cleaned and proc.stderr:
        return f"[ERROR exit={proc.returncode}] {proc.stderr[-500:]}"
    return cleaned or out


def main() -> int:
    out_path = Path(r"D:\hf_models\tools\dailv\rewrite_result.jsonl")
    results = []
    for i, (title, text) in enumerate(PARAS, 1):
        print(f"[{i}/{len(PARAS)}] {title} ...", flush=True)
        t0 = time.time()
        try:
            rewritten = rewrite_one(text)
        except Exception as e:
            rewritten = f"[EXCEPTION] {e}"
        elapsed = time.time() - t0
        item = {"title": title, "original": text, "rewritten": rewritten, "elapsed_s": round(elapsed, 1)}
        results.append(item)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  done in {elapsed:.1f}s, chars={len(rewritten)}", flush=True)

    md_path = Path(r"D:\hf_models\tools\dailv\rewrite_result.md")
    parts = ["# 改写结果\n"]
    for item in results:
        parts.append(f"## {item['title']}\n")
        parts.append(f"**原文**\n\n{item['original']}\n")
        parts.append(f"**改写**\n\n{item['rewritten']}\n")
    md_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"WROTE {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    # fresh output
    p = Path(r"D:\hf_models\tools\dailv\rewrite_result.jsonl")
    if p.exists():
        p.unlink()
    raise SystemExit(main())
