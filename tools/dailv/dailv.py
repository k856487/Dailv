#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dailv — 本地 llama.cpp 学术段落改写 CLI。

  Dailv7 / Dailv  → 7B（约 280 字/段）
  Dailv3          → 3B（约 220 字/段）

长文按最适字数自动拆分，改写后按原文段落结构拼接。
"""

from __future__ import annotations

import argparse
import codecs
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Runtime defaults
# ---------------------------------------------------------------------------

THREADS = int(os.environ.get("DAILV_THREADS", "16"))
DEVICE = os.environ.get("DAILV_DEVICE", "Vulkan0")
CHAR_DELAY = float(os.environ.get("DAILV_CHAR_DELAY", "0.032"))
CHAR_DELAY_FAST = 0.008
CHAR_DELAY_CATCHUP = 0.002

STOP_MARKERS = ("\n[ Prompt:", "\nExiting...", "\nExiting")
RESULT_LABEL = "改写结果："
THINKING_FRAMES = ("思考中.", "思考中..", "思考中...")
_DONE = object()

_SENT_SPLIT = re.compile(r"(?<=[。！？；;!?])")
_PARA_SPLIT = re.compile(r"(\n\s*\n)")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    name: str
    model_path: str
    ngl: int
    ctx: int
    n_predict: int
    target_chars: int
    max_chars: int
    min_chars: int
    system_prompt: str
    temperature: float | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


PROFILES: dict[str, Profile] = {
    "7b": Profile(
        name="7b",
        model_path=_env(
            "DAILV7_MODEL",
            r"D:\hf_models\paperpass\gguf\qwen25-7b-paperpass-rewrite-f16.gguf",
        ),
        ngl=int(_env("DAILV7_NGL", "11")),
        ctx=int(_env("DAILV7_CTX", "1024")),
        n_predict=int(_env("DAILV7_N_PREDICT", "1024")),
        target_chars=280,
        max_chars=360,
        min_chars=120,
        system_prompt=(
            "你是一名专业的中文学术文本润色助手。"
            "请将用户给出的论文段落改写得更加自然、流畅、易读，"
            "保持原意与专业术语准确，不编造事实，不增删核心论点。"
            "只输出改写后的段落正文，不要解释、不要加标题或多余说明。"
        ),
    ),
    "3b": Profile(
        name="3b",
        model_path=_env(
            "DAILV3_MODEL",
            r"D:\hf_models\paperpass\gguf\qwen25-3b-paperpass-rewrite-f16.gguf",
        ),
        ngl=int(_env("DAILV3_NGL", "28")),
        ctx=int(_env("DAILV3_CTX", "1024")),
        n_predict=int(_env("DAILV3_N_PREDICT", "1024")),
        target_chars=220,
        max_chars=300,
        min_chars=100,
        system_prompt=(
            "你是一名中文学术改写助手。将用户段落改写成更像人工撰写的学术表述："
            "避免空泛套话、排比堆砌、模板化起承转合；句式长短交错，用词具体；"
            "保持原意与术语准确，不编造事实。"
            "只输出改写后的段落正文，不要解释。"
        ),
        temperature=float(_env("DAILV3_TEMP", "0.7")),
        top_p=float(_env("DAILV3_TOP_P", "0.9")),
        repeat_penalty=float(_env("DAILV3_REPEAT", "1.15")),
    ),
}

# rewrite_batch.py compatibility
DEFAULT_NGL = PROFILES["7b"].ngl
DEFAULT_CTX = PROFILES["7b"].ctx
DEFAULT_N_PREDICT = PROFILES["7b"].n_predict
DEFAULT_THREADS = THREADS
DEFAULT_DEVICE = DEVICE
MODEL_PATH = PROFILES["7b"].model_path
SYSTEM_PROMPT = PROFILES["7b"].system_prompt


@dataclass(frozen=True)
class RunConfig:
    profile: Profile
    text: str
    ngl: int
    ctx: int
    no_split: bool
    quiet_progress: bool


# ---------------------------------------------------------------------------
# Text split / pack
# ---------------------------------------------------------------------------

def split_text(text: str, profile: Profile) -> list[tuple[str, str]]:
    """[(chunk, suffix_after), ...] — suffix restores original paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    parts = _PARA_SPLIT.split(text)
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(parts):
        block = parts[i]
        i += 1
        sep = ""
        if i < len(parts) and _PARA_SPLIT.fullmatch(parts[i] or ""):
            sep = "\n\n"
            i += 1
        block = block.strip()
        if not block:
            continue
        pieces = _pack_paragraph(block, profile)
        for j, piece in enumerate(pieces):
            out.append((piece, sep if j == len(pieces) - 1 else ""))
    return out or [(text, "")]


def _pack_paragraph(para: str, profile: Profile) -> list[str]:
    if len(para) <= profile.max_chars:
        return [para]

    sentences = [s.strip() for s in _SENT_SPLIT.split(para) if s.strip()]
    if len(sentences) <= 1:
        return [para[i : i + profile.max_chars] for i in range(0, len(para), profile.max_chars)]

    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if len(sent) > profile.max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(
                sent[i : i + profile.max_chars] for i in range(0, len(sent), profile.max_chars)
            )
            continue
        if not buf:
            buf = sent
            continue
        merged = buf + sent
        near_target = len(buf) < profile.target_chars or len(merged) <= profile.target_chars + 40
        if len(merged) <= profile.max_chars and near_target:
            buf = merged
        elif len(buf) < profile.min_chars and len(merged) <= profile.max_chars:
            buf = merged
        else:
            chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)

    if (
        len(chunks) >= 2
        and len(chunks[-1]) < profile.min_chars
        and len(chunks[-2]) + len(chunks[-1]) <= profile.max_chars + 40
    ):
        tail = chunks.pop()
        chunks[-1] += tail
    return chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_llama_cli() -> str:
    env = os.environ.get("DAILV_LLAMA_CLI")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("llama-cli")
    if found:
        return found
    raise FileNotFoundError(
        "找不到 llama-cli。请确认已安装 llama.cpp，或设置环境变量 DAILV_LLAMA_CLI。"
    )


def _normalize_profile(name: str) -> str:
    n = name.lower().strip()
    aliases = {
        "3": "3b",
        "3b": "3b",
        "dailv3": "3b",
        "7": "7b",
        "7b": "7b",
        "dailv": "7b",
        "dailv7": "7b",
    }
    return aliases.get(n, n)


def parse_args(argv: list[str] | None = None) -> RunConfig:
    argv = list(sys.argv[1:] if argv is None else argv)
    default = _normalize_profile(_env("DAILV_PROFILE", "7b"))
    if default not in PROFILES:
        default = "7b"

    p = argparse.ArgumentParser(
        prog="Dailv",
        description="学术段落改写（自动按最适字数拆分并拼接）",
    )
    p.add_argument("--profile", choices=sorted(PROFILES), default=default)
    p.add_argument("--no-split", action="store_true", help="禁用自动拆分")
    p.add_argument("--quiet-progress", action="store_true", help="不打印拆分进度")
    p.add_argument("text_and_opts", nargs="*", help="待改写文本；末尾可跟 ngl [ctx]")

    ns, unknown = p.parse_known_args(argv)
    tokens = list(ns.text_and_opts)
    for u in unknown:
        if u.startswith("-"):
            p.error(f"未知参数: {u}")
        tokens.append(u)

    if not tokens:
        p7, p3 = PROFILES["7b"], PROFILES["3b"]
        p.print_help(sys.stderr)
        print(
            f"\n推荐一次改写字数:\n"
            f"  Dailv7: 约 {p7.target_chars} 字（上限 {p7.max_chars}）\n"
            f"  Dailv3: 约 {p3.target_chars} 字（上限 {p3.max_chars}）\n"
            f"\n示例:\n"
            f"  Dailv7 本研究探讨了新媒体环境下的阅读习惯变迁。\n"
            f"  Dailv3 本研究探讨了新媒体环境下的阅读习惯变迁。\n"
            f"  Dailv7 较长文段... 11 1024\n",
            file=sys.stderr,
        )
        sys.exit(1)

    profile = PROFILES[ns.profile]
    ngl, ctx = profile.ngl, profile.ctx
    if len(tokens) >= 2 and tokens[-1].isdigit() and tokens[-2].isdigit():
        ngl, ctx = int(tokens[-2]), int(tokens[-1])
        tokens = tokens[:-2]
    elif tokens[-1].isdigit():
        ngl = int(tokens[-1])
        tokens = tokens[:-1]

    text = " ".join(tokens).strip()
    if not text:
        print("错误: 请提供待改写段落。", file=sys.stderr)
        sys.exit(1)
    if ngl < 0 or ctx < 256:
        print("错误: ngl 须 >= 0，上下文建议 >= 256。", file=sys.stderr)
        sys.exit(1)
    return RunConfig(profile, text, ngl, ctx, ns.no_split, ns.quiet_progress)


# ---------------------------------------------------------------------------
# llama.cpp I/O + UI
# ---------------------------------------------------------------------------

def _enable_ansi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _out(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def _find_stop(window: str) -> int | None:
    hits = [window.find(m) for m in STOP_MARKERS]
    hits = [p for p in hits if p >= 0]
    return min(hits) if hits else None


def _stop_prefix_hold(window: str) -> int:
    hold = 0
    for marker in STOP_MARKERS:
        for i in range(1, len(marker)):
            if window.endswith(marker[:i]):
                hold = max(hold, i)
    return hold


def _char_delay(backlog: int) -> float:
    if backlog >= 80:
        return CHAR_DELAY_CATCHUP
    if backlog >= 20:
        return CHAR_DELAY_FAST
    return CHAR_DELAY


def _thinking_loop(stop: threading.Event, done: threading.Event) -> None:
    """Blank line +「思考中」+ blank line; animate until stop."""
    try:
        _enable_ansi()
        _out("\n")
        primed = False
        i = 0
        while not stop.is_set():
            frame = THINKING_FRAMES[i % 3]
            pad = " " * (3 - len(frame) + len("思考中"))
            if not primed:
                _out(frame + pad + "\n\n\033[2A\r")
                primed = True
            else:
                _out("\r" + frame + pad)
            i += 1
            if stop.wait(0.4):
                break
    finally:
        done.set()


def _clear_thinking() -> None:
    _out("\r" + " " * 16 + "\r")


def _reveal_loop(
    q: queue.Queue,
    printed: list[bool],
    err: list,
    thinking_stop: threading.Event,
    thinking_done: threading.Event,
    result_label: str,
    show_thinking: bool,
) -> None:
    try:
        first = True
        while True:
            ch = q.get()
            if ch is _DONE:
                break
            if first:
                thinking_stop.set()
                thinking_done.wait(timeout=2.0)
                if show_thinking:
                    _clear_thinking()
                if result_label:
                    _out(result_label + "\n\n")
                first = False
            sys.stdout.write(ch)
            sys.stdout.flush()
            printed[0] = True
            delay = _char_delay(q.qsize())
            if delay > 0:
                time.sleep(delay)
    except Exception as e:
        err.append(e)
    finally:
        thinking_stop.set()
        thinking_done.wait(timeout=0.5)
        if show_thinking and not printed[0]:
            _clear_thinking()


def _enqueue(q: queue.Queue, text: str, collected: list[str]) -> None:
    for ch in text:
        if ch == "\r":
            continue
        q.put(ch)
        collected.append(ch)


def _produce_llama(
    cmd: list[str],
    q: queue.Queue,
    debug: bool,
    result: dict,
    collected: list[str],
) -> None:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if not debug else None,
            bufsize=0,
        )
    except OSError as e:
        result.update(rc=1, err=f"启动 llama-cli 失败: {e}")
        q.put(_DONE)
        return

    assert proc.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    window = ""
    streaming = False
    err_chunks: list[bytes] = []

    try:
        while True:
            b = proc.stdout.read(1)
            if not b:
                break
            piece = decoder.decode(b).replace("\r", "")
            if not piece:
                continue

            if not streaming:
                window += piece
                if len(window) > 8000:
                    window = window[-4000:]
                idx = window.rfind("\n> ")
                if idx < 0:
                    continue
                after = window[idx + 3 :]
                nl = after.find("\n")
                if nl < 0:
                    continue
                streaming = True
                window = after[nl + 1 :].lstrip("\n")
            else:
                window += piece

            stop_at = _find_stop(window)
            if stop_at is not None:
                _enqueue(q, window[:stop_at].rstrip("\n"), collected)
                while proc.stdout.read(65536):
                    pass
                window = ""
                break

            hold = _stop_prefix_hold(window)
            ready, window = (window[:-hold], window[-hold:]) if hold else (window, "")
            if ready:
                _enqueue(q, ready, collected)

        if streaming:
            window += decoder.decode(b"", final=True).replace("\r", "")
            stop_at = _find_stop(window)
            if stop_at is not None:
                window = window[:stop_at]
            _enqueue(q, window.rstrip("\n"), collected)
    except Exception as e:
        result.update(err=str(e), rc=1)
    finally:
        try:
            if proc.stderr is not None:
                err_chunks.append(proc.stderr.read() or b"")
        except Exception:
            pass
        result["rc"] = proc.wait()
        result["stderr"] = b"".join(err_chunks).decode("utf-8", "replace")
        q.put(_DONE)


def stream_rewrite(
    cmd: list[str],
    *,
    debug: bool = False,
    end: str = "\n",
    result_label: str = RESULT_LABEL,
    show_thinking: bool = True,
) -> tuple[int, str]:
    q: queue.Queue = queue.Queue()
    printed = [False]
    reveal_err: list = []
    result: dict = {"rc": 1, "stderr": "", "err": ""}
    collected: list[str] = []
    thinking_stop = threading.Event()
    thinking_done = threading.Event()

    thinker = None
    if show_thinking:
        thinker = threading.Thread(
            target=_thinking_loop, args=(thinking_stop, thinking_done), daemon=True
        )
        thinker.start()
    else:
        thinking_done.set()

    consumer = threading.Thread(
        target=_reveal_loop,
        args=(
            q,
            printed,
            reveal_err,
            thinking_stop,
            thinking_done,
            result_label,
            show_thinking,
        ),
        daemon=True,
    )
    producer = threading.Thread(
        target=_produce_llama, args=(cmd, q, debug, result, collected), daemon=True
    )
    consumer.start()
    producer.start()
    producer.join()
    consumer.join()
    thinking_stop.set()
    if thinker:
        thinker.join(timeout=1.0)

    if printed[0] and end:
        _out(end)

    if reveal_err:
        print(f"显示线程错误: {reveal_err[0]}", file=sys.stderr)
    if debug and result.get("stderr"):
        sys.stderr.write(result["stderr"])
    if result.get("err") and result["rc"] != 0:
        print(result["err"], file=sys.stderr)

    rc = int(result.get("rc", 1))
    err_text = result.get("stderr") or ""
    if rc != 0:
        if "OutOfDeviceMemory" in err_text or "failed to allocate" in err_text.lower():
            print(
                "显存不足。请减小 ngl，例如: Dailv7 <段落> 8 1024  或  Dailv3 <段落> 20 1024",
                file=sys.stderr,
            )
        elif not result.get("err"):
            print(f"推理失败 (exit {rc}):\n{err_text[-2000:]}", file=sys.stderr)
    return rc, "".join(collected)


def build_cmd(llama: str, profile: Profile, text: str, ngl: int, ctx: int) -> list[str]:
    cmd = [
        llama,
        "-m",
        profile.model_path,
        "-dev",
        DEVICE,
        "-ngl",
        str(ngl),
        "-t",
        str(THREADS),
        "-c",
        str(ctx),
        "-n",
        str(min(profile.n_predict, max(256, ctx))),
        "-sys",
        profile.system_prompt,
        "-p",
        text,
        "-st",
        "--no-display-prompt",
        "--simple-io",
        "--log-disable",
    ]
    if profile.temperature is not None:
        cmd += ["--temp", str(profile.temperature)]
    if profile.top_p is not None:
        cmd += ["--top-p", str(profile.top_p)]
    if profile.repeat_penalty is not None:
        cmd += ["--repeat-penalty", str(profile.repeat_penalty)]
    return cmd


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    profile = cfg.profile

    if not os.path.isfile(profile.model_path):
        print(f"错误: 模型不存在: {profile.model_path}", file=sys.stderr)
        if profile.name == "3b":
            print("提示: 设置环境变量 DAILV3_MODEL 指向 F16 GGUF。", file=sys.stderr)
        return 1

    try:
        llama = find_llama_cli()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    debug = os.environ.get("DAILV_DEBUG") == "1"
    units = [(cfg.text, "")] if cfg.no_split else split_text(cfg.text, profile)

    if len(units) > 1 and not cfg.quiet_progress:
        print(
            f"[Dailv{profile.name[0]}] 拆成 {len(units)} 段"
            f"（目标约 {profile.target_chars} 字，上限 {profile.max_chars}）",
            file=sys.stderr,
        )

    for i, (chunk, suffix) in enumerate(units):
        if len(units) > 1 and not cfg.quiet_progress:
            print(f"[{i + 1}/{len(units)}] {len(chunk)} 字…", file=sys.stderr)

        end = "\n" if i == len(units) - 1 else (suffix or "")
        rc, out = stream_rewrite(
            build_cmd(llama, profile, chunk, cfg.ngl, cfg.ctx),
            debug=debug,
            end=end,
            result_label=RESULT_LABEL if i == 0 else "",
            show_thinking=(i == 0),
        )
        if rc != 0:
            return rc
        if not out.strip():
            print(f"错误: 第 {i + 1} 段改写结果为空。", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
