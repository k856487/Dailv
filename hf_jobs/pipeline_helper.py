#!/usr/bin/env python3
"""Helpers for unattended HF pipeline (token validate, job poll, dataset check)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

TOKEN_FILE = Path(r"D:\hf_models\tools\dailv\.hf_token")
DISTILL_DONE_FLAG = Path(r"D:\hf_models\tools\dailv\distill_done.flag")
DISTILL_DATASET = "k86574/paperpass-rewrite-distill-3b"


def load_token() -> str | None:
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    return None


def validate_token(token: str | None) -> bool:
    if not token:
        return False
    try:
        from huggingface_hub import whoami

        whoami(token=token)
        return True
    except Exception:
        return False


def wait_valid_token(poll_sec: int = 60) -> str:
    print(f"Waiting for valid HF token in {TOKEN_FILE} ...", flush=True)
    while True:
        tok = load_token()
        if validate_token(tok):
            os.environ["HF_TOKEN"] = tok  # type: ignore[arg-type]
            print("TOKEN_OK", flush=True)
            return tok  # type: ignore[return-value]
        print("TOKEN_INVALID sleep", poll_sec, flush=True)
        time.sleep(poll_sec)


def job_stage(job_id: str, token: str) -> str | None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.inspect_job(job_id=job_id)
    st = getattr(getattr(info, "status", None), "stage", None)
    return str(st) if st else None


def dataset_exists(token: str) -> bool:
    if DISTILL_DONE_FLAG.exists():
        return True
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.repo_info(DISTILL_DATASET, repo_type="dataset")
        return True
    except Exception:
        return False


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "validate":
        ok = validate_token(load_token())
        print("OK" if ok else "FAIL")
        return 0 if ok else 1
    if cmd == "wait-token":
        wait_valid_token(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
        return 0
    if cmd == "job-stage":
        tok = load_token()
        if not validate_token(tok):
            print("NO_TOKEN")
            return 2
        print(job_stage(sys.argv[2], tok))
        return 0
    if cmd == "dataset-ready":
        tok = load_token()
        if not validate_token(tok):
            print("NO_TOKEN")
            return 2
        print("YES" if dataset_exists(tok) else "NO")
        return 0
    print("usage: validate | wait-token [sec] | job-stage JOBID | dataset-ready")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
