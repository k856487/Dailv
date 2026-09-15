#!/usr/bin/env python3
"""Submit a UV job from k86574/train-scripts. Prints job id."""
from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: submit_uv_job.py SCRIPT FLAVOR TIMEOUT NAME", file=sys.stderr)
        return 2
    script, flavor, timeout, name = sys.argv[1:5]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    url = f"https://huggingface.co/k86574/train-scripts/resolve/main/{script}"
    job = api.run_uv_job(
        url,
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": token},
        name=name,
    )
    print(job.id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
