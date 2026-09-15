#!/usr/bin/env python3
from __future__ import annotations

import os

from huggingface_hub import HfApi


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    canceled = []
    for j in api.list_jobs():
        stage = (getattr(getattr(j, "status", None), "stage", None) or "").upper()
        jid = getattr(j, "id", None)
        if not jid:
            continue
        if stage in ("RUNNING", "SCHEDULING", "PENDING", "UPDATING"):
            try:
                api.cancel_job(job_id=jid)
                canceled.append(f"{jid}:{stage}")
            except Exception as e:  # noqa: BLE001
                canceled.append(f"{jid}:err:{e}")
    print("CANCELED", ",".join(canceled) if canceled else "none")


if __name__ == "__main__":
    main()
