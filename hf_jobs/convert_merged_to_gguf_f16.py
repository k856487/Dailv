# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers>=4.51.0",
#     "accelerate>=1.0.0",
#     "huggingface_hub>=0.30.0",
#     "safetensors",
#     "sentencepiece>=0.1.99",
#     "protobuf>=3.20.0",
#     "numpy",
#     "gguf",
# ]
# ///
"""Convert merged HF model to highest-precision GGUF (F16)."""

from __future__ import annotations

import os
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from huggingface_hub import HfApi, login, snapshot_download

MERGED_ID = os.environ.get("MERGED_MODEL", "k86574/qwen25-7b-paperpass-rewrite-merged")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "k86574/qwen25-7b-paperpass-rewrite-gguf")
OUT_NAME = "qwen25-7b-paperpass-rewrite-f16.gguf"


def run(cmd: list[str], desc: str) -> None:
    print(f"==> {desc}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    login(token=token)

    run(["apt-get", "update", "-qq"], "apt update")
    run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake", "git"], "install build tools")

    merged_dir = snapshot_download(MERGED_ID, token=token)
    print(f"Merged model at {merged_dir}", flush=True)

    run(
        ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", "/tmp/llama.cpp"],
        "clone llama.cpp",
    )

    os.makedirs("/tmp/gguf", exist_ok=True)
    f16_path = f"/tmp/gguf/{OUT_NAME}"
    run(
        [
            sys.executable,
            "/tmp/llama.cpp/convert_hf_to_gguf.py",
            merged_dir,
            "--outfile",
            f16_path,
            "--outtype",
            "f16",
        ],
        "convert HF -> GGUF F16",
    )

    size_gb = os.path.getsize(f16_path) / 1e9
    print(f"F16 GGUF size: {size_gb:.2f} GB", flush=True)

    api = HfApi(token=token)
    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True, repo_type="model")
    api.upload_file(
        path_or_fileobj=f16_path,
        path_in_repo=OUT_NAME,
        repo_id=OUTPUT_REPO,
        repo_type="model",
        commit_message="Add F16 GGUF (highest precision)",
    )
    # remove old Q4 if present
    try:
        api.delete_file(
            path_in_repo="qwen25-7b-paperpass-rewrite-q4_k_m.gguf",
            repo_id=OUTPUT_REPO,
            repo_type="model",
            commit_message="Remove Q4_K_M GGUF",
        )
        print("Deleted Q4_K_M from Hub", flush=True)
    except Exception as e:
        print(f"Q4 delete skipped: {e}", flush=True)

    print(f"Done: https://huggingface.co/{OUTPUT_REPO}", flush=True)


if __name__ == "__main__":
    main()
