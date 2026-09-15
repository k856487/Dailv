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
"""Convert an already-merged HF model to GGUF (F16 + Q4_K_M) with llama.cpp."""

from __future__ import annotations

import os
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from huggingface_hub import HfApi, login, snapshot_download

MERGED_ID = os.environ.get("MERGED_MODEL", "k86574/qwen25-7b-paperpass-rewrite-merged")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "k86574/qwen25-7b-paperpass-rewrite-gguf")
QUANT = os.environ.get("QUANT", "Q4_K_M")


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

    run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", "/tmp/llama.cpp"], "clone llama.cpp")

    os.makedirs("/tmp/gguf", exist_ok=True)
    f16_path = "/tmp/gguf/qwen25-7b-paperpass-rewrite-f16.gguf"
    run(
        [sys.executable, "/tmp/llama.cpp/convert_hf_to_gguf.py", merged_dir, "--outfile", f16_path, "--outtype", "f16"],
        "convert HF -> GGUF F16",
    )

    run(
        ["cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp", "-DGGML_CUDA=OFF"],
        "cmake configure",
    )
    run(
        ["cmake", "--build", "/tmp/llama.cpp/build", "--target", "llama-quantize", "-j", "4"],
        "build llama-quantize",
    )

    quant_bin = "/tmp/llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quant_bin):
        # fallback older path
        alt = "/tmp/llama.cpp/build/bin/Release/llama-quantize"
        quant_bin = alt if os.path.exists(alt) else quant_bin

    q_path = f"/tmp/gguf/qwen25-7b-paperpass-rewrite-{QUANT.lower()}.gguf"
    run([quant_bin, f16_path, q_path, QUANT], f"quantize to {QUANT}")

    api = HfApi(token=token)
    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True, repo_type="model")
    api.upload_file(path_or_fileobj=q_path, path_in_repo=os.path.basename(q_path), repo_id=OUTPUT_REPO, repo_type="model")
    # F16 is large; skip upload to save storage/cost unless needed
    print(f"Done. GGUF at https://huggingface.co/{OUTPUT_REPO}")
    print(f"Local job file: {q_path} size={os.path.getsize(q_path)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
