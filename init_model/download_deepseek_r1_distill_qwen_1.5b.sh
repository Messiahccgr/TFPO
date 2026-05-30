#!/usr/bin/env bash
# Download DeepSeek-R1-Distill-Qwen-1.5B into ./init_model/DeepSeek-R1-Distill-Qwen-1.5B
# Usage:
#   bash init_model/download_deepseek_r1_distill_qwen_1.5b.sh
# Env (optional):
#   HF_ENDPOINT      - mirror, e.g. https://hf-mirror.com
#   HF_TOKEN         - if the model becomes gated
#   HF_HUB_ENABLE_HF_TRANSFER=1 - faster transfer (requires `pip install hf_transfer`)

set -euo pipefail

REPO_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/DeepSeek-R1-Distill-Qwen-1.5B"

mkdir -p "${TARGET_DIR}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install with: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

echo "Downloading ${REPO_ID} -> ${TARGET_DIR}"

# Skip optional / duplicate weight artifacts to save disk and bandwidth.
huggingface-cli download "${REPO_ID}" \
  --local-dir "${TARGET_DIR}" \
  --local-dir-use-symlinks False \
  --exclude "*.gguf" "*.msgpack" "consolidated.*" "original/*"

echo "Done. Model files in ${TARGET_DIR}"
