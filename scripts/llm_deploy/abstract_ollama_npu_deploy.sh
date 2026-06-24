#!/usr/bin/env bash
# Abstract Ollama deployment example: GPU vs Ascend NPU.

set -euo pipefail

# =========================
# GPU version, for reference
# =========================
# export CUDA_VISIBLE_DEVICES=0
# cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=on
# llama.cpp/build/bin/llama-server \
#   --model /path/to/model.gguf \
#   --alias model-cuda \
#   --n-gpu-layers -1
# nvidia-smi

# =========================
# NPU version
# =========================
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_VISIBLE_DEVICES}"

# shellcheck source=../common/ascend_env.sh
source "$(dirname "$0")/../common/ascend_env.sh"

MODEL_PATH="${MODEL_PATH:-/path/to/model.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-model-ascend}"
LLAMA_CPP_PORT="${LLAMA_CPP_PORT:-18080}"

cmake -S llama.cpp -B llama.cpp/build -DGGML_CANN=on

llama.cpp/build/bin/llama-server \
  --host 0.0.0.0 \
  --port "$LLAMA_CPP_PORT" \
  --model "$MODEL_PATH" \
  --alias "$MODEL_ALIAS" \
  --n-gpu-layers -1

# Verify with:
# npu-smi info
