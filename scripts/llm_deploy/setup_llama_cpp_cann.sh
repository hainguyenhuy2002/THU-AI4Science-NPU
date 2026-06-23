#!/usr/bin/env bash
# Build llama.cpp with the Ascend CANN backend.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../common/ascend_env.sh
source scripts/common/ascend_env.sh

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-$LLAMA_CPP_DIR/build}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "[ERROR] cmake is required." >&2
  exit 1
fi

if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  git clone --depth=1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
else
  git -C "$LLAMA_CPP_DIR" pull --ff-only
fi

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_BUILD_DIR" \
  -DGGML_CANN=on \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$LLAMA_CPP_BUILD_DIR" --config Release -j"$(nproc)" --target llama-server llama-quantize

echo "[OK] llama-server: $LLAMA_CPP_BUILD_DIR/bin/llama-server"
