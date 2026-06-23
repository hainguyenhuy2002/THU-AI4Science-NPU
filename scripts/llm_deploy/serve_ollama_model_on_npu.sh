#!/usr/bin/env bash
# Serve an Ollama-pulled GGUF model with llama.cpp CANN on Ascend NPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../common/ascend_env.sh
source scripts/common/ascend_env.sh

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.3:70b}"
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama 2>/dev/null || true)}"
MODEL_ALIAS="${MODEL_ALIAS:-${OLLAMA_MODEL/:/-}-ascend}"
MODEL_PATH="${MODEL_PATH:-}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama.cpp/build/bin/llama-server}"
LLAMA_CPP_PORT="${LLAMA_CPP_PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-4096}"
THREADS="${THREADS:-$(nproc)}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
LOG_DIR="${LOG_DIR:-logs}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

mkdir -p "$LOG_DIR"

if [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "[ERROR] Missing $LLAMA_SERVER_BIN. Run scripts/llm_deploy/setup_llama_cpp_cann.sh first." >&2
  exit 1
fi

if [ -z "$OLLAMA_BIN" ] && [ -x "$HOME/.local/bin/ollama" ]; then
  OLLAMA_BIN="$HOME/.local/bin/ollama"
fi

if [ -z "$MODEL_PATH" ]; then
  echo "[INFO] Finding Ollama GGUF blob for $OLLAMA_MODEL"
  MODEL_PATH="$(OLLAMA_BIN="$OLLAMA_BIN" python3 scripts/llm_deploy/find_ollama_gguf.py "$OLLAMA_MODEL")"
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

pkill -f "llama-server.*--port $LLAMA_CPP_PORT" 2>/dev/null || true

echo "[INFO] Serving $MODEL_PATH on Ascend device(s): $ASCEND_VISIBLE_DEVICES"
nohup "$LLAMA_SERVER_BIN" \
  --host 0.0.0.0 \
  --port "$LLAMA_CPP_PORT" \
  --model "$MODEL_PATH" \
  --alias "$MODEL_ALIAS" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  ${LLAMA_SERVER_EXTRA_ARGS:-} \
  > "$LOG_DIR/llm_deploy.log" 2>&1 &

echo "$!" > "$LOG_DIR/llama-server.pid"
echo "[INFO] llama-server PID: $(cat "$LOG_DIR/llama-server.pid")"

for _ in $(seq 1 "$READY_TIMEOUT"); do
  if curl -sf "http://localhost:$LLAMA_CPP_PORT/health" >/dev/null 2>&1; then
    echo "[OK] llama.cpp server is ready on http://localhost:$LLAMA_CPP_PORT"
    exit 0
  fi
  sleep 1
done

echo "[ERROR] llama.cpp server did not become ready. Check $LOG_DIR/llm_deploy.log" >&2
exit 1
