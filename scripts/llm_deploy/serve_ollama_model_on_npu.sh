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
LLAMA_CPP_HOST="${LLAMA_CPP_HOST:-127.0.0.1}"
LLAMA_CPP_PORT="${LLAMA_CPP_PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-4096}"
THREADS="${THREADS:-$(nproc)}"
N_GPU_LAYERS="${N_GPU_LAYERS:-all}"
LOG_DIR="${LOG_DIR:-logs}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
read -r -a extra_args <<< "${LLAMA_SERVER_EXTRA_ARGS:-}"

mkdir -p "$LOG_DIR"

if [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "[ERROR] Missing $LLAMA_SERVER_BIN. Run scripts/llm_deploy/setup_llama_cpp_cann.sh first." >&2
  exit 1
fi

if [ -z "$MODEL_PATH" ]; then
  if [ -z "$OLLAMA_BIN" ] && [ -x "$HOME/.local/bin/ollama" ]; then
    OLLAMA_BIN="$HOME/.local/bin/ollama"
  fi
  if [ -z "$OLLAMA_BIN" ]; then
    echo "[ERROR] Could not find ollama. Set OLLAMA_BIN=/path/to/ollama." >&2
    exit 1
  fi

  ollama_url="http://$OLLAMA_HOST/api/tags"
  if ! curl --noproxy '*' -fsS --max-time 2 "$ollama_url" >/dev/null 2>&1; then
    echo "[INFO] Starting Ollama daemon on $OLLAMA_HOST"
    nohup "$OLLAMA_BIN" serve > "$LOG_DIR/ollama.log" 2>&1 &
    ollama_pid=$!
    echo "$ollama_pid" > "$LOG_DIR/ollama.pid"
    for _ in $(seq 1 60); do
      if curl --noproxy '*' -fsS --max-time 2 "$ollama_url" >/dev/null 2>&1; then
        break
      fi
      if ! kill -0 "$ollama_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if ! curl --noproxy '*' -fsS --max-time 2 "$ollama_url" >/dev/null 2>&1; then
      echo "[ERROR] Ollama did not become ready. Check $LOG_DIR/ollama.log" >&2
      exit 1
    fi
  fi

  echo "[INFO] Finding Ollama GGUF blob for $OLLAMA_MODEL"
  MODEL_PATH="$(OLLAMA_BIN="$OLLAMA_BIN" python3 scripts/llm_deploy/find_ollama_gguf.py "$OLLAMA_MODEL")"
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

probe_host="$LLAMA_CPP_HOST"
if [ "$probe_host" = 0.0.0.0 ]; then probe_host=127.0.0.1; fi
health_url="http://$probe_host:$LLAMA_CPP_PORT/health"
if curl --noproxy '*' -sS --max-time 2 "$health_url" >/dev/null 2>&1; then
  echo "[ERROR] Port $LLAMA_CPP_PORT already has an HTTP listener. Choose a free LLAMA_CPP_PORT." >&2
  exit 1
fi

echo "[INFO] Serving $MODEL_PATH on Ascend device(s): $ASCEND_VISIBLE_DEVICES"
nohup "$LLAMA_SERVER_BIN" \
  --host "$LLAMA_CPP_HOST" \
  --port "$LLAMA_CPP_PORT" \
  --model "$MODEL_PATH" \
  --alias "$MODEL_ALIAS" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  "${extra_args[@]}" \
  > "$LOG_DIR/llm_deploy.log" 2>&1 &

server_pid=$!
echo "$server_pid" > "$LOG_DIR/llama-server.pid"
echo "[INFO] llama-server PID: $server_pid"

for _ in $(seq 1 "$READY_TIMEOUT"); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "[ERROR] llama-server exited. Check $LOG_DIR/llm_deploy.log" >&2
    exit 1
  fi
  if curl --noproxy '*' -fsS --max-time 2 "$health_url" >/dev/null 2>&1; then
    echo "[OK] llama.cpp server is ready at $health_url"
    exit 0
  fi
  sleep 1
done

echo "[ERROR] llama.cpp server did not become ready. Check $LOG_DIR/llm_deploy.log" >&2
exit 1
