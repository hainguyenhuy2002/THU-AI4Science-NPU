#!/usr/bin/env bash
# Resolve the Ollama model and serve it with llama.cpp on Ascend.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source exp_setup/ascend_env.sh

server="${LLAMA_SERVER_BIN:-llama.cpp/build/bin/llama-server}"
if [[ ! -x "$server" ]]; then
  echo "Run bash exp_setup/setup.sh deploy or set LLAMA_SERVER_BIN." >&2
  exit 1
fi

model_path="${MODEL_PATH:-}"
if [[ -z "$model_path" ]]; then
  ollama="${OLLAMA_BIN:-ollama}"
  export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT:-11434}"
  model="${OLLAMA_MODEL:-qwen2.5:0.5b-instruct-q8_0}"
  "$ollama" show "$model" >/dev/null 2>&1 || "$ollama" pull "$model"
  model_path=$("$ollama" show "$model" --modelfile | sed -n 's/^FROM[[:space:]]\{1,\}//p')
fi
if [[ ! -f "$model_path" ]]; then
  echo "Could not find the GGUF file. Use a local Ollama daemon or set MODEL_PATH." >&2
  exit 1
fi

read -r -a extra_args <<< "${LLAMA_SERVER_EXTRA_ARGS:---parallel 1 -lv 4}"
exec "$server" --host 127.0.0.1 --port "${LLAMA_CPP_PORT:-18080}" \
  --model "$model_path" --alias "${MODEL_ALIAS:-local-llm}" \
  --ctx-size "${CTX_SIZE:-4096}" --threads "${THREADS:-8}" \
  --n-gpu-layers "${N_GPU_LAYERS:-all}" "${extra_args[@]}"
