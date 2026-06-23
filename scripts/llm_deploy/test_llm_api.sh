#!/usr/bin/env bash
# Send a small OpenAI-compatible request to the CANN llama.cpp server.

set -euo pipefail

LLAMA_CPP_PORT="${LLAMA_CPP_PORT:-8080}"
MODEL_ALIAS="${MODEL_ALIAS:-llama3.3-70b-ascend}"

curl -sS "http://localhost:$LLAMA_CPP_PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_ALIAS\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello from Ascend NPU in one short sentence.\"}],
    \"max_tokens\": 32,
    \"temperature\": 0.2
  }"
echo
