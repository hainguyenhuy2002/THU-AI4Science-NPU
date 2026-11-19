#!/usr/bin/env bash
# Train a LoRA adapter on Ascend with the instruction examples in exp_setup.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source exp_setup/ascend_env.sh

mkdir -p outputs
output_dir="${OUTPUT_DIR:-$(mktemp -d "$PWD/outputs/finetuning.XXXXXX")}"
run_npu_python scripts.finetuning.finetune \
  --model "${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
  --data "${DATA_PATH:-exp_setup/finetuning.jsonl}" --output-dir "$output_dir" \
  --max-steps "${MAX_STEPS:-2}" "$@"
