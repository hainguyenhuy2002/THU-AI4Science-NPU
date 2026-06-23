#!/usr/bin/env bash
# Run a real-data LoRA fine-tuning task on Ascend NPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../common/ascend_env.sh
source scripts/common/ascend_env.sh

VENV_DIR="${VENV_DIR:-.venv-finetune}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATASET_NAME="${DATASET_NAME:-medalpaca/medical_meadow_medical_flashcards}"
DATA_PATH="${DATA_PATH:-data/medical_flashcards_lora.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_STEPS="${MAX_STEPS:-800}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen2.5-0.5b-medical-lora}"
export HF_ENDPOINT TOKENIZERS_PARALLELISM=false

mkdir -p logs data outputs

if [ ! -d "$VENV_DIR" ]; then
  bash scripts/finetune/setup_env.sh
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python scripts/finetune/prepare_dataset.py \
  --dataset "$DATASET_NAME" \
  --output "$DATA_PATH" \
  --max-samples "$MAX_SAMPLES"

npu-smi info | tee logs/finetune_npu_before.log
python scripts/finetune/train_lora_npu.py \
  --model "$BASE_MODEL" \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$MAX_STEPS"
npu-smi info | tee logs/finetune_npu_after.log
