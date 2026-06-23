#!/usr/bin/env bash
# Run the transferable CUDA-to-NPU PyTorch pattern with the CANN environment loaded.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../scripts/common/ascend_env.sh
source scripts/common/ascend_env.sh

VENV_DIR="${VENV_DIR:-.venv-graph}"
STEPS="${STEPS:-200}"
HOLD_SECONDS="${HOLD_SECONDS:-20}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python patterns/cuda_to_npu_minimal.py --device npu --steps "$STEPS" --hold-seconds "$HOLD_SECONDS"
