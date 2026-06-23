#!/usr/bin/env bash
# Run the graph-model NPU template with the CANN environment loaded.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../scripts/common/ascend_env.sh
source scripts/common/ascend_env.sh

VENV_DIR="${VENV_DIR:-.venv-graph}"
EPOCHS="${EPOCHS:-50}"
HOLD_SECONDS="${HOLD_SECONDS:-20}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python patterns/graph_model_npu_template.py --device npu --epochs "$EPOCHS" --hold-seconds "$HOLD_SECONDS"
