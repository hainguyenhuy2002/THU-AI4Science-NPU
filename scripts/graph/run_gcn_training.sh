#!/usr/bin/env bash
# Run a GCN training example on Ascend NPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../common/ascend_env.sh
source scripts/common/ascend_env.sh

VENV_DIR="${VENV_DIR:-.venv-graph}"
EPOCHS="${EPOCHS:-200}"
NODES="${NODES:-2500}"

mkdir -p logs outputs/graph

if [ ! -d "$VENV_DIR" ]; then
  bash scripts/graph/setup_env.sh
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

npu-smi info | tee logs/graph_npu_before.log
python scripts/graph/train_gcn_npu.py --epochs "$EPOCHS" --nodes "$NODES"
npu-smi info | tee logs/graph_npu_after.log
