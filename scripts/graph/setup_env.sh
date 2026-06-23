#!/usr/bin/env bash
# Create a small environment for graph-model training on Ascend NPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../common/ascend_env.sh
source scripts/common/ascend_env.sh

VENV_DIR="${VENV_DIR:-.venv-graph}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.5.1.post1}"
ASCEND_PYPI_INDEX="${ASCEND_PYPI_INDEX:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"

if ! python3 -m venv "$VENV_DIR"; then
  rm -rf "$VENV_DIR"
  python3 -m pip install --user virtualenv
  python3 -m virtualenv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install "numpy<2"
python -m pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "torch-npu==$TORCH_NPU_VERSION" --extra-index-url "$ASCEND_PYPI_INDEX"
