#!/usr/bin/env bash
# Create the LoRA fine-tuning Python environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../common/ascend_env.sh
source scripts/common/ascend_env.sh

VENV_DIR="${VENV_DIR:-.venv-finetune}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.5.1.post1}"
ASCEND_PYPI_INDEX="${ASCEND_PYPI_INDEX:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

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
python -m pip install -r requirements-finetune.txt

python - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count() if torch.npu.is_available() else 0)
PY
