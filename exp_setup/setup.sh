#!/usr/bin/env bash
# Prepare the shared runtime for the examples.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source exp_setup/ascend_env.sh

case "${1:-}" in
  deploy)
    if [[ -z "${LLAMA_SERVER_BIN:-}" && ! -x llama.cpp/build/bin/llama-server ]]; then
      repo="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
      ref="${LLAMA_CPP_REF:-ac4105d68b2955027115cf9bb50941ccf56974eb}"
      if [[ ! -e llama.cpp/.git ]]; then
        git init -q llama.cpp
        git -C llama.cpp remote add origin "$repo"
      fi
      if ! git -C llama.cpp cat-file -e "$ref^{commit}" 2>/dev/null; then
        git -C llama.cpp fetch --depth=1 origin "$ref"
        ref=FETCH_HEAD
      fi
      git -C llama.cpp checkout --detach "$ref"
      cmake -S llama.cpp -B llama.cpp/build -DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
      cmake --build llama.cpp/build --parallel "${BUILD_JOBS:-8}" --target llama-server llama-quantize
    fi
    "${LLAMA_SERVER_BIN:-llama.cpp/build/bin/llama-server}" --list-devices
    ;;
  python)
    if [[ -z "${PYTHON_BIN:-}" ]]; then
      if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
      .venv/bin/python -m pip install 'numpy<2'
      .venv/bin/python -m pip install "torch==${TORCH_VERSION:-2.5.1}" \
        --index-url https://download.pytorch.org/whl/cpu
      .venv/bin/python -m pip install "torch-npu==${TORCH_NPU_VERSION:-2.5.1.post1}" \
        --extra-index-url "${ASCEND_PYPI_INDEX:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
      .venv/bin/python -m pip install -r exp_setup/requirements.txt
    fi
    "${PYTHON_BIN:-.venv/bin/python}" - <<'PY'
import peft
import torch
import torch_npu
import transformers

if not torch.npu.is_available():
    raise RuntimeError("No Ascend NPU is visible to PyTorch.")
print(f"PyTorch {torch.__version__}, torch-npu {torch_npu.__version__}, "
      f"Transformers {transformers.__version__}, PEFT {peft.__version__}")
print(f"Visible NPUs: {torch.npu.device_count()}")
PY
    ;;
  *)
    echo "Usage: bash exp_setup/setup.sh {deploy|python}" >&2
    exit 2
    ;;
esac
