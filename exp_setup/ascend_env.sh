#!/usr/bin/env bash
# Source this before running PyTorch or llama.cpp on an Ascend device.

load_ascend_env() {
  local env_file="${CANN_ENV:-}" directory flags="$-"
  local selected="${NPU_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-0}}}" id
  local -a ids
  local -A seen=()
  if [[ ! "$selected" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Set NPU_DEVICES to comma-separated physical NPU IDs, such as 0,1." >&2
    return 1
  fi
  IFS=, read -r -a ids <<< "$selected"
  for id in "${ids[@]}"; do
    if [[ -n "${seen[$id]:-}" ]]; then
      echo "Physical NPU $id is listed more than once." >&2
      return 1
    fi
    seen[$id]=1
  done
  if [[ -z "$env_file" ]]; then
    for directory in ascend-toolkit cann cann-8.5.0; do
      env_file="/usr/local/Ascend/$directory/set_env.sh"
      if [[ -r "$env_file" ]]; then break; fi
    done
  fi
  if [[ ! -r "$env_file" ]]; then
    echo "Set CANN_ENV to the path of your CANN set_env.sh." >&2
    return 1
  fi

  # Vendor environment scripts can reference unset variables.
  set +u
  source "$env_file"
  if [[ "$flags" == *u* ]]; then set -u; fi

  export ASCEND_VISIBLE_DEVICES="$selected" ASCEND_RT_VISIBLE_DEVICES="$selected"
  NPU_COUNT=${#ids[@]}

  for directory in \
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64 \
    /usr/local/Ascend/cann/aarch64-linux/lib64 \
    /usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64; do
    if [[ -d "$directory" && ":${LD_LIBRARY_PATH:-}:" != *":$directory:"* ]]; then
      export LD_LIBRARY_PATH="$directory:${LD_LIBRARY_PATH:-}"
    fi
  done
}

run_npu_python() {
  local python="${PYTHON_BIN:-.venv/bin/python}"
  if (( NPU_COUNT > 1 )); then
    export PYTHONUNBUFFERED=1 HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
    exec "$python" -m torch.distributed.run --standalone --nproc-per-node "$NPU_COUNT" --module "$@"
  fi
  exec "$python" -u -m "$@"
}

load_ascend_env
