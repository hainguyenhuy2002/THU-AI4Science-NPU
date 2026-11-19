#!/usr/bin/env bash
# Source this before running PyTorch or llama.cpp on an Ascend device.

load_ascend_env() {
  local env_file="${CANN_ENV:-}" directory flags="$-"
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

  export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0}"
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_VISIBLE_DEVICES}"

  for directory in \
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64 \
    /usr/local/Ascend/cann/aarch64-linux/lib64 \
    /usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64; do
    if [[ -d "$directory" && ":${LD_LIBRARY_PATH:-}:" != *":$directory:"* ]]; then
      export LD_LIBRARY_PATH="$directory:${LD_LIBRARY_PATH:-}"
    fi
  done
}

load_ascend_env
