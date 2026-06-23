#!/usr/bin/env bash
# Shared Ascend/CANN environment loader.

set -euo pipefail

load_ascend_env() {
  set +u
  if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  elif [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
  elif [ -f /usr/local/Ascend/cann-8.5.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-8.5.0/set_env.sh
  else
    echo "[WARN] Ascend set_env.sh was not found under /usr/local/Ascend." >&2
  fi
  set -u

  export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0}"
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_VISIBLE_DEVICES}"

  for lib_dir in \
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64 \
    /usr/local/Ascend/cann/aarch64-linux/lib64 \
    /usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64; do
    if [ -d "$lib_dir" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":$lib_dir:"*) ;;
        *) export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}" ;;
      esac
    fi
  done
}

load_ascend_env
