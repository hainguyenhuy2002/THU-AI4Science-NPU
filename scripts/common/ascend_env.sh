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
}

load_ascend_env
