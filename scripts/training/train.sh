#!/usr/bin/env bash
# Train a graph neural network on Ascend using a synthetic graph.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source exp_setup/ascend_env.sh

mkdir -p outputs
output_dir="${OUTPUT_DIR:-$(mktemp -d "$PWD/outputs/training.XXXXXX")}"
run_npu_python scripts.training.train \
  --epochs "${EPOCHS:-20}" --nodes "${NODES:-1200}" \
  --output "$output_dir/gcn_metrics.jsonl" "$@"
