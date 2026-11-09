#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
args=()
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
exec "${PYTHON:-python3}" train_net.py "${args[@]}" \
  --config-file "${CONFIG:-configs/endovis/instance-part-segmentation/stage1_R50.yaml}" \
  --num-gpus "${NUM_GPUS:-2}" \
  OUTPUT_DIR "${OUTPUT_DIR:-output/stage1}" "$@"
