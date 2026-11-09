#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
args=()
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
checkpoint="${STAGE1_CHECKPOINT:-output/stage1/model_final.pth}"
[[ -s "$checkpoint" ]] || { echo "Stage-1 checkpoint not found: $checkpoint" >&2; exit 1; }
exec "${PYTHON:-python3}" train_net.py "${args[@]}" \
  --config-file "${CONFIG:-configs/endovis/instance-part-segmentation/stage2_R50.yaml}" \
  --num-gpus "${NUM_GPUS:-2}" \
  WSL.TEACHER_CKPT "$checkpoint" \
  OUTPUT_DIR "${OUTPUT_DIR:-output/stage2}" "$@"
