#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
case "${BACKBONE:-r50}" in
  r50|rn50) model=R50; suffix="";;
  dinob) model=DINO_B; suffix=_dino_b;;
  swinb) model=Swin_B; suffix=_swin_b;;
  *) echo "Unknown BACKBONE: ${BACKBONE} (use r50, dinob, or swinb)" >&2; exit 2;;
esac
args=()
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
checkpoint="${STAGE1_CHECKPOINT:-output/stage1${suffix}/model_final.pth}"
[[ -s "$checkpoint" ]] || { echo "Stage-1 checkpoint not found: $checkpoint" >&2; exit 1; }
exec "${PYTHON:-python3}" train_net.py "${args[@]}" \
  --config-file "${CONFIG:-configs/endovis/instance-part-segmentation/stage2_${model}.yaml}" \
  --num-gpus "${NUM_GPUS:-2}" \
  WSL.TEACHER_CKPT "$checkpoint" \
  OUTPUT_DIR "${OUTPUT_DIR:-output/stage2${suffix}}" "$@"
