#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
case "${BACKBONE:-r50}" in
  r50|rn50) model=R50; suffix="";;
  dinob) model=DINO_B; suffix=_dino_b;;
  swinb) model=Swin_B; suffix=_swin_b;;
  *) echo "Unknown BACKBONE: ${BACKBONE} (use r50, dinob, or swinb)" >&2; exit 2;;
esac
# Populate the shared torch.hub cache before distributed workers start.
if [[ "${BACKBONE:-r50}" == dinob ]]; then
  "${PYTHON:-python3}" -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
fi
args=()
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
exec "${PYTHON:-python3}" train_net.py "${args[@]}" \
  --config-file "${CONFIG:-configs/endovis/instance-part-segmentation/stage1_${model}.yaml}" \
  --num-gpus "${NUM_GPUS:-2}" \
  OUTPUT_DIR "${OUTPUT_DIR:-output/stage1${suffix}}" "$@"
