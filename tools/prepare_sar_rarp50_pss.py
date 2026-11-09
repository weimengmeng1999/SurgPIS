#!/usr/bin/env python3
"""
One-time conversion: remap SAR_RARP50_seg's raw 10-class annotations into the
3-part label space used elsewhere in this repo (PART_NAMES = [Shaft, Wrist,
Clasper]), for the weakly-supervised PSS (part semantic segmentation) branch.

Raw SAR_RARP50_seg annotation values (see SAR_RARP50_seg/README.md):
    0 Background, 1 Tool clasper, 2 Tool wrist, 3 Tool shaft,
    4 Suturing needle, 5 Thread, 6 Suction tool, 7 Needle Holder,
    8 Clamps, 9 Catheter

Target values (background=0, parts offset by 1 to match PART_NAMES order,
everything else -> ignore_label so it contributes no loss):
    0 Background, 1 Shaft, 2 Wrist, 3 Clasper, 255 ignore

MaskFormerSemanticDatasetMapper reads sem_seg_file_name directly off disk with
no remap hook, so this writes a new annotations_pss/ directory alongside the
existing images/ and annotations/ rather than remapping at load time.

Usage:
    python3 tools/prepare_sar_rarp50_pss.py --root /path/to/SAR_RARP50_seg
"""
import argparse
import os

import numpy as np
from PIL import Image

IGNORE_LABEL = 255
REMAP = {0: 0, 1: 3, 2: 2, 3: 1, 4: IGNORE_LABEL, 5: IGNORE_LABEL, 6: IGNORE_LABEL,
         7: IGNORE_LABEL, 8: IGNORE_LABEL, 9: IGNORE_LABEL}

_LUT = np.full(256, IGNORE_LABEL, dtype=np.uint8)
for src, dst in REMAP.items():
    _LUT[src] = dst


def remap_split(root, split):
    src_dir = os.path.join(root, split, "annotations")
    dst_dir = os.path.join(root, split, "annotations_pss")
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(os.listdir(src_dir))
    for i, fname in enumerate(files):
        arr = np.array(Image.open(os.path.join(src_dir, fname)))
        if arr.ndim == 3:
            arr = arr[..., 0]
        remapped = _LUT[arr]
        Image.fromarray(remapped, mode="L").save(os.path.join(dst_dir, fname))
        if (i + 1) % 2000 == 0:
            print(f"  [{split}] {i + 1}/{len(files)}", flush=True)
    print(f"{split}: wrote {len(files)} remapped masks to {dst_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="path to SAR_RARP50_seg")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    for split in args.splits:
        remap_split(args.root, split)


if __name__ == "__main__":
    main()
