#!/usr/bin/env python3
"""
Real-data smoke test for PartPQ: load one real EndoVis-RS17 image's ground truth (instance
masks from endovis_instance_val.json + the part_annotations semantic map), treat it as if it
were ALSO the prediction (i.e. predicting the ground truth perfectly), and confirm PartPQ==1.0.

This validates the COCO-RLE-decoding + part-map-loading logic used by PartPQEvaluator (the
part that doesn't require detectron2, which isn't installed in this local dev environment --
only on the training machines) against real data, not just the hand-built dummy cases in
test_part_pq.py.

Usage: python3 tests/test_part_pq_real_data.py --rs17-root /path/to/EndoVis-RS17
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "part_pq",
    os.path.join(os.path.dirname(__file__), "..", "mask2former", "evaluation", "part_pq.py"),
)
_part_pq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_part_pq)
part_pq_stats = _part_pq.part_pq_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs17-root", required=True)
    ap.add_argument("--image-id", default="seq_8_frame000")
    args = ap.parse_args()

    instance_json = json.load(open(os.path.join(args.rs17_root, "endovis_instance_val.json")))
    anns = [a for a in instance_json["annotations"] if a["image_id"] == args.image_id]
    assert len(anns) > 0, f"no annotations found for {args.image_id}"

    part_map = np.array(Image.open(
        os.path.join(args.rs17_root, "val", "part_annotations", f"{args.image_id}.png")
    ))

    gt_instances = []
    for ann in anns:
        mask = mask_util.decode(ann["segmentation"]).astype(bool)
        gt_instances.append({"category_id": ann["category_id"], "mask": mask, "part_map": part_map})

    # "Predict" the ground truth perfectly (same masks, same part map).
    pred_instances = [dict(inst) for inst in gt_instances]

    stat = part_pq_stats(gt_instances, pred_instances, num_part_classes=3)
    avg, per_category = stat.average()

    assert np.isclose(avg["PartPQ"], 1.0), avg
    assert np.isclose(avg["PartSQ"], 1.0), avg
    assert np.isclose(avg["PartRQ"], 1.0), avg
    assert avg["n"] == len(set(a["category_id"] for a in anns))
    print(f"Real-data smoke test on {args.image_id} ({len(anns)} instances): OK", avg)
    print("Per-category:", per_category)


if __name__ == "__main__":
    main()
