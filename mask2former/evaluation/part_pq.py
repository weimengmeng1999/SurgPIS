# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Meng Wei.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""NumPy PartPQ metrics adapted from TAPPS:
https://github.com/tue-mps/tapps (panoptic_parts/utils/evaluation_PartPQ.py).

Matches instances at mask IoU > 0.5 and scores matched parts by mean IoU.
Categories without part ground truth are reported separately as InstancePQ."""
from collections import defaultdict

import numpy as np

IGNORE_LABEL = 255


class PQStatCat:
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.has_part_gt = False

    def __iadd__(self, other):
        self.iou += other.iou
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.has_part_gt = self.has_part_gt or other.has_part_gt
        return self


class PQStat:
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, category_id):
        return self.pq_per_cat[category_id]

    def __iadd__(self, other):
        for category_id, cat_stat in other.pq_per_cat.items():
            self.pq_per_cat[category_id] += cat_stat
        return self

    def average(self):
        """Return (part_avg, instance_avg, per_category) PQ/SQ/RQ metrics.

        Categories are grouped by whether they have part ground truth."""
        part_pq, part_sq, part_rq, part_n = 0.0, 0.0, 0.0, 0
        inst_pq, inst_sq, inst_rq, inst_n = 0.0, 0.0, 0.0, 0
        per_category = {}
        for category_id, stat in self.pq_per_cat.items():
            if stat.tp + stat.fp + stat.fn == 0:
                continue
            pq_cat = stat.iou / (stat.tp + 0.5 * stat.fp + 0.5 * stat.fn)
            sq_cat = stat.iou / stat.tp if stat.tp != 0 else 0.0
            rq_cat = stat.tp / (stat.tp + 0.5 * stat.fp + 0.5 * stat.fn)
            per_category[category_id] = {
                "PartPQ": pq_cat, "PartSQ": sq_cat, "PartRQ": rq_cat,
                "has_part_gt": stat.has_part_gt,
            }
            if stat.has_part_gt:
                part_n += 1
                part_pq += pq_cat
                part_sq += sq_cat
                part_rq += rq_cat
            else:
                inst_n += 1
                inst_pq += pq_cat
                inst_sq += sq_cat
                inst_rq += rq_cat

        part_avg = (
            {"PartPQ": part_pq / part_n, "PartSQ": part_sq / part_n, "PartRQ": part_rq / part_n, "n": part_n}
            if part_n > 0 else {"PartPQ": 0.0, "PartSQ": 0.0, "PartRQ": 0.0, "n": 0}
        )
        inst_avg = (
            {"InstancePQ": inst_pq / inst_n, "InstanceSQ": inst_sq / inst_n, "InstanceRQ": inst_rq / inst_n, "n": inst_n}
            if inst_n > 0 else {"InstancePQ": 0.0, "InstanceSQ": 0.0, "InstanceRQ": 0.0, "n": 0}
        )
        return part_avg, inst_avg, per_category


def _mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0


def _part_mean_iou(gt_part_map, pred_part_map, gt_mask, num_part_classes):
    """Mean part IoU within labeled GT pixels; None if no parts are evaluable."""
    evaluated = np.logical_and(gt_mask, gt_part_map != IGNORE_LABEL)
    if not evaluated.any():
        return None
    gt_labels = gt_part_map[evaluated]
    pred_labels = pred_part_map[evaluated]
    ious = []
    for part_id in range(num_part_classes):
        gt_is = gt_labels == part_id
        pred_is = pred_labels == part_id
        inter = np.logical_and(gt_is, pred_is).sum()
        union = np.logical_or(gt_is, pred_is).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else None


def part_pq_stats(gt_instances, pred_instances, num_part_classes):
    """Compute per-category PQ statistics for one image.

    Instances contain category_id, a boolean (H, W) mask, and an integer
    part_map with IDs 0..num_part_classes-1 or IGNORE_LABEL.
    Uses mask IoU when parts are unavailable or num_part_classes <= 1.
    Accumulate results with +=, then call average()."""
    pq_stat = PQStat()

    categories = set(inst["category_id"] for inst in gt_instances) | \
                 set(inst["category_id"] for inst in pred_instances)

    for category_id in categories:
        gt_cat = [inst for inst in gt_instances if inst["category_id"] == category_id]
        pred_cat = [inst for inst in pred_instances if inst["category_id"] == category_id]

        # Track part-label availability independently of instance matching.
        if any(np.logical_and(inst["mask"], inst["part_map"] != IGNORE_LABEL).any() for inst in gt_cat):
            pq_stat[category_id].has_part_gt = True

        matched_pred = set()
        for gt_inst in gt_cat:
            match_found = False
            for j, pred_inst in enumerate(pred_cat):
                if j in matched_pred:
                    continue
                iou = _mask_iou(gt_inst["mask"], pred_inst["mask"])
                if iou > 0.5:
                    matched_pred.add(j)
                    match_found = True
                    if num_part_classes > 1:
                        part_iou = _part_mean_iou(
                            gt_inst["part_map"], pred_inst["part_map"],
                            gt_inst["mask"], num_part_classes,
                        )
                        quality = part_iou if part_iou is not None else iou
                    else:
                        quality = iou
                    pq_stat[category_id].tp += 1
                    pq_stat[category_id].iou += quality
                    break
            if not match_found:
                pq_stat[category_id].fn += 1

        pq_stat[category_id].fp += len(pred_cat) - len(matched_pred)

    return pq_stat
