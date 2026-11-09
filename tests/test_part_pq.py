#!/usr/bin/env python3
"""
Hand-built dummy examples for mask2former.evaluation.part_pq, with expected PartPQ values
computed by hand -- run before wiring PartPQ into anything real, per the plan's gate.

Usage: python3 tests/test_part_pq.py
"""
import importlib.util
import os

import numpy as np

# part_pq.py has zero detectron2 dependency, but importing it as `mask2former.evaluation.part_pq`
# would trigger mask2former/__init__.py's package-level dataset registrations, which DO require
# detectron2 (not installed in this local dev environment -- only on the training machines).
# Load it directly by file path instead, so this test runs anywhere with just numpy.
_spec = importlib.util.spec_from_file_location(
    "part_pq",
    os.path.join(os.path.dirname(__file__), "..", "mask2former", "evaluation", "part_pq.py"),
)
_part_pq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_part_pq)
IGNORE_LABEL, part_pq_stats = _part_pq.IGNORE_LABEL, _part_pq.part_pq_stats


def test_perfect_match():
    """1 GT instance, 1 identical pred instance, 3 part classes all present and correct
    -> PartPQ/PartSQ/PartRQ == 1.0 exactly."""
    mask = np.ones((4, 4), dtype=bool)
    part_map = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 0, 0],
    ])
    gt = [{"category_id": 1, "mask": mask, "part_map": part_map}]
    pred = [{"category_id": 1, "mask": mask, "part_map": part_map.copy()}]

    stat = part_pq_stats(gt, pred, num_part_classes=3)
    part_avg, _, _ = stat.average()
    assert np.isclose(part_avg["PartPQ"], 1.0), part_avg
    assert np.isclose(part_avg["PartSQ"], 1.0), part_avg
    assert np.isclose(part_avg["PartRQ"], 1.0), part_avg
    print("test_perfect_match: OK", part_avg)


def test_false_negative_only():
    """1 GT instance, 0 predictions -> FN only, PartPQ == 0.0."""
    mask = np.ones((4, 4), dtype=bool)
    part_map = np.zeros((4, 4), dtype=int)
    gt = [{"category_id": 1, "mask": mask, "part_map": part_map}]
    pred = []

    stat = part_pq_stats(gt, pred, num_part_classes=3)
    part_avg, _, _ = stat.average()
    assert np.isclose(part_avg["PartPQ"], 0.0), part_avg
    assert np.isclose(part_avg["PartRQ"], 0.0), part_avg
    assert part_avg["n"] == 1
    print("test_false_negative_only: OK", part_avg)


def test_false_positive_only():
    """0 GT instances, 1 spurious prediction -> FP only, InstancePQ == 0.0. With no GT
    instances at all for this category, has_part_gt can't be determined (defaults False),
    so it lands in the InstancePQ bucket rather than PartPQ."""
    mask = np.ones((4, 4), dtype=bool)
    part_map = np.zeros((4, 4), dtype=int)
    gt = []
    pred = [{"category_id": 1, "mask": mask, "part_map": part_map}]

    stat = part_pq_stats(gt, pred, num_part_classes=3)
    _, inst_avg, _ = stat.average()
    assert np.isclose(inst_avg["InstancePQ"], 0.0), inst_avg
    assert np.isclose(inst_avg["InstanceRQ"], 0.0), inst_avg
    assert inst_avg["n"] == 1
    print("test_false_positive_only: OK", inst_avg)


def test_partial_part_disagreement():
    """1 matched instance (mask IoU=1.0), but part labels disagree on 1 of 8 pixels.
    Hand-computed: GT col0=part0 (4px), col1=part1 (4px). Pred: (row0,col0) wrongly
    predicted as part1, everything else correct.
      part0: GT={4px col0}, Pred={3px col0 (rows1-3)} -> inter=3, union=4 -> IoU=0.75
      part1: GT={4px col1}, Pred={4px col1 + 1 wrong px from col0} -> inter=4, union=5 -> IoU=0.8
      mean part IoU = (0.75+0.8)/2 = 0.775
    Single matched instance -> PartPQ == PartSQ == 0.775 exactly, PartRQ == 1.0.
    """
    mask = np.ones((4, 2), dtype=bool)
    gt_part_map = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ])
    pred_part_map = np.array([
        [1, 1],  # (row0,col0) wrongly predicted as part1
        [0, 1],
        [0, 1],
        [0, 1],
    ])
    gt = [{"category_id": 1, "mask": mask, "part_map": gt_part_map}]
    pred = [{"category_id": 1, "mask": mask, "part_map": pred_part_map}]

    stat = part_pq_stats(gt, pred, num_part_classes=2)
    part_avg, _, _ = stat.average()
    assert np.isclose(part_avg["PartPQ"], 0.775), part_avg
    assert np.isclose(part_avg["PartSQ"], 0.775), part_avg
    assert np.isclose(part_avg["PartRQ"], 1.0), part_avg
    print("test_partial_part_disagreement: OK", part_avg)


def test_single_part_class_falls_back_to_mask_iou():
    """num_part_classes<=1 -> always use plain instance-mask IoU, part_map ignored.
    GT mask = pixels 0-7 of a 1x10 row, pred mask = pixels 2-9 (overlap 2-7 = 6 px).
    union = 0-9 = 10 px -> IoU = 6/10 = 0.6 exactly (>0.5, so it's a match).
    GT part_map is all-IGNORE_LABEL here -> has_part_gt is False -> lands in the
    InstancePQ bucket, not PartPQ. Single matched instance -> InstancePQ == InstanceSQ
    == 0.6 exactly, InstanceRQ == 1.0.
    """
    gt_mask = np.zeros((1, 10), dtype=bool)
    gt_mask[0, 0:8] = True
    pred_mask = np.zeros((1, 10), dtype=bool)
    pred_mask[0, 2:10] = True
    # part_map is nonsense/irrelevant here -- must not affect the result at all.
    junk_part_map = np.full((1, 10), IGNORE_LABEL, dtype=int)

    gt = [{"category_id": 1, "mask": gt_mask, "part_map": junk_part_map}]
    pred = [{"category_id": 1, "mask": pred_mask, "part_map": junk_part_map}]

    stat = part_pq_stats(gt, pred, num_part_classes=1)
    _, inst_avg, _ = stat.average()
    assert np.isclose(inst_avg["InstancePQ"], 0.6), inst_avg
    assert np.isclose(inst_avg["InstanceSQ"], 0.6), inst_avg
    assert np.isclose(inst_avg["InstanceRQ"], 1.0), inst_avg
    print("test_single_part_class_falls_back_to_mask_iou: OK", inst_avg)


def test_no_valid_gt_part_pixels_falls_back_to_mask_iou():
    """A matched instance whose GT part_map is entirely IGNORE_LABEL (e.g. part
    annotations missing for this instance) -> falls back to plain mask IoU rather than
    crashing or silently producing a 0/nonsense part-IoU, AND has_part_gt is False so it
    lands in InstancePQ rather than being folded into PartPQ."""
    mask = np.ones((4, 4), dtype=bool)
    gt_part_map = np.full((4, 4), IGNORE_LABEL, dtype=int)
    pred_part_map = np.zeros((4, 4), dtype=int)  # doesn't matter, GT side is all-ignore

    gt = [{"category_id": 1, "mask": mask, "part_map": gt_part_map}]
    pred = [{"category_id": 1, "mask": mask, "part_map": pred_part_map}]

    stat = part_pq_stats(gt, pred, num_part_classes=3)
    part_avg, inst_avg, _ = stat.average()
    # mask IoU is 1.0 here (identical masks), so the fallback value is exactly 1.0,
    # but it must show up under InstancePQ (n=1), with PartPQ empty (n=0).
    assert part_avg["n"] == 0, part_avg
    assert inst_avg["n"] == 1, inst_avg
    assert np.isclose(inst_avg["InstancePQ"], 1.0), inst_avg
    print("test_no_valid_gt_part_pixels_falls_back_to_mask_iou: OK", inst_avg)


def test_has_part_gt_splits_categories_into_separate_buckets():
    """Two categories in the same image: one with real part GT (-> PartPQ), one with
    none at all (-> InstancePQ). They must NOT be blended into the same average -- this
    is the whole point of the has_part_gt split (e.g. Ultrasound Probe has no part labels
    in the real dataset and shouldn't dilute a metric called "Part"PQ)."""
    mask = np.ones((2, 2), dtype=bool)
    real_part_map = np.zeros((2, 2), dtype=int)  # category 1: real part labels (imperfect pred below)
    no_part_map = np.full((2, 2), IGNORE_LABEL, dtype=int)  # category 2: no part labels at all

    gt = [
        {"category_id": 1, "mask": mask, "part_map": real_part_map},
        {"category_id": 2, "mask": mask, "part_map": no_part_map},
    ]
    pred = [
        {"category_id": 1, "mask": mask, "part_map": real_part_map.copy()},
        {"category_id": 2, "mask": mask, "part_map": no_part_map.copy()},
    ]

    stat = part_pq_stats(gt, pred, num_part_classes=3)
    part_avg, inst_avg, per_cat = stat.average()
    assert part_avg["n"] == 1, part_avg
    assert inst_avg["n"] == 1, inst_avg
    assert np.isclose(part_avg["PartPQ"], 1.0), part_avg
    assert np.isclose(inst_avg["InstancePQ"], 1.0), inst_avg
    assert per_cat[1]["has_part_gt"] is True, per_cat
    assert per_cat[2]["has_part_gt"] is False, per_cat
    print("test_has_part_gt_splits_categories_into_separate_buckets: OK", part_avg, inst_avg)


def test_multi_category_macro_average():
    """Two categories: one a perfect match (PartPQ=1.0), one pure FN (PartPQ=0.0).
    Macro-average over categories -> PartPQ == 0.5 exactly (not weighted by pixel/instance
    count)."""
    mask = np.ones((2, 2), dtype=bool)
    part_map = np.zeros((2, 2), dtype=int)

    gt = [
        {"category_id": 1, "mask": mask, "part_map": part_map},
        {"category_id": 2, "mask": mask, "part_map": part_map},
    ]
    pred = [
        {"category_id": 1, "mask": mask, "part_map": part_map.copy()},
        # category 2 has no prediction at all -> FN
    ]

    stat = part_pq_stats(gt, pred, num_part_classes=1)
    part_avg, _, per_cat = stat.average()
    assert np.isclose(part_avg["PartPQ"], 0.5), part_avg
    assert part_avg["n"] == 2
    assert np.isclose(per_cat[1]["PartPQ"], 1.0), per_cat
    assert np.isclose(per_cat[2]["PartPQ"], 0.0), per_cat
    print("test_multi_category_macro_average: OK", part_avg)


def test_accumulation_across_images():
    """PQStat.__iadd__ correctly sums tp/fp/fn/iou across multiple images (i.e. multiple
    calls to part_pq_stats) before a single final .average() call."""
    mask = np.ones((2, 2), dtype=bool)
    part_map = np.zeros((2, 2), dtype=int)

    # image 1: perfect match
    stat1 = part_pq_stats(
        [{"category_id": 1, "mask": mask, "part_map": part_map}],
        [{"category_id": 1, "mask": mask, "part_map": part_map.copy()}],
        num_part_classes=1,
    )
    # image 2: pure FN
    stat2 = part_pq_stats(
        [{"category_id": 1, "mask": mask, "part_map": part_map}],
        [],
        num_part_classes=1,
    )

    total = stat1
    total += stat2
    part_avg, _, _ = total.average()
    # category 1 now has tp=1, fn=1, fp=0, iou=1.0 -> PQ = 1.0/(1+0+0.5) = 0.6667
    assert np.isclose(part_avg["PartPQ"], 1.0 / 1.5), part_avg
    print("test_accumulation_across_images: OK", part_avg)


if __name__ == "__main__":
    test_perfect_match()
    test_false_negative_only()
    test_false_positive_only()
    test_partial_part_disagreement()
    test_single_part_class_falls_back_to_mask_iou()
    test_no_valid_gt_part_pixels_falls_back_to_mask_iou()
    test_has_part_gt_splits_categories_into_separate_buckets()
    test_multi_category_macro_average()
    test_accumulation_across_images()
    print("\nAll PartPQ dummy-example tests passed.")
