# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Meng Wei.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Detectron2 PartPQ evaluator using catalog ground truth.

Requires INSTANCE_ON and PANOPTIC_ON to produce instance and part predictions.
"""
import logging

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image

import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

from .part_pq import IGNORE_LABEL, PQStat, part_pq_stats


def _map_global_part_id_to_generic(global_part_map):
    """Map global part IDs to 0–2, treating background as IGNORE_LABEL."""
    generic = np.where(
        global_part_map == 0,
        IGNORE_LABEL,
        (global_part_map.astype(np.int64) - 1) % 3,
    )
    return generic.astype(np.int64)


class PartPQEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir=None, distributed=True, score_threshold=0.5):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._metadata = MetadataCatalog.get(dataset_name)
        self._num_part_classes = 3
        self._checked_config = False
        # Filter low-confidence predictions before PQ matching.
        self._score_threshold = score_threshold

        # Cache ground truth from the registered dataset loader.
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self._gt_by_image_id = {d["image_id"]: d for d in dataset_dicts}

        # Evaluate only categories present in this dataset’s ground truth.
        contiguous_id_map = self._metadata.thing_dataset_id_to_contiguous_id
        self._valid_category_ids = set()
        for d in dataset_dicts:
            for ann in d["instances"]:
                self._valid_category_ids.add(contiguous_id_map[ann["category_id"]])

        self._pq_stat = PQStat()

    def reset(self):
        self._pq_stat = PQStat()

    def _load_gt_instances(self, image_id):
        gt_dict = self._gt_by_image_id.get(image_id)
        if gt_dict is None:
            self._logger.warning(f"PartPQEvaluator: no ground truth found for image_id={image_id}")
            return []

        part_map = np.array(Image.open(gt_dict["sem_seg_file"]))

        # Map ground-truth categories to the prediction’s contiguous IDs.
        contiguous_id_map = self._metadata.thing_dataset_id_to_contiguous_id

        gt_instances = []
        for ann in gt_dict["instances"]:
            mask = mask_util.decode(ann["segmentation"]).astype(bool)
            gt_instances.append({
                "category_id": contiguous_id_map[ann["category_id"]],
                "mask": mask,
                "part_map": part_map,
            })
        return gt_instances

    def _build_pred_instances(self, output):
        if not self._checked_config:
            assert "instances" in output, (
                "PartPQEvaluator requires MODEL.MASK_FORMER.TEST.INSTANCE_ON=True "
                "(outputs['instances'] is missing)"
            )
            assert "parts" in output, (
                "PartPQEvaluator requires MODEL.MASK_FORMER.TEST.PANOPTIC_ON=True "
                "(outputs['parts'] is missing)"
            )
            self._checked_config = True

        instances = output["instances"]
        keep = instances.scores.cpu().numpy() > self._score_threshold
        pred_masks = instances.pred_masks.cpu().numpy().astype(bool)[keep]
        pred_classes = instances.pred_classes.cpu().numpy()[keep]
        generic_part_map = _map_global_part_id_to_generic(output["parts"].cpu().numpy())

        pred_instances = []
        for i in range(len(pred_classes)):
            category_id = int(pred_classes[i])
            if category_id not in self._valid_category_ids:
                # Skip categories absent from the dataset annotations.
                continue
            pred_instances.append({
                "category_id": category_id,
                "mask": pred_masks[i],
                "part_map": generic_part_map,
            })
        return pred_instances

    def process(self, inputs, outputs):
        for input_, output in zip(inputs, outputs):
            image_id = input_.get("image_id") or input_["file_name"]
            gt_instances = self._load_gt_instances(image_id)
            pred_instances = self._build_pred_instances(output)
            self._pq_stat += part_pq_stats(gt_instances, pred_instances, self._num_part_classes)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            pq_stats = comm.gather(self._pq_stat, dst=0)
            if not comm.is_main_process():
                return {}
            total = PQStat()
            for stat in pq_stats:
                total += stat
        else:
            total = self._pq_stat

        part_avg, inst_avg, per_category = total.average()
        per_category_named = {
            self._metadata.thing_classes[cat_id] if cat_id < len(self._metadata.thing_classes) else cat_id: v
            for cat_id, v in per_category.items()
        }
        self._logger.info(f"PartPQ (categories with real part GT): {part_avg}")
        self._logger.info(f"InstancePQ (categories with no part GT, plain-IoU fallback): {inst_avg}")
        self._logger.info(f"Per-category (has_part_gt marks which bucket each landed in): {per_category_named}")
        # Return scalar metrics; per-category details are logged above.
        return {"partpq": part_avg, "instance_pq": inst_avg}
