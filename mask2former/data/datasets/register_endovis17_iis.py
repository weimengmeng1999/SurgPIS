# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

ROBO_CATEGORIES = [
        {"id": 1, "name": "Bipolar Forceps"},
        {"id": 2, "name": "Prograsp Forceps"},
        {"id": 3, "name": "Large Needle Driver"},
        {"id": 4, "name": "Monopolar Curved Scissors"},
        {"id": 5, "name": "Ultrasound Probe"},
        {"id": 6, "name": "Suction Instrument"},
        {"id": 7, "name": "Clip Applier"},
        {"id": 8, "name": "Vessel Sealer"},
        {"id": 9, "name": "Grasping Retractor"},
    ] 


INSTANCE_PART_CATEGORIES_INF = [
        {"id": 1, "name": "Bipolar Forceps", "part": [1, 2, 3]},
        {"id": 2, "name": "Prograsp Forceps", "part": [4, 5, 6]},
        {"id": 3, "name": "Large Needle Driver", "part": [7, 8, 9]},
        {"id": 4, "name": "Monopolar Curved Scissors", "part": [10, 11, 12]},
        {"id": 5, "name": "Ultrasound Probe", "part": [13, 14, 15]},
        {"id": 6, "name": "Suction Instrument", "part": [16, 17, 18]},
        {"id": 7, "name": "Clip Applier", "part": [19, 20, 21]},
        {"id": 8, "name": "Vessel Sealer", "part": [22, 23, 24]},
        {"id": 9, "name": "Grasping Retractor", "part": [25, 26, 27]},
    ] 

_PREDEFINED_SPLITS = {
    # point annotations without masks
    "endovis17_iis_train": (
        "train/images",
        "endovis_instance_train.json",
    ),
    "endovis17_iis_val": (
        "val/images",
        "endovis_instance_val.json",
    ),
}


def _get_robo_instances_meta():
    thing_ids = [k["id"] for k in ROBO_CATEGORIES]
    # assert len(thing_ids) == 100, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ROBO_CATEGORIES]

    train_sids2inf_pids = {
    i: inst["part"] for i, inst in enumerate(INSTANCE_PART_CATEGORIES_INF)
}   
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "train_sids2inf_pids": train_sids2inf_pids,
    }
    return ret


def register_all_robo_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_robo_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
_root = '/nfs/home/mwei/EndoVis-RS17'
register_all_robo_instance(_root)
# _root = '/nfs/home/mwei/Endovis2018-1exp/ins_seg'
# register_all_robo_instance(_root)

