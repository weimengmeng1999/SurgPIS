import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# SAR_RARP50_seg's raw annotations use 10 classes (background, 3 generic parts,
# plus 6 other objects: suturing needle, thread, suction tool, needle holder,
# clamps, catheter -- see SAR_RARP50_seg/README.md). Only the 3 generic parts
# match this repo's PartDecoder output space (PART_NAMES = [Shaft, Wrist,
# Clasper]), so tools/prepare_sar_rarp50_pss.py remaps the raw masks into
# annotations_pss/ ahead of time: 0=Background, 1=Shaft, 2=Wrist, 3=Clasper,
# 255=ignore (everything else). This registration reads that remapped copy.
SAR_RARP50_PSS_CATEGORIES = [
    {"name": "Background", "id": 0, "trainId": 0},
    {"name": "Shaft", "id": 1, "trainId": 1},
    {"name": "Wrist", "id": 2, "trainId": 2},
    {"name": "Clasper", "id": 3, "trainId": 3},
]


def _get_sar_rarp50_pss_meta():
    stuff_dataset_id_to_contiguous_id = {k["id"]: k["trainId"] for k in SAR_RARP50_PSS_CATEGORIES}
    stuff_classes = [k["name"] for k in SAR_RARP50_PSS_CATEGORIES]
    return {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }


def register_all_sar_rarp50_pss(root):
    meta = _get_sar_rarp50_pss_meta()
    for split in ("train", "test"):
        image_dir = os.path.join(root, split, "images")
        gt_dir = os.path.join(root, split, "annotations_pss")
        name = f"sar_rarp50_pss_{split}"

        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )

        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.environ.get("SAR_RARP50_ROOT", os.path.join(os.environ.get("DETECTRON2_DATASETS", "datasets"), "SAR-RARP50"))
register_all_sar_rarp50_pss(_root)
