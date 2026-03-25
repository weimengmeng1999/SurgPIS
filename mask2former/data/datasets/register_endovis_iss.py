import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# ENDOVIS_PART_SEG_FULL_CATEGORIES_18 = [
#         {"id": 1, "name": "Bipolar Forceps", "part": [1, 2, 3]},
#         {"id": 2, "name": "Prograsp Forceps", "part": [1, 2, 3]},
#         {"id": 3, "name": "Large Needle Driver", "part": [1, 2, 3]},
#         {"id": 4, "name": "Monopolar Curved Scissors", "part": [1, 2, 3]},
#         {"id": 5, "name": "Ultrasound Probe", "part": [1, 2, 3]},
#         {"id": 6, "name": "Suction Instrument", "part": [1, 2, 3]},
#         {"id": 7, "name": "Clip Applier", "part": [1, 2, 3]},
#         {"id": 8, "name": "Vessel Sealer", "part": [1, 2, 3]},
#         {"id": 9, "name": "Grasping Retractor", "part": [1, 2, 3]},
#     ] 

# ENDOVIS_PART_SEG_FULL_CATEGORIES_17 = [
#         {"id": 1, "name": "Bipolar Forceps", "part": [1, 2, 3]},
#         {"id": 2, "name": "Prograsp Forceps", "part": [1, 2, 3]},
#         {"id": 3, "name": "Large Needle Driver", "part": [1, 2, 3]},
#         {"id": 4, "name": "Vessel Sealer", "part": [1, 2, 3]},
#         {"id": 5, "name": "Grasping Retractor", "part": [1, 2, 3]},
#         {"id": 6, "name": "Monopolar Curved Scissors", "part": [1, 2, 3]},
#         {"id": 7, "name": "Ultrasound Probe", "part": [1, 2, 3]},
# ]

ENDOVIS_IST_SEG_FULL_CATEGORIES = [{"name": "Tissue Background", "id": 0, "trainId": 0},
                                    {"name": "Bipolar Forceps", "id": 1, "trainId": 1},
                                    {"name": "Prograsp Forceps", "id": 2, "trainId": 2},
                                    {"name": "Large Needle Driver", "id": 3, "trainId": 3},
                                    {"name": "Vessel Sealer", "id": 4, "trainId": 8},
                                    {"name": "Grasping Retractor", "id": 5, "trainId": 9},
                                    {"name": "Monopolar Curved Scissors", "id": 6, "trainId": 4},
                                    {"name": "Ultrasound Probe", "id": 7, "trainId": 5},
                                    ]
# ENDOVIS_PART_SEG_FULL_CATEGORIES = [
#                                     {"name": "Instrument Shaft", "id": 1, "trainId": 0},
#                                     {"name": "Instrument Wrist", "id": 2, "trainId": 1},
#                                     {"name": "Instrument Claspers", "id": 3, "trainId": 2},
#                                     ]


def _get_endovis_full_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in ENDOVIS_IST_SEG_FULL_CATEGORIES]

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k["id"]: k["trainId"] for k in ENDOVIS_IST_SEG_FULL_CATEGORIES}
    stuff_classes = [k["name"] for k in ENDOVIS_IST_SEG_FULL_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret



def register_all_endovis_iss_full(root):
    # /nfs/home/mwei/EndoVis2018
    # root_dir = os.path.join(root, "mskfm")
    root_dir = root
    meta = _get_endovis_full_meta()
    # for name, dirname in [("train", "train"), ("test", "test")]:
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root_dir, dirname, 'images')
        # gt_dir = os.path.join(root_dir, dirname, 'parts_masks')
        gt_dir = os.path.join(root_dir, dirname, 'annotations')
        # gt_dir = "/nfs/home/mwei/EndoVis2018/mskfm/train/parts_masks_igl0"
        name = f"endovis17_full_iss_seg_{name}"
        DatasetCatalog.register(
        # name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
    )

        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

_root = "/nfs/home/mwei/EndoVis2017-refine"
register_all_endovis_iss_full(_root)
