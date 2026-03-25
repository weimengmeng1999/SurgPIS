import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets import load_sem_seg

SEMANTIC_PART_CATEGORIES = [
                                    {"name": "Instrument Shaft", "id": 1},
                                    {"name": "Instrument Claspers", "id": 2},
                                    {"name": "instrument wrist", "id": 3},
                                    ]

STUFF_CATEGORIES = [{"id": 1, "name": "Tissue Background"}]

# Define instance segmentation categories
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

INSTANCE_PART_CATEGORIES = [
        {"id": 1, "name": "Bipolar Forceps", "part": [1, 2, 3]},
        {"id": 2, "name": "Prograsp Forceps", "part": [1, 2, 3]},
        {"id": 3, "name": "Large Needle Driver", "part": [1, 2, 3]},
        {"id": 4, "name": "Monopolar Curved Scissors", "part": [1, 2, 3]},
        {"id": 5, "name": "Ultrasound Probe", "part": [1, 2, 3]},
        {"id": 6, "name": "Suction Instrument", "part": [1, 2, 3]},
        {"id": 7, "name": "Clip Applier", "part": [1, 2, 3]},
        {"id": 8, "name": "Vessel Sealer", "part": [1, 2, 3]},
        {"id": 9, "name": "Grasping Retractor", "part": [1, 2, 3]},
    ]


def _get_combined_meta():
    """Combines metadata for semantic segmentation, instance segmentation, and instance-part segmentation."""

    # Instance segmentation metadata
    thing_ids = [k["id"] for k in ROBO_CATEGORIES]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ROBO_CATEGORIES]

    # Semantic-part segmentation metadata
    part_thing_ids = [k["id"] for k in SEMANTIC_PART_CATEGORIES]
    part_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(part_thing_ids)}
    part_classes = [k["name"] for k in SEMANTIC_PART_CATEGORIES]
    
    # Instance-part segmentation metadata
    train_sids2train_pids = {
    i: inst["part"] for i, inst in enumerate(INSTANCE_PART_CATEGORIES)
}   
    
    train_sids2inf_pids = {
    i: inst["part"] for i, inst in enumerate(INSTANCE_PART_CATEGORIES_INF)
}   
    
    #binady segmentation metadata
    stuff_ids = [k["id"] for k in STUFF_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in STUFF_CATEGORIES]

    
    return {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "part_classes": part_classes,
        "part_dataset_id_to_contiguous_id": part_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "train_sids2train_pids": train_sids2train_pids,
        "train_sids2inf_pids": train_sids2inf_pids,
    }


def load_combined_dataset(image_dir, sem_seg_dir, instance_json):
    """Loads dataset while keeping semantic, instance, and instance-part segmentation separate."""
    # Load semantic segmentation masks
    sem_seg_data = load_sem_seg(sem_seg_dir, image_dir, gt_ext="png", image_ext="png")

    # Load instance segmentation
    instance_data = load_coco_json(instance_json, image_dir)

    # Create lookup dictionaries
    instance_dict = {d["file_name"]: d["annotations"] for d in instance_data}

    dataset_dicts = []
    for sem in sem_seg_data:
        filename = sem["file_name"]

        dataset_dicts.append({
            "file_name": filename,
            "image_id": os.path.splitext(os.path.basename(filename))[0],
            "sem_seg_file": sem["sem_seg_file_name"],  # Semantic segmentation stored separately
            "instances": instance_dict.get(filename, []),  # Instance segmentation
        })

    return dataset_dicts


def register_inp_dataset(root):
    """Registers dataset combining semantic, instance, and instance-part segmentation separately."""
    for name, dirname in [("train", "train"), ("val", "val")]:
        dataset_name = f"endovis_inp_{name}"

        image_root = os.path.join(root, dirname, "images")
        sem_seg_root = os.path.join(root, dirname, "part_annotations")
        instance_json = os.path.join(root, f"endovis_instance_{name}.json")
        panoptic_json = os.path.join(root, f"endovis_pan_{name}.json")
        
        DatasetCatalog.register(
            dataset_name,
            lambda: load_combined_dataset(image_root, sem_seg_root, instance_json),
        )

        MetadataCatalog.get(dataset_name).set(
            image_root=image_root,
            sem_seg_root=sem_seg_root,
            panoptic_root = image_root.replace("images", "annotations"),
            panoptic_json = panoptic_json,
            gt_dir = image_root.replace("images", "annotations"),
            evaluator_type="coco",
            ignore_label=255,
            **_get_combined_meta(),
        )


# Register the dataset
_root = '/nfs/home/mwei/EndoVis2018-refine'
register_inp_dataset(_root)



