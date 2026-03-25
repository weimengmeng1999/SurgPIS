import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F
import pycocotools.mask as mask_util

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask
from detectron2.data import MetadataCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform

# from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
# from .mask_former_instance_dataset_mapper import MaskFormerInstanceDatasetMapper

__all__ = ["MaskFormerInspartDatasetMapper"]

def map_value(x):
    if x % 3 == 1:
        return 0  
    elif x % 3 == 2:
        return 1  
    else:
        return 2 

class MaskFormerInspartDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        size_divisibility,
        ignore_label,
        meta,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.meta = meta

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "meta": meta
        }
        return ret


    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # print(dataset_dict["file_name"])
        
        utils.check_image_size(dataset_dict, image)
        # Semantic segmentation mask
        if "sem_seg_file" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        # h, w
        # image_shape = (128, 128)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        # Instance segmentation
        assert "instances" in dataset_dict
        for anno in dataset_dict["instances"]:
            anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("instances")
            if obj.get("iscrowd", 0) == 0
        ]


        masks, classes = [], []
        for obj in annos:
            classes.append(int(obj["category_id"]))
            segm = obj["segmentation"]
            if isinstance(segm, list):
                masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
            elif isinstance(segm, dict):
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                masks.append(segm)
            else:
                raise ValueError(f"Unsupported segmentation type: {type(segm)}")

                # Prepare tensors
        classes_np = classes
        classes = torch.tensor(classes, dtype=torch.int64)
        masks_ins = [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [0, self.size_divisibility - image_size[1], 0, self.size_divisibility - image_size[0]]
            image = F.pad(image, padding_size, value=128).contiguous()
            dataset_dict["image"] = image
            image_shape = (image.shape[-2], image.shape[-1])
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
                dataset_dict["sem_seg"] = sem_seg_gt.long()
                sem_seg_gt = sem_seg_gt.cpu().numpy()
                instances_sem_part = Instances(image_shape)
                classes_sem_part = np.unique(sem_seg_gt)
                # remove ignored region
                # if self.ignore_label != -1:
                classes_sem_part = classes_sem_part[classes_sem_part != self.ignore_label]
                # instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
                instances_sem_part.gt_classes = torch.tensor([1, 2, 3], dtype=torch.int64)

                sem_part_masks = []
                # for class_id in classes:
                #     masks.append(sem_seg_gt == class_id)
                for class_id in range(3):
                    if class_id in classes_sem_part:
                        sem_part_masks.append(sem_seg_gt == class_id)
                    else:
                    # masks.append(np.zeros((sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])))
                        sem_part_masks.append(np.zeros(image_shape))

                if len(sem_part_masks) == 0:
                # Some image does not have annotation (all ignored)
                # instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
                    instances_sem_part.gt_masks = torch.zeros((0, image_shape[0], image_shape[1]))
                else:
                    sem_part_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in sem_part_masks])
                    )
                    instances_sem_part.gt_masks = sem_part_masks.tensor
                dataset_dict["instances_sem_part"] = instances_sem_part
            masks_ins = [F.pad(x, padding_size, value=0).contiguous() for x in masks_ins]
            masks = [x.cpu().numpy().astype(np.uint8) for x in masks_ins]

        instances = Instances(image_shape)
        instances.gt_classes = classes

        masks_parts = []
        num_parts = []
        part_ids_list = []
        for idx, mask in enumerate(masks):
            mask_parts_per_img = []
            part_ids_list_per_img = []
            class_id = classes_np[idx]
            part_ids = self.meta.train_sids2train_pids[class_id-1]
            for part_id in part_ids:
                part_mask = sem_seg_gt == map_value(part_id)
                inst_part_mask = np.logical_and(part_mask, mask)
                mask_parts_per_img.append(inst_part_mask)
                part_ids_list_per_img.append(part_id - 1)
            if len(mask_parts_per_img) > 0:
                mask_parts_per_img = BitMasks(
                    torch.stack(
                        [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in mask_parts_per_img])
                ).tensor
            else:
                mask_parts_per_img = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            
            num_parts.append(len(part_ids_list_per_img))
            part_ids_list.append(torch.from_numpy(np.ascontiguousarray(part_ids_list_per_img, dtype=np.longlong)))
            masks_parts.append(mask_parts_per_img)

        instances.gt_masks = BitMasks(torch.stack(masks_ins)).tensor if masks_ins else torch.zeros((0, *image_shape))
        instances.gt_masks_parts = masks_parts
        instances.gt_num_parts = torch.from_numpy(np.ascontiguousarray(num_parts))
        instances.gt_part_ids = part_ids_list
        dataset_dict["instances"] = instances

        return dataset_dict