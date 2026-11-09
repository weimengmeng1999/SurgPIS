# Copyright: SurgPIS- Meng Wei
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
import numpy as np
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks



# NOTE: previously torch.jit.script-wrapped for a minor speedup, but the
# scripted version crashes non-deterministically under newer torch/CUDA
# ("RuntimeError: Global alloc not supported yet" from the legacy JIT fuser)
# on certain ragged-batch shapes. Same math, eager mode, no crash.
dice_loss_jit = dice_loss


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = sigmoid_ce_loss  # see note above dice_loss_jit


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, parts_decoder):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.parts_decoder = parts_decoder

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_parts(self, queries, mask_features, targets, indices):
        # Per img in batch, retrieve selected queries and pad to shape of max
        # First retrieve value of max_num_queries
        max_num_queries = np.max([tgt_idx.shape[0] for (src_idx, tgt_idx) in indices])

        selected_queries = list()
        for i, idx in enumerate(indices):
            # Pad the indices to make sure it is all the same length
            src_idx = torch.cat([idx[0], torch.full(size=[max_num_queries-idx[0].shape[0]],
                                                    fill_value=0,
                                                    dtype=idx[0].dtype,
                                                    device=idx[0].device)])
            selected_queries.append(queries[i][src_idx])

        # selected_queries shape: [Nb, max_num_queries, C]
        selected_queries = torch.stack(selected_queries)

        # TODO: find better solution than going to numpy
        num_parts_per_query = list()
        part_ids_per_query = list()
        # print('indices', indices)
        for tgt, idx in zip(targets, indices):
            num = tgt["num_masks_parts"]
            pt_ids = tgt["part_ids"]
            # print('pt_ids_ori', pt_ids)
            tgt_idx = idx[1]
            tgt_idx_cpu = tgt_idx.detach().cpu().numpy()
            # print('tgt_idx_cpu_ori', tgt_idx_cpu)
            num_parts_per_query.append(num[tgt_idx])
            part_ids_per_query.append([pt_ids[t_idx] for t_idx in tgt_idx_cpu])

        # masks_parts is shape [Num_parts, H, W]
        masks_parts = self.parts_decoder(selected_queries,
                                         num_parts_per_query,
                                         mask_features,
                                         part_ids_per_query)
        indices_np = [idx.detach().cpu().numpy() for i, (_, idx) in enumerate(indices)]
        masks_parts_gt = list()
        for i, idx in enumerate(indices_np):
            masks_parts_gt_per_img = targets[i]["masks_parts"]
            # print(targets[i]["masks_parts"])
            masks_parts_gt_per_img_selected = [masks_parts_gt_per_img[tgt_idx] for tgt_idx in idx]
            if len(masks_parts_gt_per_img_selected) != 0:
                masks_parts_gt.append(torch.cat(masks_parts_gt_per_img_selected, dim=0))

        if len(masks_parts_gt) != 0:
            masks_parts_gt = torch.cat(masks_parts_gt, dim=0)
        else:
            masks_parts_gt = torch.zeros_like(masks_parts)

        if masks_parts.numel() == 0:
            zero = masks_parts.sum() * 0.0
            return {"loss_mask_parts": zero, "loss_dice_parts": zero}

        # Apply losses as usual
        src_masks_parts = masks_parts[:, None]
        tgt_masks_parts = masks_parts_gt[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks_parts,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            point_labels = point_sample(
                tgt_masks_parts.float(),
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks_parts,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        num_masks = tgt_masks_parts.shape[0]

        if num_masks == 0:
            num_masks = 1

        loss_dice_parts = dice_loss_jit(point_logits, point_labels, num_masks)
        loss_mask_parts = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)

        losses = {
            "loss_mask_parts": loss_mask_parts,
            "loss_dice_parts": loss_dice_parts,
        }

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets_instances):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        targets = targets_instances
        # print('targets["labels"]', targets[0]["labels"])
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        mask_features = outputs["mask_features"]

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, mask_features=mask_features)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

            queries = outputs_without_aux["decoder_output"]
            losses_parts = self.loss_masks_parts(queries,
                                                    mask_features,
                                                    targets,
                                                    indices)
            losses.update(losses_parts)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, mask_features=mask_features)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    
                    queries = aux_outputs["decoder_output"]
                    l_dict.update(self.loss_masks_parts(queries, mask_features, targets, indices))
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class SetCriterionWeakSup(nn.Module):
    """Weakly-supervised (stage-2) loss: a student-teacher consistency loss
    against the teacher's pseudo-labels, combined with a loss against
    whichever partial ground truth is available for a given training image --
    IIS instance labels (`forward`) or PSS part-semantic labels (`forward_pss`).
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, parts_decoder,
                 num_part_classes=3, explicit_tissue_background=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_part_classes: number of generic part classes (e.g. 3 for shaft/wrist/clasper),
                used by the PSS part-semantic-map aggregation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.parts_decoder = parts_decoder
        self.num_part_classes = num_part_classes
        self.explicit_tissue_background = explicit_tissue_background

    def loss_labels(self, outputs, targets, indices, num_masks, pseudo = False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if pseudo:
            # Consistency loss against the teacher's pseudo-labels: an
            # unmatched query may just be an instance the (imperfect) teacher
            # missed, not genuine background. Defaulting it to no-object here
            # would train the student to copy the teacher's misses rather
            # than only its confident hits, so only supervise matched queries.
            if target_classes_o.numel() > 0:
                loss_ce = F.cross_entropy(src_logits[idx], target_classes_o, self.empty_weight)
            else:
                loss_ce = src_logits.sum() * 0.0
            losses = {"loss_ce_pd": loss_ce}
        else:
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
            )
            target_classes[idx] = target_classes_o
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks, pseudo = False):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        
        if pseudo:
            losses = {
            "loss_mask_pd": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice_pd": dice_loss_jit(point_logits, point_labels, num_masks),
        }
        else:
            losses = {
                "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
                "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
            }

        del src_masks
        del target_masks
        return losses

    def loss_masks_parts(self, queries, mask_features, targets, indices):
        # Per img in batch, retrieve selected queries and pad to shape of max
        # First retrieve value of max_num_queries
        max_num_queries = np.max([tgt_idx.shape[0] for (src_idx, tgt_idx) in indices])

        selected_queries = list()
        for i, idx in enumerate(indices):
            # Pad the indices to make sure it is all the same length
            src_idx = torch.cat([idx[0], torch.full(size=[max_num_queries-idx[0].shape[0]],
                                                    fill_value=0,
                                                    dtype=idx[0].dtype,
                                                    device=idx[0].device)])
            selected_queries.append(queries[i][src_idx])

        # selected_queries shape: [Nb, max_num_queries, C]
        selected_queries = torch.stack(selected_queries)
        # print(targets)

        # TODO: find better solution than going to numpy
        num_parts_per_query = list()
        # num_parts_per_query = [t["num_parts_per_query"] for t in targets]
        part_ids_per_query = list()
        # part_ids_per_query = [t["part_ids_per_query"] for t in targets]
        # print('indices_iis', indices)
        for tgt, idx in zip(targets, indices):
            num = tgt["num_parts_per_query"]
            pt_ids = tgt["part_ids_per_query"]
            # print('ptids', pt_ids)
            tgt_idx = idx[1]
            tgt_idx_cpu = tgt_idx.detach().cpu().numpy()

            # print('tgt_idx_cpu', tgt_idx_cpu)

            num_parts_per_query.append(num[tgt_idx])
            part_ids_per_query.append([pt_ids[t_idx] for t_idx in tgt_idx_cpu])

        # masks_parts is shape [Num_parts, H, W]
        masks_parts = self.parts_decoder(selected_queries,
                                         num_parts_per_query,
                                         mask_features,
                                         part_ids_per_query)
        indices_np = [idx.detach().cpu().numpy() for i, (_, idx) in enumerate(indices)]
        masks_parts_gt = list()
        for i, idx in enumerate(indices_np):
            masks_parts_gt_per_img = targets[i]["masks_parts"]
            masks_parts_gt_per_img_selected = [masks_parts_gt_per_img[tgt_idx] for tgt_idx in idx]
            if len(masks_parts_gt_per_img_selected) != 0:
                masks_parts_gt.append(torch.cat(masks_parts_gt_per_img_selected, dim=0))

        if len(masks_parts_gt) != 0:
            masks_parts_gt = torch.cat(masks_parts_gt, dim=0)
        else:
            masks_parts_gt = torch.zeros_like(masks_parts)

        if masks_parts.numel() == 0:
            zero = masks_parts.sum() * 0.0
            return {"loss_mask_parts_pd": zero, "loss_dice_parts_pd": zero}

        # Apply losses as usual
        src_masks_parts = masks_parts[:, None]
        tgt_masks_parts = masks_parts_gt[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks_parts,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            point_labels = point_sample(
                tgt_masks_parts.float(),
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks_parts,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        num_masks = tgt_masks_parts.shape[0]

        if num_masks == 0:
            num_masks = 1

        loss_dice_parts = dice_loss_jit(point_logits, point_labels, num_masks)
        loss_mask_parts = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)

        losses = {
            "loss_mask_parts_pd": loss_mask_parts,
            "loss_dice_parts_pd": loss_dice_parts,
        }

        return losses

    def teacher_consistency_losses(self, layer_outputs, mask_features, targets_pseudo, suffix=""):
        """Student-teacher consistency loss: the same instrument class/mask/dice
        + part mask/dice terms as the fully supervised loss (SetCriterion), but
        matched against the teacher's pseudo-labels instead of real ground
        truth. Shared by both the IIS (`forward`) and PSS (`forward_pss`)
        branches, since this term doesn't depend on which partial ground truth
        accompanies the image.
        """
        indices_pd = self.matcher(layer_outputs, targets_pseudo)

        num_masks_pd = sum(len(t["labels"]) for t in targets_pseudo)
        num_masks_pd = torch.as_tensor(
            [num_masks_pd], dtype=torch.float, device=layer_outputs["pred_logits"].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks_pd)
        num_masks_pd = torch.clamp(num_masks_pd / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, layer_outputs, targets_pseudo, indices_pd, num_masks_pd, pseudo=True))

        queries = layer_outputs["decoder_output"]
        losses.update(self.loss_masks_parts(queries, mask_features, targets_pseudo, indices_pd))

        if suffix:
            losses = {k + suffix: v for k, v in losses.items()}
        return losses

    def aggregate_part_semantic_map(self, pred_logits, pred_masks, part_masks_dense):
        """Builds a soft part-level semantic map from every query's
        instrument-class probability and part-mask predictions.

        pred_logits: [Nb, Q, num_classes+1], last channel = no-object.
        pred_masks: [Nb, Q, H, W], instrument-level mask logits.
        part_masks_dense: [Nb, Q, num_part_classes, H, W], from
            PartDecoder.compute_part_masks_dense -- already generic
            (channel k = k-th part regardless of instrument identity).

        Returns s_hat: [Nb, num_part_classes+1, H, W], channel 0 = background,
        channels 1..num_part_classes follow the part-decoder's own order.
        """
        assert part_masks_dense.shape[2] == self.num_part_classes, (
            f"PartDecoder produced {part_masks_dense.shape[2]} part channels but this "
            f"criterion was configured with num_part_classes={self.num_part_classes} -- "
            "they must be built from the same cfg.MODEL.MASK_FORMER.NUM_PART_CLASSES."
        )
        # rho_bg below is a product over all Q queries -- under fp16 autocast
        # this underflows to exactly 0 once roughly a dozen queries have
        # moderate-to-high overlapping confidence at the same pixel (common at
        # busy instance-boundary regions), which kills its gradient right where
        # mask precision matters most. Force fp32 for this whole computation.
        with autocast(enabled=False):
            pred_logits = pred_logits.float()
            pred_masks = pred_masks.float()
            part_masks_dense = part_masks_dense.float()

            probs = F.softmax(pred_logits, dim=-1)  # [Nb, Q, C+1]
            bg_idx = probs.shape[-1] - 1
            y = pred_masks.sigmoid()  # [Nb, Q, H, W]
            m = part_masks_dense.sigmoid()  # [Nb, Q, K, H, W]

            # most probable class per query (may itself be no-object); only
            # queries whose argmax is a real instrument contribute to the parts
            gamma = probs.argmax(dim=-1)  # [Nb, Q]
            is_fg = (gamma != bg_idx)
            if self.explicit_tissue_background:
                is_fg = is_fg & (gamma != 0)
            is_fg = is_fg.to(probs.dtype)
            c_gamma = probs.gather(-1, gamma.unsqueeze(-1)).squeeze(-1)  # [Nb, Q]
            weight = is_fg * c_gamma
            
            claim = weight.unsqueeze(-1).unsqueeze(-1) * y  # [Nb, Q, H, W], each in [0,1]
            if self.explicit_tissue_background:
                rho_bg = torch.einsum("bq,bqhw->bhw", probs[..., 0], y)
            else:
                rho_bg = torch.prod(1 - claim, dim=1)  # [Nb, H, W]

            rho_parts = torch.einsum("bq,bqkhw->bkhw", weight, m)  # [Nb, K, H, W]

            rho = torch.cat([rho_bg.unsqueeze(1), rho_parts], dim=1)  # [Nb, K+1, H, W]
            s_hat = rho / rho.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return s_hat

    def loss_pss_semantic(self, outputs, mask_features, targets_pss):
        """Compares the aggregated part-semantic map against ground-truth
        part-semantic labels, one channel per part class, averaged over the
        num_part_classes parts (background is excluded from this average --
        it's only used inside aggregate_part_semantic_map to build s_hat).
        targets_pss entries carry a "sem_seg" [H, W] long tensor with class ids
        0..num_part_classes and 255 (ignore) for pixels belonging to neither
        background nor a modelled part.
        """
        part_masks_dense = self.parts_decoder.compute_part_masks_dense(
            outputs["decoder_output"], mask_features
        )
        s_hat = self.aggregate_part_semantic_map(
            outputs["pred_logits"], outputs["pred_masks"], part_masks_dense
        )

        sem_seg_gt = torch.stack([t["sem_seg"] for t in targets_pss]).to(device=s_hat.device)
        if s_hat.shape[-2:] != sem_seg_gt.shape[-2:]:
            s_hat = F.interpolate(s_hat, size=sem_seg_gt.shape[-2:], mode="bilinear", align_corners=False)

        valid = sem_seg_gt != 255
        num_part_classes = s_hat.shape[1] - 1

        # Keep the zero connected to the graph (via s_hat) rather than a bare
        # Python float, so a batch with no valid pixels at all (e.g. an
        # all-ignore crop) still contributes a well-defined, finite loss.
        zero = s_hat.sum() * 0.0
        loss_ce = zero
        loss_dice = zero
        if valid.any():
            # s_hat is already a probability (post-sigmoid, renormalized), not
            # logits, so this needs plain binary_cross_entropy rather than the
            # logits version -- but that op is disallowed under autocast, so
            # force fp32 here explicitly.
            with autocast(enabled=False):
                for k in range(1, num_part_classes + 1):
                    pred_k = s_hat[:, k][valid].float().clamp(min=1e-6, max=1 - 1e-6)
                    gt_k = (sem_seg_gt == k).float()[valid]
                    loss_ce = loss_ce + F.binary_cross_entropy(pred_k, gt_k)
                    numerator = 2 * (pred_k * gt_k).sum()
                    denominator = pred_k.sum() + gt_k.sum()
                    loss_dice = loss_dice + (1 - (numerator + 1) / (denominator + 1))

        return {
            "loss_mask_pss": loss_ce / num_part_classes,
            "loss_dice_pss": loss_dice / num_part_classes,
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, pseudo = False):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, pseudo)

    def forward(self, outputs, targets_inp, targets_iis):
        """IIS branch: instrument class + mask/dice loss against the real weak
        IIS labels, plus the shared teacher-consistency loss (against the
        teacher's pseudo-labels, targets_inp).
        """
        targets = targets_iis
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        mask_features = outputs["mask_features"]

        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # IIS's own direct-GT losses get an "_iis" suffix so they don't share
        # key names with the supervised branch's loss_ce/loss_mask/loss_dice --
        # both branches run every iteration and get merged into one dict in
        # train_loop.py, so identical names would silently overwrite one
        # another there instead of summing (the actual backward pass isn't
        # affected either way, since it sums the tensors before that merge,
        # but logged metrics were losing the IIS branch's own values).
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, False))
        losses = {k + "_iis": v for k, v in losses.items()}
        losses.update(self.teacher_consistency_losses(outputs_without_aux, mask_features, targets_inp))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                l_dict = {}
                for loss in self.losses:
                    l_dict.update(self.get_loss(loss, aux_outputs, targets, indices, num_masks, False))
                l_dict = {k + f"_iis_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

                losses.update(self.teacher_consistency_losses(aux_outputs, mask_features, targets_inp, suffix=f"_{i}"))

        return losses

    def forward_pss(self, outputs, targets_inp, targets_pss):
        """PSS branch: part-semantic aggregation loss against the real weak
        PSS labels, plus the shared teacher-consistency loss (against the
        teacher's pseudo-labels, targets_inp).
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        mask_features = outputs["mask_features"]

        losses = {}
        losses.update(self.loss_pss_semantic(outputs_without_aux, mask_features, targets_pss))
        losses.update(self.teacher_consistency_losses(outputs_without_aux, mask_features, targets_inp))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                l_dict = self.loss_pss_semantic(aux_outputs, mask_features, targets_pss)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

                losses.update(self.teacher_consistency_losses(aux_outputs, mask_features, targets_inp, suffix=f"_{i}"))

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    
