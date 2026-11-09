# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss



# NOTE: previously torch.jit.script-wrapped for a minor speedup, but the
# scripted version crashes non-deterministically under newer torch/CUDA
# ("RuntimeError: Global alloc not supported yet" from the legacy JIT fuser)
# whenever the batch has a ragged number of targets per image -- exactly the
# shape this sees during WSL consistency loss, where different images can
# have different numbers of confident teacher pseudo-instances. Same math,
# eager mode, no crash.
batch_dice_loss_jit = batch_dice_loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = batch_sigmoid_ce_loss  # see note above batch_dice_loss_jit


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0,
                 cost_mask_parts: float = 0, cost_dice_parts: float = 0, parts_decoder: nn.Module = None,
                 use_parts: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
            cost_mask_parts: relative weight of the part-level focal mask loss in the matching cost
            cost_dice_parts: relative weight of the part-level dice loss in the matching cost
            parts_decoder: module that predicts, for every query, a mask per fixed part class
                (see PartDecoder.compute_part_masks_dense). Required when use_parts is True.
            use_parts: whether to include the part-level mask cost in the cost matrix. Only
                takes effect for targets that carry "masks_parts"/"part_ids"/"num_masks_parts".
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_mask_parts = cost_mask_parts
        self.cost_dice_parts = cost_dice_parts
        self.parts_decoder = parts_decoder
        self.use_parts = use_parts

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        assert not use_parts or parts_decoder is not None, "use_parts requires a parts_decoder"

        self.num_points = num_points

    def _part_cost_for_image(self, decoder_output_b, mask_features_b, targets_b, point_coords, num_queries, device):
        """Computes the part-level mask cost between every query and every
        target instance in one image, returning a [num_queries, num_targets] matrix.
        Targets without part annotations (e.g. background) contribute zero cost.
        """
        num_targets = len(targets_b["labels"])
        cost_parts = torch.zeros(num_queries, num_targets, device=device)

        if num_targets == 0:
            return cost_parts

        # part_masks_dense: [num_queries, num_part_classes, H, W]
        part_masks_dense = self.parts_decoder.compute_part_masks_dense(
            decoder_output_b[None], mask_features_b[None]
        )[0]

        num_part_classes = part_masks_dense.shape[1]
        part_masks_dense = part_masks_dense.flatten(0, 1)[:, None]
        point_logits_parts = point_sample(
            part_masks_dense,
            point_coords.repeat(part_masks_dense.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)
        # [num_queries, num_part_classes, num_points]
        point_logits_parts = point_logits_parts.view(num_queries, num_part_classes, -1)

        # Sample all GT part masks under the ambient (enabled) autocast context,
        # same as the instrument-level cost above, so grid_sample's dtype
        # harmonization applies here too before we force fp32 below.
        gt_points_per_target = {}
        for i in range(num_targets):
            if targets_b["num_masks_parts"][i].item() == 0:
                continue
            gt_parts_i = targets_b["masks_parts"][i][:, None].to(part_masks_dense)
            gt_points_per_target[i] = point_sample(
                gt_parts_i,
                point_coords.repeat(gt_parts_i.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

        with autocast(enabled=False):
            point_logits_parts = point_logits_parts.float()

            for i in range(num_targets):
                if i not in gt_points_per_target:
                    continue
                num_parts_i = targets_b["num_masks_parts"][i].item()
                part_ids_i = targets_b["part_ids"][i]
                gt_parts_i = gt_points_per_target[i].float()

                ce_sum = 0.0
                dice_sum = 0.0
                for k in range(num_parts_i):
                    pred_k = point_logits_parts[:, part_ids_i[k], :]
                    gt_k = gt_parts_i[k:k + 1, :]
                    ce_sum = ce_sum + batch_sigmoid_ce_loss_jit(pred_k, gt_k)
                    dice_sum = dice_sum + batch_dice_loss_jit(pred_k, gt_k)

                cost_parts[:, i] = (
                    self.cost_mask_parts * (ce_sum / num_parts_i)
                    + self.cost_dice_parts * (dice_sum / num_parts_i)
                ).squeeze(1)

        return cost_parts

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, mask_features=None):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )

            if self.use_parts and mask_features is not None and "masks_parts" in targets[b]:
                C = C + self._part_cost_for_image(
                    outputs["decoder_output"][b], mask_features[b], targets[b],
                    point_coords, num_queries, out_mask.device,
                )

            C = C.reshape(num_queries, -1).cpu()

            # print('here is C ', C)

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, mask_features=None):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks
                 "decoder_output": Tensor of dim [batch_size, num_queries, hidden_dim], required when use_parts is True

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks
                 "masks_parts"/"part_ids"/"num_masks_parts": optional, per-instance part-level ground truth,
                           used to add the part-level mask cost when use_parts is True

            mask_features: Tensor of dim [batch_size, num_channels, H, W], the shared pixel-decoder
                 features needed by parts_decoder to predict part masks for every query. Required
                 (together with target part annotations) for the part-level cost to be computed.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, mask_features)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
            "cost_mask_parts: {}".format(self.cost_mask_parts),
            "cost_dice_parts: {}".format(self.cost_dice_parts),
            "use_parts: {}".format(self.use_parts),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
