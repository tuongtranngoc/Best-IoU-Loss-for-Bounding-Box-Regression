from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch


def compute_loss(target_bboxes, pred_bboxes):
    """Reference: https://arxiv.org/abs/1902.09630

    Args:
        target_bboxes (torch.Tensor): Target boundong boxes, [N, H, W, 4]
        pred_bboxes (torch.Tensor): Predicted bounding boxes, [N, H,W, 4]
    """
    # Compute intersection
    x1 = torch.max(target_bboxes[..., 0], pred_bboxes[..., 0])
    y1 = torch.max(target_bboxes[..., 1], pred_bboxes[..., 1])
    x2 = torch.min(target_bboxes[..., 2], pred_bboxes[..., 2])
    y2 = torch.min(target_bboxes[..., 3], pred_bboxes[..., 3])

    intersections = torch.clamp(x2-x1, 0) * torch.clamp(y2-y1, 0)

    # Compute Union
    A = abs((target_bboxes[..., 2]-target_bboxes[..., 0]) * (target_bboxes[..., 3]-target_bboxes[..., 1]))
    B = abs((pred_bboxes[..., 2]-pred_bboxes[..., 0]) * (pred_bboxes[..., 3]-pred_bboxes[..., 1]))

    unions = A + B - intersections
    iou = intersections / unions

    cx1 = torch.min(target_bboxes[..., 0], pred_bboxes[..., 0])
    cy1 = torch.min(target_bboxes[..., 1], pred_bboxes[..., 1])
    cx2 = torch.max(target_bboxes[..., 2], pred_bboxes[..., 2])
    cy2 = torch.max(target_bboxes[..., 3], pred_bboxes[..., 3])

    C = (cx2 - cx1) * (cy2 - cy1)

    return iou-(C-unions)/C
    
