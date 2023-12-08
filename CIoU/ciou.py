from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import math


def compute_loss(target_bboxes, pred_bboxes):
    """Reference: https://arxiv.org/pdf/1911.08287.pdf
    Args:
        target_bboxes (torch.Tensor): Target boundong boxes, [N, H, W, 4]
        pred_bboxes (torch.Tensor): Predicted bounding boxes, [N, H,W, 4]
    """
    # Compute intersections
    eps = 1e-7
    x1 = torch.max(target_bboxes[..., 0], pred_bboxes[..., 0])
    y1 = torch.max(target_bboxes[..., 1], pred_bboxes[..., 1])
    x2 = torch.min(target_bboxes[..., 2], pred_bboxes[..., 2])
    y2 = torch.min(target_bboxes[..., 3], pred_bboxes[..., 3])

    intersects = torch.clamp((x2-x1), min=0.0) * torch.clamp((y2-y1), min=0.0)

    # Compute unions
    A = abs((target_bboxes[..., 2]-target_bboxes[..., 0]) * target_bboxes[..., 3]-target_bboxes[..., 1])
    B = abs((pred_bboxes[..., 2]-pred_bboxes[..., 0]) * pred_bboxes[..., 3]-pred_bboxes[..., 1])

    unions = A + B - intersects

    ious = intersects / unions

    cx1 = torch.min(target_bboxes[..., 0], pred_bboxes[..., 0])
    cy1 = torch.min(target_bboxes[..., 1], pred_bboxes[..., 1])
    cx2 = torch.max(target_bboxes[..., 2], pred_bboxes[..., 2])
    cy2 = torch.max(target_bboxes[..., 3], pred_bboxes[..., 3])
    
    # Compute Euclidean between central points and diagonal lenght
    c_dist = ((target_bboxes[..., 2] + target_bboxes[..., 0] - pred_bboxes[..., 2] - pred_bboxes[..., 0]) ** 2 + \
              (target_bboxes[..., 3] + target_bboxes[..., 1] - pred_bboxes[..., 3] - pred_bboxes[..., 1]) ** 2) / 4
    
    diagonal_l2 = (cx2-cx1) **2 + (cy2-cy1) ** 2

    # Postive trade-off parameter and asspect ratio
    v = (4/(math.pi**2)) * torch.pow((torch.atan((target_bboxes[..., 2]-target_bboxes[..., 0])/(target_bboxes[..., 3]-target_bboxes[..., 1]+eps))- \
        torch.atan((pred_bboxes[..., 2]-pred_bboxes[..., 0])/(pred_bboxes[..., 3]-pred_bboxes[..., 1]+eps))), 2)
    
    with torch.no_grad():
        alpha = v / (1 - ious + v + eps)

    cious = ious - (c_dist / diagonal_l2 + alpha * v)
    cious = torch.clamp(cious, min=-1.0, max=1.0)

    return cious

