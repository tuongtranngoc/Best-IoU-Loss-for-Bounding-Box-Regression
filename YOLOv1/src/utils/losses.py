from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import cfg
from .torch_utils import *


class SumSquaredError(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, apply_IoU=None) -> None:
        super(SumSquaredError, self).__init__()
        self.apply_IoU = apply_IoU
        self.cls_loss = nn.MSELoss()
        self.bbox_loss = nn.MSELoss()
        self.conf_loss = nn.MSELoss()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, gt, pred):
        bz = gt.size(0)
        gt_bboxes, gt_conf, gt_cls = BoxUtils.reshape_data(gt)
        pred_bboxes, pred_conf, pred_cls = BoxUtils.reshape_data(pred)
        
        # Compute IOU between GT and PRED
        gt_bboxes = gt_bboxes.clone()
        ious = IoULoss.compute_iou(gt_bboxes, pred_bboxes)
        max_ious, max_idxs = torch.max(ious, dim=-1)
        
        # Create responsible for predicting for each object based on 
        # prediction has the highest current IOU with GT
        one_obj_ij = (gt_conf[..., 0] == 1)
        one_obj_i = (gt_conf[..., 0] == 1)[..., 0]  
        
        idxs = torch.where(one_obj_i==True)
        for bz, j, i in zip(*idxs):
            bz, j, i = bz.item(), j.item(), i.item()
            max_idx = max_idxs[bz, j, i]
            one_obj_ij[bz, j, i, 1-max_idx] = False
            gt_conf[bz, j, i, max_idx, 0] = max_ious[bz, j, i]
            gt_conf[bz, j, i, 1-max_idx, 0] = 0
        
        one_noobj_ij = ~one_obj_ij
        
        # Apply IOU loss for bounding box regression
        if self.apply_IoU is not None:
            if self.apply_IoU=="GIoU":
                box_loss = 1 - IoULoss.compute_GIoU(gt_bboxes, pred_bboxes)
            elif self.apply_IoU=="DIoU":
                box_loss = 1 - IoULoss.compute_DIoU(gt_bboxes, pred_bboxes)
            else:
                raise Exception("If using apply_IoU, Please use one of following loss functions: GIoU, DIoU")
            box_loss = box_loss[one_obj_ij].mean()
            self.lambda_coord = 1.0
            self.lambda_noobj = 1.0
        else:
            box_loss = self.bbox_loss(pred_bboxes[one_obj_ij], gt_bboxes[one_obj_ij])
        
        obj_loss = self.conf_loss(pred_conf[..., 0][one_obj_ij], gt_conf[..., 0][one_obj_ij])

        noobj_loss = self.conf_loss(pred_conf[..., 0][one_noobj_ij], gt_conf[..., 0][one_noobj_ij])
        # class loss
        cls_loss = self.cls_loss(pred_cls[one_obj_i], gt_cls[one_obj_i])
        # bounding box loss
        box_loss = self.lambda_coord  * box_loss
        # confidence loss
        conf_loss = (self.lambda_noobj * noobj_loss + obj_loss)
        
        return box_loss, conf_loss, cls_loss