from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

def compute_loss(target_bboxes, pred_bboxes):
    """Reference: https://arxiv.org/abs/1902.09630
        target_bboxes: ground-truth boxes [N, H, W]
        pred_bboxes: predicted boxes [N, H, W]
    """
    
