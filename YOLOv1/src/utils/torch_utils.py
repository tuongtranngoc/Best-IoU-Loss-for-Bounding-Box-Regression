
import cv2
import torch
import torchvision
import numpy as np

from ..utils import cfg
from ..data.utils import Unnormalize


class IoULoss:
    
    @classmethod
    def compute_iou(cls, target, pred):
        # eps = 1e-6
        x = BoxUtils.decode_yolo(target[..., :4])
        y = BoxUtils.decode_yolo(pred[..., :4])
        x1 = torch.max(x[..., 0], y[..., 0])
        y1 = torch.max(x[..., 1], y[..., 1])
        x2 = torch.min(x[..., 2], y[..., 2])
        y2 = torch.min(x[..., 3], y[..., 3])
        intersects = torch.clamp((x2-x1), 0) * torch.clamp((y2-y1), 0)
        unions = abs((x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])) + abs((y[..., 2] - y[..., 0]) * (y[..., 3] - y[..., 1])) - intersects
        intersects[intersects.gt(0)] = intersects[intersects.gt(0)] / unions[intersects.gt(0)]
        return intersects

    @classmethod
    def compute_GIoU(cls, target, pred):
        x = BoxUtils.decode_yolo(target[..., :4])
        y = BoxUtils.decode_yolo(pred[..., :4])

        x1 = torch.max(x[..., 0], y[..., 0])
        y1 = torch.max(x[..., 1], y[..., 1])
        x2 = torch.min(x[..., 2], y[..., 2])
        y2 = torch.min(x[..., 3], y[..., 3])
        intersects = torch.clamp((x2-x1), 0) * torch.clamp((y2-y1), 0)
        unions = abs((x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])) + abs((y[..., 2] - y[..., 0]) * (y[..., 3] - y[..., 1])) - intersects
        intersects[intersects.gt(0)] = intersects[intersects.gt(0)] / unions[intersects.gt(0)]
        
        cx1 = torch.min(x[..., 0], y[..., 0])
        cy1 = torch.min(x[..., 1], y[..., 1])
        cx2 = torch.max(x[..., 2], y[..., 2])
        cy2 = torch.max(x[..., 3], y[..., 3])
        c_intersects = (cx2-cx1) * (cy2-cy1)
    
        return intersects - (c_intersects - unions) / c_intersects

    @classmethod
    def compute_DIoU(cls, target, pred):
        x = BoxUtils.decode_yolo(target[..., :4])
        y = BoxUtils.decode_yolo(pred[..., :4])

        x1 = torch.max(x[..., 0], y[..., 0])
        y1 = torch.max(x[..., 1], y[..., 1])
        x2 = torch.min(x[..., 2], y[..., 2])
        y2 = torch.min(x[..., 3], y[..., 3])
        intersects = torch.clamp((x2-x1), 0) * torch.clamp((y2-y1), 0)
        unions = abs((x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])) + abs((y[..., 2] - y[..., 0]) * (y[..., 3] - y[..., 1])) - intersects
        intersects[intersects.gt(0)] = intersects[intersects.gt(0)] / unions[intersects.gt(0)]

        cx1 = torch.min(x[..., 0], y[..., 0])
        cy1 = torch.min(x[..., 1], y[..., 1])
        cx2 = torch.max(x[..., 2], y[..., 2])
        cy2 = torch.max(x[..., 3], y[..., 3])
        c_dist = ((x[..., 2] + x[..., 0] - y[..., 2] - y[..., 0]) ** 2 + (x[..., 3]+ x[..., 1] - y[..., 3] - y[..., 1]) ** 2) / 4
        diagonal_l2 = torch.clamp((cx2-cx1), 0) **2 + torch.clamp((cy2-cy1), 0) ** 2

        return intersects - c_dist / diagonal_l2


class BoxUtils:
    S = cfg.models.grid_size
    B = cfg.models.num_bboxes
    C = cfg.models.num_classes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def decode_yolo(cls, bboxes):
        bz = bboxes.size(0)
        idxs_i = torch.arange(cls.S, device=cls.device)
        idxs_j = torch.arange(cls.S, device=cls.device)
        pos_j, pos_i = torch.meshgrid(idxs_j, idxs_i, indexing='ij')
        pos_i = pos_i.expand((bz, -1, -1)).unsqueeze(3).expand((-1, -1, -1, 2))
        pos_j = pos_j.expand((bz, -1, -1)).unsqueeze(3).expand((-1, -1, -1, 2))
        xc = (bboxes[..., 0] + pos_i) / cls.S
        yc = (bboxes[..., 1] + pos_j) / cls.S
        x1 = torch.clamp(xc - bboxes[..., 2] **2 / 2, min=0)
        y1 = torch.clamp(yc - bboxes[..., 3] **2 / 2, min=0)
        x2 = torch.clamp(xc + bboxes[..., 2] **2 / 2, max=1)
        y2 = torch.clamp(yc + bboxes[..., 3] **2 / 2, max=1)
        
        return torch.stack((x1, y1, x2, y2) ,dim=-1)
    
    @classmethod
    def reshape_data(cls,data):
        data_cls = data[..., 10:]
        data_conf = data[..., 8:10].reshape((-1, cls.S, cls.S, cls.B, 1))
        data_bboxes = data[..., :8].reshape((-1, cls.S, cls.S, cls.B, 4))
        return data_bboxes, data_conf, data_cls
    
    @classmethod
    def to_numpy(cls, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise Exception(f"{data} is a type of {type(data)}, not numpy/tensor type")
    
    @classmethod
    def image_to_numpy(cls, image):
        if isinstance(image, torch.Tensor):
            if image.dim() > 3:
                image = image.squeeze()
            image = image.detach().cpu().numpy()
            image = image.transpose((1, 2, 0))
            image = Unnormalize()(image)
            image = np.ascontiguousarray(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise Exception(f"{image} is a type of {type(image)}, not numpy/tensor type")
        
    @classmethod
    def nms(self, pred_bboxes, pred_confs, pred_cls, iou_thresh, conf_thresh):
        conf_mask = torch.where(pred_confs>=conf_thresh)[0]
        pred_bboxes = pred_bboxes[conf_mask]
        pred_confs = pred_confs[conf_mask]
        pred_cls = pred_cls[conf_mask]

        idxs = torchvision.ops.nms(pred_bboxes, pred_confs, iou_thresh)
        nms_bboxes = pred_bboxes[idxs]
        nms_confs = pred_confs[idxs]
        nms_classes = pred_cls[idxs]
        
        return nms_bboxes, nms_confs, nms_classes