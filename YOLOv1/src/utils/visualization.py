import os
import json

import cv2
import torch
import random
from ..data.utils import *
from .torch_utils import *


class Drawer:
    def __init__(self) -> None:
        self.id_map = json.load(open(cfg.dataset.label2id))
        self.id2classes = {
            self.id_map[k]: k
            for k in self.id_map.keys()
        }
        self.lw = 1
        self.colors = self.class2color()

    def class2color(self):
        colors = {
            k: tuple([random.randint(0, 255) for _ in range(3)])
            for k in self.id_map.keys()
        }
        colors['groundtruth'] = (255, 0, 0)
        colors['background'] = (128, 128, 128)
        return colors

    def unnormalize_bboxes(self, bbox:list):
        return [b * cfg.models.image_size[0] for b in bbox]

    def draw_box_label(self, image, bbox, conf, label, type_label=None):
        _bbox = self.unnormalize_bboxes(bbox)
        _label = self.id2classes[label+1]
    
        if type_label == 'gt': 
            color = self.colors['groundtruth']
            _text = _label
        elif type_label == 'pred':
            _text = _label + '-' + str(round(conf, 3))
            color = self.colors[_label]
        else:
            _text = _label + '-' + str(round(conf, 3))
            color = self.colors['background']

        cv2.rectangle(image,
                    (int(_bbox[0]), int(_bbox[1])),
                    (int(_bbox[2]), int(_bbox[3])),
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

        cv2.putText(image,
                    _text,
                    (int(_bbox[0]), int(_bbox[1]+0.025*cfg.models.image_size[0])),
                    0,
                    self.lw / 3,
                    color=color,
                    thickness=self.lw,
                    lineType=cv2.LINE_AA)
        
        return image


class Debuger:
    def __init__(self, save_debug_path) -> None:
        self.S = cfg.models.grid_size
        self.B = cfg.models.num_bboxes
        self.C = cfg.models.num_classes
        self.save_debug_path = save_debug_path

    def debug_output(self, dataset, idxs, model, type_infer, device, conf_thresh, apply_mns=True):
        os.makedirs(f'{self.save_debug_path}/{type_infer}', exist_ok=True)
        model.eval()
        images, targets = [], []
        for index in idxs:
            image, target = dataset[index]
            images.append(image)
            targets.append(target)

        targets = torch.stack(targets, dim=0).to(device)
        images = torch.stack(images, dim=0).to(device)
        
        pred = model(images)

        for i in range(images.size(0)):
            gt_bboxes, gt_conf, gt_cls = Vizualization.reshape_data(targets[i].unsqueeze(0))
            pred_bboxes, pred_conf, pred_cls = Vizualization.reshape_data(pred[i].unsqueeze(0))
            gt_bboxes, gt_conf, gt_cls = gt_bboxes.reshape((-1, 4)), gt_conf.reshape(-1), gt_cls.reshape(-1)
            pred_bboxes, pred_conf, pred_cls = pred_bboxes.reshape((-1, 4)), pred_conf.reshape(-1), pred_cls.reshape(-1)
            
            if apply_mns is True:
                pred_bboxes, pred_conf, pred_cls = BoxUtils.nms(pred_bboxes, 
                                                                pred_conf, 
                                                                pred_cls, 
                                                                iou_thresh=cfg.models.iou_thresh, conf_thresh=cfg.models.conf_thresh)

            gt_bboxes, gt_conf, gt_cls = Vizualization.label2numpy(gt_bboxes, gt_conf, gt_cls)
            pred_bboxes, pred_conf, pred_cls = Vizualization.label2numpy(pred_bboxes, pred_conf, pred_cls)
            
            image = images[i]

            image = Vizualization.draw_debug(image, gt_bboxes, gt_conf, gt_cls, conf_thresh, 'gt')
            image = Vizualization.draw_debug(image, pred_bboxes, pred_conf, pred_cls, conf_thresh, 'pred')
            cv2.imwrite(f'{self.save_debug_path}/{type_infer}/{i}.png', image)


class Vizualization:
    S = cfg.models.grid_size
    B = cfg.models.num_bboxes
    C = cfg.models.num_classes
    save_debug_path = cfg.debugging.prediction_debug
    os.makedirs(save_debug_path, exist_ok=True)
    drawer = Drawer()
    
    @classmethod
    def reshape_data(cls, out):
        pred_bboxes = out[..., :8]
        pred_confs = out[..., 8:10]
        pred_cls = torch.argmax(out[..., 10:], dim=-1).unsqueeze(-1).expand((-1, -1, -1, 2))
        pred_bboxes = BoxUtils.decode_yolo(pred_bboxes.reshape(-1, cls.S, cls.S, cls.B, 4))

        return pred_bboxes, pred_confs, pred_cls
    
    @classmethod
    def label2numpy(cls, *args):
        args_list = []
        for i in range(len(args)):
            args_list.append(BoxUtils.to_numpy(args[i]))
        return args_list

    @classmethod
    def image2numpy(cls, images):
        images = BoxUtils.image_to_numpy(images)
        return images
    
    @classmethod
    def draw_debug(cls, image, bboxes, confs, classes, conf_thresh, type_draw='pred'):
        image = cls.image2numpy(image)
        bboxes, confs, classes = cls.label2numpy(bboxes, confs, classes)
        for bbox, conf, label in zip(bboxes, confs, classes):
            if conf >= conf_thresh:
                image = cls.drawer.draw_box_label(image, bbox, conf, label, type_draw)
        return image