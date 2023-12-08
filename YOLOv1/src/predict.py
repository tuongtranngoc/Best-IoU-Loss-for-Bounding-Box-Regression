from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
from tqdm import tqdm

from . import cfg
from .utils.visualization import *
from .models.modules.yolo import YoloModel


class Predictor:
    def __init__(self, args) -> None:
        self.args = args
        self.transform = A.Compose(
            [
                A.Resize(cfg.models.image_size[0], cfg.models.image_size[1]),
                A.Normalize(),
                ToTensorV2(),
            ])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YoloModel(
            input_size=cfg.models.image_size[0],
            backbone=self.args.model_type,
            num_classes=cfg.models.num_classes,
            pretrained=False).to(self.device)
        self.model = self.load_weight(self.model, self.args.weight_path)
        
    def predict(self, image_pth):
        image = cv2.imread(image_pth)
        image = self._tranform(image)
        image = image.to(self.device)
        image = image.unsqueeze(0)
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(image)
            pred_bboxes, pred_conf, pred_cls = Vizualization.reshape_data(out)
            pred_bboxes, pred_conf, pred_cls = pred_bboxes.reshape((-1, 4)), pred_conf.reshape(-1), pred_cls.reshape(-1)
            pred_bboxes, pred_conf, pred_cls = BoxUtils.nms(pred_bboxes, pred_conf, pred_cls, self.args.iou_thresh, self.args.conf_thresh)
            image = Vizualization.draw_debug(image, pred_bboxes, pred_conf, pred_cls, cfg.trainval.conf_thresh)
            cv2.imwrite(f'{cfg.debugging.prediction_debug}/{os.path.basename(image_pth)}', image)

    def _tranform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        return image["image"]
    
    def load_weight(self, model, weight_path):
        if os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            return model
        else:
            raise Exception(f"Path to model {weight_path} not exist")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet34',
                        help='Model selection: resnet18, resnet34, resnet50')
    parser.add_argument('--weight_path', type=str, 
                        help='Path to model weight')
    parser.add_argument('--input_folder', type=str,
                        help='Path to input images')
    parser.add_argument('--output_folder', type=str,
                        help='Path to predicted output')
    parser.add_argument('--conf_thresh', type=float, default=cfg.trainval.conf_thresh,
                        help='Confidence threshold for nms')
    parser.add_argument('--iou_thresh', type=float, default=cfg.trainval.iou_thresh,
                        help='IoU threshold for nms')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    predictor = Predictor(args)
    IMAGE_ID = "dataset/VOC/images_id/test2007.txt"
    IMAGE_PTH = "dataset/VOC/images/test2007"
    with open(IMAGE_ID, 'r') as f_id:
        list_img_ids = f_id.readlines()
        for img_id in tqdm(list_img_ids):
            img_id = img_id.strip()
            image_path = os.path.join(IMAGE_PTH, img_id + '.jpg')
            result = predictor.predict(image_path)
