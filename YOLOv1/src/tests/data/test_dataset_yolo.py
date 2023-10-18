import os
import cv2

from ...utils import *
from ...data import CFG
from ...data.augmentation import *
from ...data.dataset import BaseDatset
from ...utils.torch_utils import BoxUtils

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TestYoloDatset(BaseDatset):
    def __init__(self, image_path, label_path, txt_path, is_augment=False) -> None:
        self.cfg = CFG
        self.aug = AlbumAug()
        self.txt_path = txt_path
        self.image_path = image_path    
        self.label_path = label_path
        self.is_augment = is_augment
        self.image_size = self.cfg['image_size']
        self.tranform = Transform(self.image_size)
        self.dataset_voc = self.load_dataset_voc_format(self.image_path, self.label_path, self.txt_path)
    
    def get_image_label(self, image_pth, bboxes, labels):
        image = cv2.imread(image_pth).astype(np.float32)
        if self.is_augment:
            image, bboxes, labels = self.aug(image, bboxes, labels)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes, labels = self.tranform(image, bboxes, labels)
        return image, bboxes, labels
    
    def make_grid_cells(self, cls_ids, boxes, image, index):
        S = self.cfg['S']
        B = self.cfg['B']
        C = self.cfg['C']
        # Divide the input image into SXS grid cells
        grid_cell_i = self.image_size[0] / S
        grid_cell_j = self.image_size[1] / S
        # ===> Debug
        image = BoxUtils.image_to_numpy(image)
        w, h = image.shape[:2]
        for c in range(0, S+1):
            image = cv2.line(image, (int(grid_cell_i) * c, 0), ((int(grid_cell_i) * c, h)), color=(0, 0, 255), thickness=2)
            image = cv2.line(image, (0, int(grid_cell_j) * c), (w, int(grid_cell_j) * c), color=(0, 0, 255), thickness=2)

        # Define input shape
        target_cxcywh = torch.zeros((S, S, 5 * B + C), dtype=torch.float32)
        
        for class_id, bbox in zip(cls_ids, boxes):
            x_min, y_min, x_max, y_max = bbox.copy()
            # Compute center of an object
            x_c = (x_min + x_max) / 2
            y_c = (y_min + y_max) / 2
            
            # Determine cell position of object
            pos_i = int(x_c // grid_cell_i)
            pos_j = int(y_c // grid_cell_j)

            if target_cxcywh[pos_j, pos_i, 8] > 0 or target_cxcywh[pos_j, pos_i, 9] > 0:
                continue
            image = cv2.circle(image, (int(x_c), int(y_c)), color=(255, 0, 0), thickness=-1, radius=5)
            image = cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(255, 0, 0), thickness=1)
            # Each grid cell contains:
            #   bbox1: [x_c, y_c, w_box, h_box, p_c, p0, p1, ..., pn]
            #   bbox2: [x_c, y_c, w_box, h_box, p_c, p0, p1, ..., pn]
            
            p_cls = torch.zeros(C, dtype=torch.float32)
            p_cls[int(class_id)-1] = 1.0
            conf_cls = torch.ones((2, )).long()
            box = torch.FloatTensor([
                (x_c - (pos_i * grid_cell_i)) / grid_cell_i,
                (y_c - (pos_j * grid_cell_j)) / grid_cell_j, 
                np.sqrt((x_max - x_min) / self.image_size[0]), 
                np.sqrt((y_max - y_min) / self.image_size[1])]).repeat(B)
        
            # Assign bboxes to each grid cell
            grid_cell = torch.cat([box, conf_cls, p_cls], dim=-1)
            target_cxcywh[pos_j, pos_i, :] = grid_cell

            
        os.makedirs(os.path.join(self.cfg['test_cases'], 'test_dataset_yolo'), exist_ok=True)
        cv2.imwrite(os.path.join(self.cfg['test_cases'], 'test_dataset_yolo', f'{index}.png'), image)
        return target_cxcywh
    
    def __getitem__(self, index):
        image_path, labels = self.dataset_voc[index]
        cls_ids, bboxes = labels[:, 0], labels[:, 1:]
        image, bboxes, cls_ids = self.get_image_label(image_path, bboxes, cls_ids)
        target = self.make_grid_cells(cls_ids, bboxes, image, index)
        return image, target

    def __len__(self):
        return len(self.dataset_voc)
    

print("Testing dataset_yolo ...")
ds = TestYoloDatset('dataset/VOC/images', 'dataset/VOC/labels', ['dataset/VOC/images_id/test2007.txt'], False)
val_dataloader = DataLoader(ds, batch_size=20, shuffle=True)

next(iter(val_dataloader))