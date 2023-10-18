from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
import albumentations as A

from .utils import *


class AlbumAug:
    def __init__(self) -> None:
        self.transform = A.Compose([
            A.BBoxSafeRandomCrop(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Affine(p=0.3, rotate=15),
            A.ShiftScaleRotate(p=0.2, rotate_limit=15),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20),
            A.ToGray(p=0.01),
            A.Blur(p=0.01, blur_limit=5),
            A.MedianBlur(p=0.01, blur_limit=5),
            A.RandomBrightnessContrast(p=0.3),
            ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2),
        )
    
    def __call__(self, image, bboxes, labels):
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        transformed_image = transformed['image'] 
        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_labels = transformed['labels']
        return transformed_image, transformed_bboxes, transformed_labels
