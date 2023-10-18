
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class Transform:
    def __init__(self, image_size) -> None:
        self.image_size = image_size
        self.transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )

    def __call__(self, image, bboxes, labels):
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        transformed_image = transformed['image'] 
        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_labels = transformed['labels']
        return transformed_image, transformed_bboxes, transformed_labels


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, image):
        image -= (self.mean * 255.)
        image /= (self.std * 255.)
        return image


class Unnormalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image):
        image *= (self.std * 255.)
        image += (self.mean * 255.)
        image = np.clip(image, 0, 255).astype(np.float32)
        return image