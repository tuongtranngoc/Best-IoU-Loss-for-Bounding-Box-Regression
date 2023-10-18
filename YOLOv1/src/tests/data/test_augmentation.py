import cv2
import json
from ...config import CFG as cfg
from torch.utils.data import DataLoader
from ...data.dataset_yolo import YoloDatset
from ...utils.visualization import Vizualization

ds = YoloDatset('dataset/VOC/images', 'dataset/VOC/labels', ['dataset/VOC/images_id/test2007.txt'], True)
val_dataloader = DataLoader(ds, batch_size=10, shuffle=False)

id_map = json.load(open('dataset/VOC/label_to_id.json'))
id2classes = {
    id_map[k]: k
    for k in id_map.keys()
}

def test():
    for i, (images, labels) in enumerate(val_dataloader):
        gt_bboxes, gt_conf, gt_cls = Vizualization.reshape_data(labels[i].unsqueeze(0))
        gt_bboxes, gt_conf, gt_cls = gt_bboxes.reshape((-1, 4)), gt_conf.reshape(-1), gt_cls.reshape(-1)
        gt_bboxes, gt_conf, gt_cls = Vizualization.label2numpy(gt_bboxes, gt_conf, gt_cls)
        image = images[i]
        image = Vizualization.draw_debug(image, gt_bboxes, gt_conf, gt_cls, cfg['conf_thresh'], 'gt')
        cv2.imwrite(f'{cfg["augmentation_debug"]}/{i}.png', image)


if __name__ == "__main__":
    test()