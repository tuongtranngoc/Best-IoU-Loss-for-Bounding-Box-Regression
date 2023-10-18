import os
import cv2
import json
import glob
import xmltodict
from .utils import *
from tqdm import tqdm
from ..config.configs import Configuration as cfg
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import Dataset


from ..data import logger


class BaseDataset(Dataset):
    def __init__(self) -> None:
        pass

    def load_data_coco_format(self): 
        dataset = []
        with open(self.label_path, 'r') as f_json:
            data = json.load(f_json)
        f_json.close()
        images = {
            x['id']: x
            for x in data['images']
        }
        annotations = data['annotations']
    
        anns_imgid = defaultdict(list)
        for anns in annotations:
            anns_imgid[anns['image_id']].append(anns)

        for img_id, anns in tqdm(anns_imgid.items(), desc="Parsing coco data ..."):
            image = images[img_id]
            fname, w, h = image['file_name'], image['width'], image['height']
            img_pth = os.path.join(self.image_path, fname)
            if os.path.exists(img_pth):
                img = cv2.imread(img_pth)
            else:
                continue
            bboxes = []
            for ann in anns:
                if ann['iscrowd']: continue
                bbox = ann['bbox']
                cate = ann['category_id'] - 1
                bbox_info = [cate] + bbox
                if bbox_info not in bboxes:
                    bboxes.append(bbox_info)
        
            dataset.append([img, np.array(bboxes)])
        
        return dataset

    def load_dataset_voc_format(self, image_dirs, anno_dirs, txt_files):
        logger.info(f'Loading voc dataset from {txt_files}')
        dataset = []
        id_map = json.load(open(cfg.dataset.label2id))
        if len(txt_files) > 0:
            for txt_file in txt_files:
                image_ids = []
                image_ids = open(txt_file).read().strip().split('\n')
                root_dir = os.path.basename(txt_file).split('.txt')[0]
                anno_dir = os.path.join(anno_dirs, root_dir)
                image_dir = os.path.join(image_dirs, root_dir)
                for im_id in tqdm(image_ids, desc="Parsing VOC data ..."):
                    anno_path = os.path.join(anno_dir, im_id + '.xml')
                    anno = EasyDict(xmltodict.parse(open(anno_path).read())).annotation
                    image_path = os.path.join(image_dir, anno.filename)
                    if type(anno.object) is not list:
                        anno.object = [anno.object]
                    bboxes = []
                    for item in anno.object:
                        if item.difficult == '1': continue
                        box = item.bndbox
                        box = [box.xmin, box.ymin, box.xmax, box.ymax]
                        box = [eval(c) for c in box]
                        label = id_map[item.name]
                        box_info = [label] + box
                        bboxes.append(box_info)
                    dataset.append([image_path, np.array(bboxes, dtype=np.float32)])
        return dataset