from easydict import EasyDict

class Configuration:
    models = EasyDict({
        "grid_size": 7,
        "num_bboxes": 2,
        "num_classes": 20,
        "image_size": [448, 448]
    })

    dataset = EasyDict({
        "image_path": "dataset/VOC/images",
        "anno_path": "dataset/VOC/labels",
        "txt_train_path": ["dataset/VOC/images_id/trainval2007.txt", "dataset/VOC/images_id/trainval2012.txt"],
        "txt_val_path": ['dataset/VOC/images_id/test2007.txt'],
        "label2id": "dataset/VOC/label_to_id.json",
    })

    trainval = EasyDict({
        "epochs": 150,
        "eval_step": 5,
        "bz_train": 24,
        "bz_valid": 8,
        "n_workers": 8,
        "iou_thresh": 0.4,
        "conf_thresh": 0.45,
        "apply_iou": 'giou'
    })

    debugging = EasyDict({
        "tensorboard_debug": "exps/tensorboard",
        "decode_yolo_debug": "exps/decode_yolo",
        "training_debug": "exps/training",
        "dataset_debug": "exps/dataset",
        "valid_debug": "exps/valid",
        "test_cases": "exps/test_cases",
        "prediction_debug": "exps/prediction",
        "ckpt_dirpath": "src/weights",
        "conf_debug": 0.3,
        "idxs_debug": [0, 1, 2, 3, 4, 5, 6, 7],
        "augmentation_debug": "exps/augmentation",
        "log_file": "logs/yolov1.log",
    })