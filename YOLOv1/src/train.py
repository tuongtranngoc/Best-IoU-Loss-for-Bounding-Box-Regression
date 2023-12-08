from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import argparse
from torch.utils.data import DataLoader

from . import cfg
from src.eval import VocEval
from src.utils.visualization import *

from .data.utils import *
from .utils.metrics import BatchMeter
from .utils.visualization import Debuger
from .data.dataset_yolo import YoloDatset
from .utils.losses import SumSquaredError
from .models.modules.yolo import YoloModel
from .utils.tensorboard import Tensorboard

from .utils.logger import Logger
logger = Logger.get_logger("TRAINING")


class Trainer:
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_map = 0.0
        self.start_epoch = 1
        self.create_model()
        self.create_dataloader()
        self.debuger = Debuger(cfg.debugging.training_debug)
        self.eval = VocEval(
            self.val_dataset,
            self.model,
            cfg.trainval.bz_valid,
            False,
            cfg.trainval.n_workers,
            False,
            cfg.trainval.iou_thresh,
            cfg.trainval.conf_thresh)

    def create_dataloader(self):
        self.train_dataset = YoloDatset(
            cfg.dataset.image_path,
            cfg.dataset.anno_path,
            cfg.dataset.txt_train_path,
            is_augment=True)
        self.val_dataset = YoloDatset(
            cfg.dataset.image_path,
            cfg.dataset.anno_path,
            cfg.dataset.txt_val_path)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.trainval.bz_train,
            shuffle=True,
            num_workers=cfg.trainval.n_workers,
            pin_memory=False)

    def create_model(self):
        self.model = YoloModel(
            input_size=cfg.models.image_size[0],
            backbone=self.args.backbone,
            num_classes=cfg.models.num_classes,
            pretrained=True,).to(self.device)
        if cfg.trainval.apply_iou is not None:
            logger.info(f'Apply {cfg.trainval.apply_iou} for loss function ...')
        self.loss_fn = SumSquaredError(apply_IoU=cfg.trainval.apply_iou).to(self.device)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, weight_decay=5e-4, lr=1e-3)
        #self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [90, 120], gamma=0.1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, amsgrad=True)
        
        if self.args.resume:
            logger.info("Resuming training ...")
            last_ckpt = os.path.join(cfg.debugging.ckpt_dirpath, self.args.backbone, 'last.pt')
            if os.path.exists(last_ckpt):
                ckpt = torch.load(last_ckpt, map_location=self.device)
                self.start_epoch = self.resume_training(ckpt)
                logger.info(f"Loading checkpoint with start epoch: {self.start_epoch}, best mAP_50: {self.best_map}")
                

    def train(self):
        for epoch in range(self.start_epoch, cfg.trainval.epochs):
            mt_box_loss = BatchMeter()
            mt_conf_loss = BatchMeter()
            mt_cls_loss = BatchMeter()

            for bz, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)
                out = self.model(images)

                box_loss, conf_loss, class_loss = self.loss_fn(labels, out)
                total_loss = box_loss + conf_loss + class_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                mt_box_loss.update(box_loss.item())
                mt_conf_loss.update(conf_loss.item())
                mt_cls_loss.update(class_loss.item())

                print(f"Epoch {epoch} Batch {bz+1}/{len(self.train_loader)}, box_loss: {mt_box_loss.get_value(): .5f}, conf_loss: {mt_conf_loss.get_value():.5f}, class_loss: {mt_cls_loss.get_value():.5f}",
                        end="\r")

                Tensorboard.add_scalars(
                    "train",
                    epoch,
                    box_loss=mt_box_loss.get_value('mean'),
                    conf_loss=mt_conf_loss.get_value("mean"),
                    cls_loss=mt_cls_loss.get_value("mean"))

            logger.info(f"Epoch: {epoch} - box_loss: {mt_box_loss.get_value('mean'): .5f}, conf_loss: {mt_conf_loss.get_value('mean'): .5f}, class_loss: {mt_cls_loss.get_value('mean'): .5f}")

            if epoch % cfg.trainval.eval_step == 0:
                metrics = self.eval.evaluate()
                Tensorboard.add_scalars(
                    "eval_loss",
                    epoch,
                    box_loss=metrics["eval_box_loss"].get_value("mean"),
                    conf_loss=metrics["eval_conf_loss"].get_value("mean"),
                    cls_loss=metrics["eval_cls_loss"].get_value("mean"))

                Tensorboard.add_scalars(
                    "eval_map",
                    epoch,
                    mAP=metrics["eval_map"].get_value("mean"),
                    mAP_50=metrics["eval_map_50"].get_value("mean"),
                    mAP_75=metrics["eval_map_75"].get_value("mean"))

                if metrics["eval_map_50"].get_value("mean") > self.best_map:
                    self.best_map = metrics["eval_map_50"].get_value("mean")
                    best_ckpt = os.path.join(cfg.debugging.ckpt_dirpath, self.args.backbone, 'best.pt')
                    self.save_ckpt(best_ckpt, self.best_map, epoch)

            last_ckpt = os.path.join(cfg.debugging.ckpt_dirpath, self.args.backbone, 'last.pt')
            self.save_ckpt(last_ckpt, self.best_map, epoch)

            # Debug image at each training time
            with torch.no_grad():
                self.debuger.debug_output(
                    self.train_dataset,
                    cfg.debugging.idxs_debug,
                    self.model,
                    "train",
                    self.device,
                    cfg.debugging.conf_debug,
                    True,
                )
                self.debuger.debug_output(
                    self.val_dataset,
                    cfg.debugging.idxs_debug,
                    self.model,
                    "val",
                    self.device,
                    cfg.debugging.conf_debug,
                    True,
                )

    def save_ckpt(self, save_path, best_acc, epoch):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_map50": best_acc,
            "epoch": epoch,
        }
        logger.info(f"Saving checkpoint to {save_path}")
        torch.save(ckpt_dict, save_path)
    
    def resume_training(self, ckpt):
        self.best_map = ckpt['best_map50']
        start_epoch = ckpt['epoch'] + 1
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.model.load_state_dict(ckpt['model'])

        return start_epoch


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Model selection contain: vgg16, vgg16-bn, resnet18, resnet34')
    parser.add_argument('--resume', nargs='?', const=True, default=False, 
                        help='Resume most recent training')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    trainer = Trainer(args)
    trainer.train()
