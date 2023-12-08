import sys
from pathlib import Path

import gdown
import torch
from torch import nn

from .backbone import build_backbone
from .neck import ConvBlock
from .head import YoloHead

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


model_urls = {
    "yolov1-vgg16": "https://drive.google.com/file/d/1yIEFsSXlsOeJVAnt164NBGmZPg8J_ZRm/view?usp=share_link",
    "yolov1-vgg16-bn": "https://drive.google.com/file/d/1NSHsPiJc3EVAo8SQX2HqSpCK3iQVocNa/view?usp=share_link",
    "yolov1-resnet18": "https://drive.google.com/file/d/1EETZU5z4c1lff3zOBk6jHFwBsORd065X/view?usp=share_link",
    "yolov1-resnet34": "https://drive.google.com/file/d/1-AAAFd8ADxquma5u36mOHB9eBM514RzI/view?usp=share_link",
    "yolov1-resnet50": "https://drive.google.com/file/d/1oc8dNiQGImQFy2aXmU7NlupL_13vvib4/view?usp=share_link",
}


class YoloModel(nn.Module):
    def __init__(self, input_size, backbone, num_classes, pretrained=True):
        super().__init__()
        self.stride = 64
        self.grid_size = input_size // self.stride
        self.num_classes = num_classes
        self.backbone, feat_dims = build_backbone(arch_name=backbone)
        self.neck = ConvBlock(in_channels=feat_dims, out_channels=512)
        self.head = YoloHead(in_channels=512, num_classes=num_classes)
        
        if pretrained:
            download_path = ROOT / "weights" / f"yolov1-{backbone}.pt"
            if not download_path.is_file():
                gdown.download(model_urls[f"yolov1-{backbone}"], str(download_path), quiet=False, fuzzy=True)
            ckpt = torch.load(download_path, map_location="cpu")
            self.load_state_dict(ckpt["model_state"], strict=False)
        
    def forward(self, x):
        self.device = x.device
        out = self.backbone(x)
        out = self.neck(out)
        out = self.head(out)
        out = out.permute(0, 2, 3, 1).contiguous()

        pred_box = torch.sigmoid(out[..., :8])
        pred_obj = torch.sigmoid(out[..., 8:10])
        pred_cls = torch.sigmoid(out[..., 10:])
   
        return torch.cat((pred_box, pred_obj, pred_cls), dim=-1)