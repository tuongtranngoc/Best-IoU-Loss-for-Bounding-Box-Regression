import torch
from torch import nn



class YoloHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_attributes = (1 + 4) * 2 + self.num_classes
        self.detect = nn.Conv2d(in_channels, self.num_attributes, kernel_size=1)
        

    def forward(self, x):
        out = self.detect(x)
        return out