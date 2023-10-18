from torch import nn
from .element import Conv



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=1),
            Conv(out_channels, out_channels*2, kernel_size=3, padding=1), 
            Conv(out_channels*2, out_channels, kernel_size=1), 
            Conv(out_channels, out_channels*2, kernel_size=3, padding=1),
            Conv(out_channels*2, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.convs(x)