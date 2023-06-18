import torch.nn as nn

from models.backbones.cvt.decoder import *
from models.backbones.cvt.encoder import *


def convert(intrins):
    intrins[:, :, 0, 0] *= W / 1600
    intrins[:, :, 0, 2] *= W / 1600
    intrins[:, :, 1, 1] *= (H + O) / 900
    intrins[:, :, 1, 2] *= (H + O) / 900
    intrins[:, :, 1, 2] -= O
    return intrins


class Shrink(nn.Module):
    def __init__(self):
        super(Shrink, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(7, 5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        dim_last: int = 64,
        n_classes: int = 4,
    ):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(128, [128, 128, 64])

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, n_classes, 1))

        self.drop = False

    def forward(self, images, intrinsics, extrinsics):
        x, atts = self.encoder(images, intrinsics, extrinsics)
        y = self.decoder(x)

        return self.to_logits(y)