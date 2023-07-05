from focal_loss.focal_loss import FocalLoss

from models.model import Model
from tools.loss import *
from tools.uncertainty import *


class Baseline(Model):
    def __init__(self, *args, **kwargs):
        super(Baseline, self).__init__(*args, **kwargs)

    @staticmethod
    def aleatoric(logits):
        return entropy(logits)

    @staticmethod
    def epistemic(logits):
        return entropy(logits)

    @staticmethod
    def activate(logits):
        return torch.softmax(logits, dim=1)

    def loss(self, logits, target):
        if self.loss_type == 'ce':
            return ce_loss(logits, target, weights=self.weights).mean()
        elif self.loss_type == 'focal':
            return focal_loss(logits, target, weights=self.weights, n=2).mean()
        else:
            raise NotImplementedError()

    def forward(self, images, intrinsics, extrinsics):
        return self.backbone(images, intrinsics, extrinsics)
