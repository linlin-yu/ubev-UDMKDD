from models.model import Model
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
        return torch.nn.functional.cross_entropy(logits, target,
                                                 weight=self.weights)

    def forward(self, images, intrinsics, extrinsics):
        return self.backbone(images, intrinsics, extrinsics)
