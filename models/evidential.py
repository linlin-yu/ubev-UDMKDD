from models.model import Model
from tools.uncertainty import *
from tools.loss import *

import torch.distributions as D


class Evidential(Model):
    def __init__(self, *args, **kwargs):
        super(Evidential, self).__init__(*args, **kwargs)

        self.weights = self.weights.view(1, self.weights.shape[0], 1, 1)

    @staticmethod
    def aleatoric(alpha):
        return dissonance(alpha)

    @staticmethod
    def epistemic(alpha):
        return vacuity(alpha)

    @staticmethod
    def activate(alpha):
        return alpha / torch.sum(alpha, dim=1, keepdim=True)

    def loss(self, alpha, y, entropy_lambda=.0001):
        A = uce_loss(alpha, y, self.weights)

        if entropy_lambda > 0:
            A += entropy_reg(alpha)

        return A

    def forward(self, images, intrinsics, extrinsics, limit=None):
        evidence = self.backbone(images, intrinsics, extrinsics).relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1

        return alpha

