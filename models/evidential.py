from models.model import Model
from tools.uncertainty import *


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

    def loss(self, alpha, y):
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha) + 1e-10) * self.weights, dim=1, keepdim=True)

        return A.mean()

    def forward(self, images, intrinsics, extrinsics, limit=None):
        evidence = self.backbone(images, intrinsics, extrinsics).relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)

        alpha = evidence + 1
        return alpha

