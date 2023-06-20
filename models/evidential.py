from models.model import Model
from tools.uncertainty import *
from tools.loss import *


class Evidential(Model):
    def __init__(self, *args, **kwargs):
        super(Evidential, self).__init__(*args, **kwargs)

    @staticmethod
    def aleatoric(alpha):
        return dissonance(alpha)

    @staticmethod
    def epistemic(alpha):
        return vacuity(alpha)

    @staticmethod
    def activate(alpha):
        return alpha / torch.sum(alpha, dim=1, keepdim=True)

    def loss(self, alpha, y, beta_lambda=.0005):
        if self.loss_type == 'ce':
            A = uce_loss(alpha, y, weights=self.weights)
        elif self.loss_type == 'focal':
            A = u_focal_loss(alpha, y, weights=self.weights)
        else:
            raise NotImplementedError()

        if beta_lambda > 0:
            A += entropy_reg(alpha, beta_lambda)

        return A.mean()

    def loss_ood(
        self, alpha, y, ood,
        beta_lambda=.0005,
        ood_lambda=.0001
    ):
        A = uce_loss(alpha, y, self.weights)

        if beta_lambda > 0:
            A += entropy_reg(alpha, beta_reg=beta_lambda)

        A = A[1 - ood].mean()

        if ood_lambda > 0:
            A += ood_reg(alpha, ood) * ood_lambda

        return A

    def forward(self, images, intrinsics, extrinsics, limit=None):
        evidence = self.backbone(images, intrinsics, extrinsics).relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1

        return alpha

