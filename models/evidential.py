from models.model import Model
from tools.loss import *
from tools.uncertainty import *


class Evidential(Model):
    def __init__(self, *args, **kwargs):
        super(Evidential, self).__init__(*args, **kwargs)

        self.beta_lambda = 0.0005
        print(f"BETA LAMBDA: {self.beta_lambda}")

    @staticmethod
    def aleatoric(alpha):
        soft = Evidential.activate(alpha)
        max_soft, hard = soft.max(dim=1)
        return (1 - max_soft[:, None, :, :]) / torch.max(1 - max_soft[:, None, :, :])
        # return dissonance(alpha)

    @staticmethod
    def epistemic(alpha):
        return vacuity(alpha)

    @staticmethod
    def activate(alpha):
        return alpha / torch.sum(alpha, dim=1, keepdim=True)

    def loss(self, alpha, y):
        if self.loss_type == 'ce':
            A = uce_loss(alpha, y, weights=self.weights)
        elif self.loss_type == 'focal':
            A = u_focal_loss(alpha, y, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if self.beta_lambda > 0:
            A += entropy_reg(alpha, self.beta_lambda)

        return A.mean()

    def loss_ood(
        self, alpha, y, ood,
        beta_lambda=.0005,
        ood_lambda=.1,
    ):
        A = uce_loss(alpha, y, self.weights)

        if beta_lambda > 0:
            A += entropy_reg(alpha, beta_reg=beta_lambda)

        A = A[(1 - ood).unsqueeze(1).bool()].mean()

        if ood_lambda > 0:
            A += ood_reg(alpha, ood) * ood_lambda

        return A

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood):
        self.opt.zero_grad(set_to_none=True)

        outs = self(images, intrinsics, extrinsics)
        preds = self.activate(outs)

        loss = self.loss_ood(outs, labels.to(self.device), ood)
        loss.backward()

        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()

        return outs, preds, loss

    def forward(self, images, intrinsics, extrinsics, limit=None):
        evidence = self.backbone(images, intrinsics, extrinsics).relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1

        return alpha

