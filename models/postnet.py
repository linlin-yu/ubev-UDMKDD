from models.backbones.lss.lift_splat_shoot import BevEncodePostnet

from models.model import *
from tools.loss import *
from tools.uncertainty import *
import torch.nn as nn


class Postnet(Model):
    def __init__(self, *args, **kwargs):
        super(Postnet, self).__init__(*args, **kwargs)

    def create_backbone(self, backbone):
        self.backbone = nn.DataParallel(
            backbones[backbone](n_classes=self.n_classes).to(self.device),
            output_device=self.device,
            device_ids=self.devices
        )

        if backbone == 'lss':
            self.backbone.module.bevencode = BevEncodePostnet(inC=self.backbone.module.camC, outC=self.backbone.module.outC).to(self.device)

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
            A = u_focal_loss(alpha, y, weights=self.weights, n=2)
        else:
            raise NotImplementedError()

        if beta_lambda > 0:
            A += entropy_reg(alpha, beta_lambda)

        return A.mean()

    def loss_ood(
        self, alpha, y, ood,
        beta_lambda=.0005,
        ood_lambda=.0005
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

