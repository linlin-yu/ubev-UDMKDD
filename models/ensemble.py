import torch

from models.model import *
from tools.loss import *
from tools.uncertainty import *

import torch.nn as nn


class ModelPackage(nn.Module):
    def __init__(self, model, n_models, n_classes):
        super(ModelPackage, self).__init__()
        self.models = nn.ModuleList([model(n_classes=n_classes) for _ in range(n_models)])

    def forward(self, images, intrinsics, extrinsics):
        out = [model(images[0], intrinsics[0], extrinsics[0]) for model in self.models]

        return torch.stack(out)


class Ensemble(Model):
    def __init__(self, *args, **kwargs):
        super(Ensemble, self).__init__(*args, **kwargs)
        self.n_models = 3

    def create_backbone(self, backbone, n_models=3):
        print("Ensemble activation")

        self.backbone = nn.DataParallel(
            ModelPackage(
                backbones[backbone],
                n_models=n_models,
                n_classes=self.n_classes
            ).to(self.device),
            output_device=self.device,
            device_ids=self.devices,
            dim=1
        )

    @staticmethod
    def aleatoric(logits):
        unc = entropy(logits, dim=2)
        return torch.mean(unc, dim=0)

    @staticmethod
    def epistemic(logits):
        var = torch.var(Ensemble.activate(logits), dim=0)

        return 1 - 1 / var

    @staticmethod
    def activate(logits):
        probs = torch.softmax(logits, dim=2)
        return torch.mean(probs, dim=0)

    def loss(self, logits, target):
        losses = torch.zeros(logits.shape[0])

        for i in range(logits.shape[0]):
            if self.loss_type == 'ce':
                losses[i] = ce_loss(logits[i], target, weights=self.weights).mean()
            elif self.loss_type == 'focal':
                losses[i] = focal_loss(logits[i], target, weights=self.weights, n=2).mean()
            else:
                raise NotImplementedError()

        return losses.mean()

    def forward(self, images, intrinsics, extrinsics):
        return self.backbone(images[None], intrinsics[None], extrinsics[None])

