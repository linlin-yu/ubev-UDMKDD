import torch

import torch.distributions as D
import torch.nn.functional as F


def ce_loss(logits, target, weights=None):
    return F.cross_entropy(logits, target, weight=weights, reduction='none')


def focal_loss(logits, target, weights=None, n=2):
    ce = ce_loss(logits, target, weights=weights)
    probs = torch.exp(-ce)

    focal = ((1 - probs) ** n * ce).unsqueeze(1)

    return focal


def uce_loss(alpha, y, weights=None):
    S = torch.sum(alpha, dim=1, keepdim=True)
    B = y * (torch.digamma(S) - torch.digamma(alpha) + 1e-10)

    if weights is not None:
        B *= weights.view(1, -1, 1, 1)

    A = torch.sum(B, dim=1, keepdim=True)

    return A


def u_focal_loss(alpha, y, weights=None, n=2):
    S = torch.sum(alpha, dim=1, keepdim=True)

    a0 = S
    aj = torch.gather(alpha, 1, torch.argmax(y, dim=1, keepdim=True))

    print(torch.mean(alpha))

    B = y * (gamma(a0 - aj + n) * gamma(a0) / (gamma(a0 + n) * gamma(a0 - aj))) * (torch.digamma(a0 + n) - torch.digamma(aj))

    if weights is not None:
        B *= weights.view(1, -1, 1, 1)

    A = torch.sum(B, dim=1, keepdim=True)

    return A


def entropy_reg(alpha, beta_reg=.0005):
    alpha = alpha.permute(0, 2, 3, 1)

    reg = D.Dirichlet(alpha).entropy().unsqueeze(1)

    return -beta_reg * reg


def ood_reg(alpha, ood):
    alpha = alpha.permute(0, 2, 3, 1)

    alpha_d = D.Dirichlet(alpha)
    target_d = D.Dirichlet(torch.ones_like(alpha))

    reg = D.kl.kl_divergence(alpha_d, target_d).unsqueeze(1)

    return reg[ood].mean()


def gamma(x):
    return torch.exp(torch.lgamma(x))