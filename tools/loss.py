import torch

import torch.distributions as D


def uce_loss(alpha, y, weights):
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha) + 1e-10) * weights, dim=1, keepdim=True)

    return A.mean()


def entropy_reg(alpha, beta_reg=.0005):
    reg = D.Dirichlet(alpha).entropy()

    return -beta_reg * reg.mean()
