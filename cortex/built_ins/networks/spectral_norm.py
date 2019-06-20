# Implementation based on original paper:
# https://github.com/pfnet-research/sngan_projection

from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch


def sn_weight(weight, u, height, n_power_iterations=1, eps=1e-12):
    with torch.no_grad():
        for _ in range(n_power_iterations):
            v = F.normalize(torch.mv(weight.view(height, -1).t(), u),
                            dim=0, eps=eps)
            u = F.normalize(torch.mv(weight.view(height, -1), v),
                            dim=0, eps=eps, out=u)
        if n_power_iterations > 0:
            u = u.clone()

    sigma = u.dot(weight.view(height, -1).mv(v))
    return torch.div(weight, sigma), u


def SNConv2d(*args, n_power_iterations=1, eps=1e-12, dim=None, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs), name='weight', eps=eps,
                         n_power_iterations=n_power_iterations, dim=dim)


def SNLinear(*args, n_power_iterations=1, eps=1e-12, dim=None, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs), name='weight', eps=eps,
                         n_power_iterations=n_power_iterations, dim=dim)


def SNEmbedding(*args, n_power_iterations=1, eps=1e-12, dim=None, **kwargs):
    return spectral_norm(nn.Embedding(*args, **kwargs), name='weight', eps=eps,
                         n_power_iterations=n_power_iterations, dim=dim)
