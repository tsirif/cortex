# Implementation based on:
# https://github.com/ajbrock/BigGAN-PyTorch

import torch
from torch import nn
from torch.nn import functional as F

from .spectral_norm import SNConv2d


class SelfAttention(nn.Module):
    def __init__(self, dim_h, spectral_norm=False):
        super(SelfAttention, self).__init__()
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d

        self.theta = Conv2d(dim_h, dim_h // 8, kernel_size=1,
                            padding=0, bias=False)
        self.phi = Conv2d(dim_h, dim_h // 8, kernel_size=1,
                          padding=0, bias=False)
        self.g = Conv2d(dim_h, dim_h // 2, kernel_size=1,
                        padding=0, bias=False)
        self.o = Conv2d(dim_h // 2, dim_h, kernel_size=1,
                        padding=0, bias=False)

        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        _, c, h, w = x.size()
        theta = self.theta(x).view(-1, c // 8, h * w).transpose(1, 2)
        phi = F.max_pool2d(self.phi(x), 2).view(-1, c // 8, (h * w) // 4)
        beta = F.softmax(theta.bmm(phi), -1).transpose(1, 2)
        g = F.max_pool2d(self.g(x), 2).view(-1, c // 2, (h * w) // 4)
        o = self.o(g.bmm(beta).view(-1, c // 2, h, w))
        return self.gamma * o + x
