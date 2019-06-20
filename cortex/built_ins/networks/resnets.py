'''Residual encoder / decoder

'''
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import SelfAttention
from .base_network import BaseNet
from .modules import View
from .utils import (apply_nonlinearity, apply_layer_dict,
                    finish_layer_1d, finish_layer_2d, get_nonlinearity)
from .spectral_norm import SNConv2d, SNLinear, SNEmbedding


logger = logging.getLogger(__name__)


def calc_decoder_rounds(n, u, h):
    no = n - u
    rounds = max(no - u, 0) * [None]
    rounds += min(u, no) * ['up', None]
    rounds += max(u - no, 0) * ['up']
    no = n - h
    hrounds = max(no - h, 0) * [1]
    hrounds += min(h, no) * [1, 2]
    hrounds += max(h - no, 0) * [2]
    return list(zip(rounds, hrounds))


def calc_encoder_rounds(n, u, h):
    dec_rounds = calc_decoder_rounds(n, u, h)
    rounds = [('down' if up else None, div) for (up, div) in dec_rounds]
    return list(reversed(rounds))


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_x, dim_y, f_size,
                 alpha=1, resample=None, wide=True,
                 dim_in_cond=0, spectral_norm=False, name=None, **layer_args):
        super(ResBlock, self).__init__()
        if resample not in ('up', 'down', None):
            raise ValueError('invalid resample value: {}'.format(resample))
        assert(f_size % 2 == 1)
        if name is None:
            name = 'resblock_({}/{}_{})'.format(dim_in, dim_out, str(resample))
        self.name = name

        self.resample = resample
        self.alpha = alpha
        pad = (f_size - 1) // 2  # padding that preserves image w and h
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        dim_c_hidden = max(dim_in, dim_out) if wide is True else min(dim_in, dim_out)
        dim_x_hidden, dim_y_hidden = dim_x, dim_y
        if resample == 'up':
            dim_x_hidden, dim_y_hidden = dim_x * 2, dim_y * 2

        self.skip = None
        if dim_in != dim_out:
            self.skip = Conv2d(dim_in, dim_out, 1, padding=0, bias=False)

        self.pre = finish_layer_2d(name + '_pre', dim_x, dim_y, dim_in,
                                   dim_in_cond=dim_in_cond, spectral_norm=spectral_norm,
                                   **layer_args)

        self.main = nn.ModuleDict()
        conv1 = Conv2d(dim_in, dim_c_hidden, f_size, padding=pad, bias=False)
        self.main[name + '_stage1'] = conv1
        finish_layer_2d(name + '_stage1',
                        dim_x_hidden, dim_y_hidden, dim_c_hidden,
                        models_output=self.main,
                        dim_in_cond=dim_in_cond, spectral_norm=spectral_norm,
                        inplace_nonlin=True, **layer_args)
        conv2 = Conv2d(dim_c_hidden, dim_out, f_size, padding=pad, bias=False)
        self.main[name + '_stage2'] = conv2

    def _shortcut(self, x):
        if self.resample == 'up':
            x = F.interpolate(x, scale_factor=2)
        if self.skip is not None:
            x = self.skip(x)
        if self.resample == 'down':
            x = F.avg_pool2d(x, kernel_size=2)
        return x

    def forward(self, x, y=None):
        h = apply_layer_dict(self.pre, x, y=y)
        if self.resample == 'up':
            h = F.interpolate(h, scale_factor=2)
        h = apply_layer_dict(self.main, h, y=y)
        if self.resample == 'down':
            h = F.avg_pool2d(h, kernel_size=2)
        return self.alpha * h + self._shortcut(x)


class Decoder(nn.Module):
    cond_block_suffix = '_cond'

    def __init__(self, dim_in, shape_out,
                 dim_h=64, dim_h_max=1024, n_steps=3, incl_attblock=-1,
                 hierarchical=False, n_targets=0, dim_embed=0,
                 output_nonlinearity=None,
                 f_size=3, wide=True, spectral_norm=False, **layer_args):
        super(Decoder, self).__init__()
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear
        self.output_nonlinearity = output_nonlinearity

        self.hierarchical = hierarchical
        if hierarchical is True:
            dim_in_step = dim_in // (n_steps + 1)
            self.dim_in_net = dim_in - dim_in_step * n_steps
        else:
            dim_in_step = 0
            self.dim_in_net = dim_in

        self.embedding = None
        if n_targets > 0:
            assert(dim_embed > 0)
            self.embedding = nn.Embedding(n_targets, dim_embed)
            dim_in_step += dim_embed

        if not isinstance(incl_attblock, tuple):
            incl_attblock = (incl_attblock,)
        incl_attblock = tuple(x if x >= 0 else n_steps + x for x in incl_attblock)

        dim_x_out, dim_y_out, dim_h_out = shape_out
        dim_h_ = dim_h

        dim_x = max(dim_x_out // 2**n_steps, 4)
        dim_y = max(dim_y_out // 2**n_steps, 4)
        dim_h = min(dim_h_ * 2**n_steps, dim_h_max)
        dim_out = dim_x * dim_y * dim_h

        up_steps = int(math.log(dim_x_out // dim_x, 2))
        h_steps = int(math.log(dim_h // dim_h_, 2))
        logger.info("Building ResNet Decoder. steps={},up={},h={},attblock={}".format(
            n_steps, up_steps, h_steps, incl_attblock))
        rounds = calc_decoder_rounds(n_steps, up_steps, h_steps)

        self.init = nn.Sequential()
        self.init.add_module(
            'linear_({}/{})'.format(self.dim_in_net, dim_out),
            Linear(self.dim_in_net, dim_out, bias=True))
        self.init.add_module(
            'reshape_{}to{}x{}x{}'.format(dim_out, dim_h, dim_x, dim_y),
            View(-1, dim_h, dim_x, dim_y))

        self.steps = nn.ModuleList()
        dim_out = dim_h
        for i, (resample, div) in enumerate(rounds):
            step = nn.ModuleDict()

            dim_in = dim_out
            if i in incl_attblock:
                attblock = SelfAttention(dim_in, spectral_norm=spectral_norm)
                step['attblock_{}x{}x{}'.format(dim_in, dim_x, dim_y)] = attblock

            dim_out //= div
            resblock = ResBlock(dim_in, dim_out, dim_x, dim_y, f_size,
                                resample=resample, wide=wide,
                                dim_in_cond=dim_in_step,
                                spectral_norm=spectral_norm, **layer_args)
            name = resblock.name + (self.cond_block_suffix if dim_in_step > 0 else '')
            step[name] = resblock
            dim_x *= (2 if resample == 'up' else 1)
            dim_y *= (2 if resample == 'up' else 1)

            self.steps.append(step)

        assert(dim_out == dim_h_)
        assert(dim_x == dim_x_out)
        assert(dim_y == dim_y_out)
        name = 'conv_({}/{})'.format(dim_out, dim_h_out)
        final = finish_layer_2d('pre_' + name,
                                dim_x, dim_y, dim_out, **layer_args)
        pad = (f_size - 1) // 2  # padding that preserves image w and h
        final[name] = Conv2d(dim_out, dim_h_out, f_size, padding=pad, bias=False)
        self.final = nn.Sequential(final)

    def forward_from_embedding(self, x, y=None, nonlinearity=None, **nonlin_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity

        # Prepare input to network
        if self.hierarchical:
            z = x[:, :self.dim_in_net]
            chunks = torch.split(x[:, self.dim_in_net:], len(self.steps), 1)
            if self.embedding is not None:
                assert(y is not None)
                y = [torch.cat([y, inp], 1) for inp in chunks]
            else:
                y = chunks
        elif self.embedding is not None:
            z = x
            assert(y is not None)
            y = [y] * len(self.steps)
        else:
            z = x
            y = [None] * len(self.steps)

        # Forward pass
        h = self.init(z)
        for i, step in enumerate(self.steps):
            h = apply_layer_dict(step, h, y=y[i],
                                 conditional_suffix=self.cond_block_suffix)
        h = self.final(h)

        return apply_nonlinearity(h, nonlinearity, **nonlin_args)

    def forward(self, x, y=None, **nonlin):
        if self.embedding is not None:
            assert(y is not None)
            y = self.embedding(y)
        return self.forward_from_embedding(x, y=y, **nonlin)


class Encoder(nn.Module):
    def __init__(self, shape_in, dim_out,
                 dim_h=64, dim_h_max=1024, n_steps=3, incl_attblock=0,
                 n_targets=0,
                 output_nonlinearity=None,
                 f_size=3, wide=True, spectral_norm=False, **layer_args):
        super(Encoder, self).__init__()
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear
        Embedding = SNEmbedding if spectral_norm else nn.Embedding

        self.output_nonlinearity = output_nonlinearity
        if not isinstance(incl_attblock, tuple):
            incl_attblock = (incl_attblock,)
        incl_attblock = tuple(x if x >= 0 else n_steps + x for x in incl_attblock)

        dim_x_in, dim_y_in, dim_h_in = shape_in
        dim_out_ = dim_out
        dim_out = dim_h_ = dim_h

        dim_x = max(dim_x_in // 2**n_steps, 4)
        dim_y = max(dim_y_in // 2**n_steps, 4)
        dim_h = min(dim_h_ * 2**n_steps, dim_h_max)

        down_steps = int(math.log(dim_x_in // dim_x, 2))
        h_steps = int(math.log(dim_h // dim_h_, 2))
        logger.info("Building ResNet Encoder. steps={},down={},h={},attblock={}".format(
            n_steps, down_steps, h_steps, incl_attblock))
        rounds = calc_encoder_rounds(n_steps, down_steps, h_steps)

        pad = (f_size - 1) // 2  # padding that preserves image w and h
        self.init = Conv2d(dim_h_in, dim_out, f_size, padding=pad, bias=False)

        self.steps = nn.ModuleList()
        for i, (resample, mul) in enumerate(rounds):
            step = nn.ModuleDict()

            dim_in = dim_out
            dim_out *= mul
            resblock = ResBlock(dim_in, dim_out, dim_x_in, dim_y_in, f_size,
                                resample=resample, wide=wide,
                                spectral_norm=spectral_norm, **layer_args)
            step[resblock.name] = resblock
            dim_x_in //= (2 if resample == 'down' else 1)
            dim_y_in //= (2 if resample == 'down' else 1)

            if i in incl_attblock:
                attblock = SelfAttention(dim_out, spectral_norm=spectral_norm)
                step['attblock_{}x{}x{}'.format(dim_out, dim_x_in, dim_y_in)] = attblock

            self.steps.append(step)

        assert(dim_out == dim_h)
        assert(dim_x_in == dim_x)
        assert(dim_y_in == dim_y)
        self.final_activation = finish_layer_2d('final', dim_x, dim_y, dim_h,
                                                inplace_nonlin=True, **layer_args)
        self.final_linear = Linear(dim_h, dim_out_, bias=False)
        self.embedding = None
        if n_targets > 0:
            assert(dim_out_ == 1)
            self.embedding = Embedding(n_targets, dim_h)

    def forward(self, x, y=None, nonlinearity=None, **nonlin_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity

        x = self.init(x)
        for step in self.steps:
            x = apply_layer_dict(step, x)
        x = apply_layer_dict(self.final_activation, x).sum((2, 3))

        out = self.final_linear(x)
        class_out = 0
        if self.embedding is not None:
            assert(y is not None)
            class_out = self.embedding(y).mul(x).sum(1, keepdim=True)

        return apply_nonlinearity(out + class_out, nonlinearity, **nonlin_args)
