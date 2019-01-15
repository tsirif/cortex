'''Residual encoder / decoder

'''
import logging
import math

import torch.nn as nn

from .base_network import BaseNet
from .modules import View
from .utils import (apply_nonlinearity,
                    finish_layer_1d, finish_layer_2d, get_nonlinearity)
from .SpectralNormLayer import SNConv2d, SNLinear


logger = logging.getLogger('cortex.models' + __name__)


def calc_decoder_rounds(n, u, h):
    no = n - u
    rounds = max(no - u, 0) * [None]
    rounds += min(u, no) * ['up', None]
    rounds += max(u - no, 0) * ['up']
    hrounds = []
    no = max(h - u, 0)
    for r in reversed(rounds):
        if h > 0 and r == 'up':
            hrounds.append(2)
            h -= 1
        elif no > 0:
            hrounds.append(2)
            no -= 1
        else:
            hrounds.append(1)
    return list(zip(rounds, reversed(hrounds))) + [(None, 1)]


def calc_encoder_rounds(n, u, h):
    dec_rounds = calc_decoder_rounds(n, u, h)
    rounds = [('down' if up else None, div) for (up, div) in dec_rounds]
    return list(reversed(rounds))


class ConvMeanPool(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None,
                 prefix='', spectral_norm=False, bias=False):
        super(ConvMeanPool, self).__init__()

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)
        name = 'cmp' + prefix

        models.add_module(name,
                          Conv2d(dim_in, dim_out, f_size, 1, 1, bias=bias))
        models.add_module(name + '_pool',
                          nn.AvgPool2d(2, count_include_pad=False))

        if nonlinearity:
            models.add_module('{}_{}'.format(
                name, nonlinearity.__class__.__name__), nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


class MeanPoolConv(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None,
                 prefix='', spectral_norm=False, bias=False):
        super(MeanPoolConv, self).__init__()

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)
        name = 'mpc' + prefix

        models.add_module(name + '_pool',
                          nn.AvgPool2d(2, count_include_pad=False))
        models.add_module(name,
                          Conv2d(dim_in, dim_out, f_size, 1, 1, bias=bias))

        if nonlinearity:
            models.add_module(
                '{}_{}'.format(name, nonlinearity.__class__.__name__),
                nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None,
                 prefix='', spectral_norm=False, bias=False):
        super(UpsampleConv, self).__init__()

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)
        name = prefix + '_usc'

        models.add_module(name + '_up', nn.Upsample(scale_factor=2))
        models.add_module(name,
                          Conv2d(dim_in, dim_out, f_size, 1, 1, bias=bias))

        if nonlinearity:
            models.add_module(
                '{}_{}'.format(name, nonlinearity.__class__.__name__),
                nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_x, dim_y, f_size, resample=None,
                 name='resblock', nonlinearity='ReLU', alpha=1,
                 spectral_norm=False, **layer_args):
        super(ResBlock, self).__init__()

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        nonlinearity = get_nonlinearity(nonlinearity)
        self.alpha = alpha

        if resample not in ('up', 'down', None):
            raise ValueError('invalid resample value: {}'.format(resample))

        skip_models = nn.Sequential()

        if dim_in != dim_out:
            if resample == 'down':
                conv = MeanPoolConv(dim_in, dim_out, f_size, prefix=name,
                                    spectral_norm=spectral_norm)
            elif resample == 'up':
                conv = UpsampleConv(dim_in, dim_out, f_size, prefix=name,
                                    spectral_norm=spectral_norm)
            elif resample is None:
                conv = Conv2d(dim_in, dim_out, f_size, 1, 1, bias=False)
            skip_models.add_module(name + '_skip', conv)
        else:
            if resample == 'down':
                skip_models.add_module(name + '_skip_pool',
                                       nn.AvgPool2d(2, count_include_pad=False))
            elif resample == 'up':
                skip_models.add_module(name + '_skip_up',
                                       nn.Upsample(scale_factor=2))
            elif resample is None:
                pass

        self.skip_models = skip_models

        models = nn.Sequential()

        # Stage 1
        finish_layer_2d(models, name, dim_x, dim_y, dim_in,
                        nonlinearity=nonlinearity, **layer_args)
        dim_c_hidden = min(dim_in, dim_out)
        if resample == 'down' or resample is None:
            conv = Conv2d(dim_in, dim_c_hidden, f_size, 1, 1, bias=True)
            dim_x_hidden, dim_y_hidden = dim_x, dim_y
        elif resample == 'up':
            conv = UpsampleConv(dim_in, dim_c_hidden, f_size,
                                prefix=name + '_stage1',
                                spectral_norm=spectral_norm, bias=True)
            dim_x_hidden, dim_y_hidden = dim_x * 2, dim_y * 2
        models.add_module(name + '_stage1', conv)
        finish_layer_2d(models, name + '_stage1',
                        dim_x_hidden, dim_y_hidden, dim_c_hidden,
                        nonlinearity=nonlinearity, **layer_args)

        # Stage 2
        if resample == 'down':
            conv = ConvMeanPool(dim_c_hidden, dim_out, f_size,
                                prefix=name,
                                spectral_norm=spectral_norm, bias=False)
        elif resample == 'up' or resample is None:
            conv = Conv2d(dim_c_hidden, dim_out, f_size, 1, 1, bias=False)
        models.add_module(name + '_stage2', conv)

        self.models = models

    def forward(self, x):
        y = self.models(x)
        y_skip = self.skip_models(x)
        return self.alpha * y + y_skip


class ResDecoder(nn.Module):
    def __init__(self, shape, dim_in=None, dim_h=64, dim_h_max=512, n_steps=3,
                 f_size=3, nonlinearity='ReLU', output_nonlinearity=None,
                 **layer_args):
        super(ResDecoder, self).__init__()
        models = nn.Sequential()

        logger.debug('Output shape: {}'.format(shape))
        dim_x_, dim_y_, dim_out_ = shape
        dim_h_ = dim_h

        nonlinearity = get_nonlinearity(nonlinearity)
        self.output_nonlinearity = output_nonlinearity

        dim_x = max(dim_x_ // 2**n_steps, 4)
        dim_y = max(dim_y_ // 2**n_steps, 4)
        dim_h = min(dim_h_ * 2**n_steps, dim_h_max)

        up_steps = int(math.log(dim_x_ / dim_x, 2))
        h_steps = int(math.log(dim_h / dim_h_, 2))
        rounds = calc_decoder_rounds(n_steps, up_steps, h_steps)

        dim_out = dim_x * dim_y * dim_h

        name = 'init_({}/{})_0'.format(dim_in, dim_out)
        models.add_module(name, nn.Linear(dim_in, dim_out))
        models.add_module('reshape_{}to{}x{}x{}to'
                          .format(dim_out, dim_h, dim_x, dim_y),
                          View(-1, dim_h, dim_x, dim_y))

        dim_out = dim_h
        for i, (resample, div) in enumerate(rounds):
            dim_in = dim_out
            dim_out //= div
            name = 'resblock_({}/{}_{:s})_{}'.format(dim_in, dim_out,
                                                     resample, i + 1)
            resblock = ResBlock(dim_in, dim_out, dim_x, dim_y, f_size,
                                resample=resample, name=name,
                                nonlinearity=nonlinearity, **layer_args)
            models.add_module(name, resblock)
            dim_x *= (2 if resample == 'up' else 1)
            dim_y *= (2 if resample == 'up' else 1)

        assert(dim_out == dim_h_)
        assert(dim_x == dim_x_)
        assert(dim_y == dim_y_)
        name = 'conv_({}/{})_final'.format(dim_out, dim_out_)
        finish_layer_2d(models, 'pre_' + name, dim_x, dim_y, dim_out,
                        nonlinearity=nonlinearity, **layer_args)
        models.add_module(name, nn.ConvTranspose2d(
            dim_out, dim_out_, f_size, 1, 1, bias=True))

        self.models = models

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        elif not nonlinearity:
            nonlinearity = None

        x = self.models(x)

        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)


class ResEncoder(BaseNet):
    def __init__(self, shape, dim_out=None, dim_h=64, dim_h_max=512, n_steps=3,
                 fully_connected_layers=None,
                 f_size=3, nonlinearity='ReLU', output_nonlinearity=None,
                 spectral_norm=False, **layer_args):
        super(ResEncoder, self).__init__(
            nonlinearity=nonlinearity, output_nonlinearity=output_nonlinearity)

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear

        fully_connected_layers = fully_connected_layers or []
        if isinstance(fully_connected_layers, int):
            fully_connected_layers = [fully_connected_layers]

        logger.debug('Input shape: {}'.format(shape))
        dim_x_, dim_y_, dim_in_ = shape
        dim_h_ = dim_h
        dim_out_ = dim_out

        dim_x = max(dim_x_ // 2**n_steps, 4)
        dim_y = max(dim_y_ // 2**n_steps, 4)
        dim_h = min(dim_h_ * 2**n_steps, dim_h_max)

        down_steps = int(math.log(dim_x_ / dim_x, 2))
        h_steps = int(math.log(dim_h / dim_h_, 2))
        rounds = calc_encoder_rounds(n_steps, down_steps, h_steps)

        name = 'conv_({}/{})_0'.format(dim_in_, dim_h_)
        self.models.add_module(name, Conv2d(
            dim_in_, dim_h_, f_size, 1, 1, bias=True))

        dim_out = dim_h_
        for i, (resample, mul) in enumerate(rounds):
            dim_in = dim_out
            dim_out *= mul
            name = 'resblock_({}/{}_{:s})_{}'.format(dim_in, dim_out,
                                                     resample, i + 1)
            resblock = ResBlock(dim_in, dim_out, dim_x_, dim_y_, f_size,
                                resample=resample, name=name,
                                nonlinearity=nonlinearity,
                                spectral_norm=spectral_norm, **layer_args)
            self.models.add_module(name, resblock)
            dim_x_ //= (2 if resample == 'down' else 1)
            dim_y_ //= (2 if resample == 'down' else 1)

        assert(dim_out == dim_h)
        assert(dim_x_ == dim_x)
        assert(dim_y_ == dim_y)
        final_depth = dim_out
        dim_out = dim_x * dim_y * dim_out
        name = 'reshape_{}x{}x{}to{}'.format(dim_x, dim_y, final_depth, dim_out)
        self.models.add_module(name, View(-1, dim_out))
        finish_layer_1d(self.models, 'post_' + name, dim_out,
                        nonlinearity=nonlinearity, **layer_args)

        dim_out = self.add_linear_layers(dim_out, fully_connected_layers,
                                         Linear=Linear, **layer_args)
        self.add_output_layer(dim_out, dim_out_, Linear=Linear)

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        elif not nonlinearity:
            nonlinearity = None

        x = self.models(x)

        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)
