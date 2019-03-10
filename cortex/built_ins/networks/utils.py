'''Utils for networks

'''

import functools
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def get_nonlinearity(nonlinearity=None, **kwargs):

    def get_from_nn(cls, **kwargs_):
        # TODO Expose hyperparameters of non-linearities later
        # This is adhoc to set neg_slope at 0.02
        if cls == nn.LeakyReLU:
            kwargs_.update(negative_slope=0.02)

        return cls(**kwargs_)

    if nonlinearity is None:
        return

    if callable(nonlinearity):
        if isinstance(nonlinearity, nn.Module):
            cls = type(nonlinearity)
            nonlinearity = get_from_nn(cls, **kwargs)

    elif hasattr(nn, nonlinearity):
        cls = getattr(nn, nonlinearity)
        nonlinearity = get_from_nn(cls, **kwargs)

    elif nonlinearity == 'tanh':
        nonlinearity = torch.tanh

    elif hasattr(nn.functional, nonlinearity):
        nonlinearity = getattr(nn.functional, nonlinearity)
        nonlinearity = functools.partial(nonlinearity, **kwargs)

    else:
        raise ValueError(
            "Could not resolve non linearity: {}".format(repr(nonlinearity)))

    return nonlinearity


def finish_layer_2d(models, name, dim_x, dim_y, dim_out,
                    dropout=False, inplace_dropout=False,
                    layer_norm=False, batch_norm=False,
                    nonlinearity=None, inplace_nonlin=False):
    if layer_norm and batch_norm:
        logger.warning('Ignoring batch_norm because layer_norm is True')

    if dropout:
        models.add_module(name + '_do', nn.Dropout2d(p=dropout,
                                                     inplace=inplace_dropout))

    if layer_norm:
        models.add_module(name + '_ln', nn.LayerNorm((dim_out, dim_x, dim_y)))
    elif batch_norm:
        models.add_module(name + '_bn', nn.BatchNorm2d(dim_out))

    if nonlinearity:
        inplace_nonlin = inplace_nonlin or layer_norm or batch_norm
        nonlinearity = get_nonlinearity(nonlinearity, inplace=inplace_nonlin)
        models.add_module(
            '{}_{}'.format(name, nonlinearity.__class__.__name__),
            nonlinearity)


def finish_layer_1d(models, name, dim_out,
                    dropout=False, inplace_dropout=False,
                    layer_norm=False, batch_norm=False,
                    nonlinearity=None, inplace_nonlin=False):
    if layer_norm and batch_norm:
        logger.warning('Ignoring batch_norm because layer_norm is True')

    if dropout:
        models.add_module(name + '_do', nn.Dropout(p=dropout,
                                                   inplace=inplace_dropout))

    if layer_norm:
        models.add_module(name + '_ln', nn.LayerNorm(dim_out))
    elif batch_norm:
        models.add_module(name + '_bn', nn.BatchNorm1d(dim_out))

    if nonlinearity:
        inplace_nonlin = inplace_nonlin or layer_norm or batch_norm
        nonlinearity = get_nonlinearity(nonlinearity, inplace=inplace_nonlin)
        models.add_module(
            '{}_{}'.format(name, nonlinearity.__class__.__name__),
            nonlinearity)


def apply_nonlinearity(x, nonlinearity, **nonlinearity_args):
    if nonlinearity:
        nonlinearity = get_nonlinearity(nonlinearity, **nonlinearity_args)
        #  if isinstance(nonlinearity, nn.PReLU):
        #      nonlinearity.to(x.device)
        #  try:
        x = nonlinearity(x)
        #  except RuntimeError:
        #      nonlinearity.to('cpu')
        #      x = nonlinearity(x, **nonlinearity_args)
    return x
