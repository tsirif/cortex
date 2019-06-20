'''Utils for networks

'''

import functools
import logging

from torch import nn

from .conditional_batch_norm import ConditionalBatchNorm

logger = logging.getLogger(__name__)


def get_nonlinearity(nonlinearity=None, **kwargs):

    def get_from_nn(cls, **kwargs_):
        return cls(**kwargs_)

    if not nonlinearity:
        return

    if callable(nonlinearity):
        if isinstance(nonlinearity, nn.Module):
            cls = type(nonlinearity)
            nonlinearity = get_from_nn(cls, **kwargs)
        else:
            nonlinearity = functools.partial(nonlinearity, **kwargs)

    elif hasattr(nn, nonlinearity):
        cls = getattr(nn, nonlinearity)
        nonlinearity = get_from_nn(cls, **kwargs)

    elif hasattr(nn.functional, nonlinearity):
        nonlinearity = getattr(nn.functional, nonlinearity)
        nonlinearity = functools.partial(nonlinearity, **kwargs)

    else:
        raise ValueError(
            "Could not resolve non linearity: {}".format(repr(nonlinearity)))

    return nonlinearity


def finish_layer_2d(name, dim_x, dim_y, dim_out, models_output=None,
                    dropout=False, inplace_dropout=False,
                    layer_norm=False, batch_norm=False,
                    spectral_norm=False, dim_in_cond=0,
                    nonlinearity=None, inplace_nonlin=False, **nonlin_args):
    if layer_norm and batch_norm:
        logger.warning('Ignoring batch_norm because layer_norm is True')
    assert((dim_in_cond > 0 and batch_norm) or dim_in_cond == 0)

    models = models_output or nn.ModuleDict()

    if dropout:
        models[name + '_do'] = nn.Dropout2d(p=dropout, inplace=inplace_dropout)

    if layer_norm:
        models[name + '_ln'] = nn.LayerNorm((dim_out, dim_x, dim_y))
    elif batch_norm:
        if dim_in_cond > 0:
            models[name + '_cbn'] = ConditionalBatchNorm(dim_in_cond,
                                                         dim_out,
                                                         nn.BatchNorm2d,
                                                         spectral_norm=spectral_norm)
        else:
            models[name + '_bn'] = nn.BatchNorm2d(dim_out)

    if nonlinearity:
        inplace_nonlin = inplace_nonlin or layer_norm or batch_norm
        nonlin_args['inplace'] = inplace_nonlin
        nonlinearity = get_nonlinearity(nonlinearity, **nonlin_args)
        models['{}_{}'.format(name, nonlinearity.__class__.__name__)] = nonlinearity

    return models


def finish_layer_1d(name, dim_out, models_output=None,
                    dropout=False, inplace_dropout=False,
                    layer_norm=False, batch_norm=False,
                    spectral_norm=False, dim_in_cond=0,
                    nonlinearity=None, inplace_nonlin=False, **nonlin_args):
    if layer_norm and batch_norm:
        logger.warning('Ignoring batch_norm because layer_norm is True')
    assert((dim_in_cond > 0 and batch_norm) or dim_in_cond == 0)

    models = models_output or nn.ModuleDict()

    if dropout:
        models[name + '_do'] = nn.Dropout(p=dropout, inplace=inplace_dropout)

    if layer_norm:
        models[name + '_ln'] = nn.LayerNorm(dim_out)
    elif batch_norm:
        if dim_in_cond > 0:
            models[name + '_cbn'] = ConditionalBatchNorm(dim_in_cond,
                                                         dim_out,
                                                         nn.BatchNorm1d,
                                                         spectral_norm=spectral_norm)
        else:
            models[name + '_bn'] = nn.BatchNorm1d(dim_out)

    if nonlinearity:
        inplace_nonlin = inplace_nonlin or layer_norm or batch_norm
        nonlin_args['inplace'] = inplace_nonlin
        nonlinearity = get_nonlinearity(nonlinearity, **nonlin_args)
        models['{}_{}'.format(name, nonlinearity.__class__.__name__)] = nonlinearity

    return models


def apply_nonlinearity(x, nonlinearity, **nonlin_args):
    nonlinearity = get_nonlinearity(nonlinearity, **nonlin_args)
    if nonlinearity:
        # XXX take special care for PReLU later
        x = nonlinearity(x)
    return x


def apply_layer_dict(models, x, y=None, conditional_suffix=None):
    conditional_suffix = conditional_suffix or tuple()
    if not isinstance(conditional_suffix, tuple):
        conditional_suffix = (conditional_suffix,)
    conditional_suffix += ('_cbn',)  # for conditional batch norm

    for name, model in models.items():
        if name.endswith(conditional_suffix):
            assert(y is not None)
            x = model(x, y)
        else:
            x = model(x)

    return x
