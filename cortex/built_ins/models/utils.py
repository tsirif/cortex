'''Model misc utilities.

'''

import logging
import math

from sklearn import svm
import torch
from torch import nn


logger = logging.getLogger(__name__)


def log_sum_exp(x, axis=None):
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def cross_correlation(X, remove_diagonal=False):
    X_s = X / X.std(0)
    X_m = X_s - X_s.mean(0)
    b, dim = X_m.size()
    correlations = (X_m.unsqueeze(2).expand(b, dim, dim) *
                    X_m.unsqueeze(1).expand(b, dim, dim)).sum(0) / float(b)
    if remove_diagonal:
        Id = torch.eye(dim)
        Id = torch.autograd.Variable(Id.cuda(), requires_grad=False)
        correlations -= Id

    return correlations


def perform_svc(X, Y, clf=None):
    if clf is None:
        clf = svm.LinearSVC()
        clf.fit(X, Y)

    Y_hat = clf.predict(X)

    return clf, Y_hat


mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5,
                           pad=2, stride=2, min_dim=7)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def update_encoder_args(x_shape, model_type='convnet', encoder_args=None):
    encoder_args = encoder_args or {}
    if model_type == 'resnet':
        from cortex.built_ins.networks.resnets import Encoder
        encoder_args_ = {}
    elif model_type == 'convnet':
        from cortex.built_ins.networks.convnets import SimpleConvEncoder as Encoder
        encoder_args_ = {k: v for k, v in convnet_encoder_args_.items()}
    elif model_type == 'mnist':
        from cortex.built_ins.networks.convnets import SimpleConvEncoder as Encoder
        encoder_args_ = {k: v for k, v in mnist_encoder_args_.items()}
    elif model_type.split('.')[0] == 'tv':
        from cortex.built_ins.networks.torchvision import models
        model_attributes = model_type.split('.')
        if len(model_attributes) != 2:
            raise ValueError('`tvr` model type should be in form `tv.<MODEL>`')
        model_key = model_attributes[1]

        try:
            tv_model = getattr(models, model_key)
        except AttributeError:
            raise NotImplementedError(model_attributes[1])

        # TODO This lambda function is necessary because Encoder takes shape
        # and dim_out.
        Encoder = (lambda shape, dim_out=None, n_steps=None,
                   **kwargs: tv_model(num_classes=dim_out, **kwargs))
        encoder_args_ = {}
    elif model_type.split('.')[0] == 'tv-wrapper':
        from cortex.built_ins.networks import tv_models_wrapper as models
        model_attributes = model_type.split('.')

        if len(model_attributes) != 2:
            raise ValueError(
                '`tv-wrapper` model type should be in form'
                ' `tv-wrapper.<MODEL>`')
        model_key = model_attributes[1]

        try:
            Encoder = getattr(models, model_key)
        except AttributeError:
            raise NotImplementedError(model_attributes[1])
        encoder_args_ = {}
    else:
        raise NotImplementedError(model_type)

    encoder_args_.update(**encoder_args)
    log_x_shape_f, log_x_shape_i = math.modf(math.log(x_shape[0], 2))
    assert(log_x_shape_f == 0)
    log_x_shape_i = int(log_x_shape_i)
    n_steps = log_x_shape_i - 2
    encoder_args_['n_steps'] = max(n_steps, encoder_args_['n_steps'])

    return Encoder, encoder_args_


mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4,
                           pad=1, stride=2, n_steps=2)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def update_decoder_args(x_shape, model_type='convnet', decoder_args=None):
    decoder_args = decoder_args or {}
    if model_type == 'resnet':
        from cortex.built_ins.networks.resnets import Decoder
        decoder_args_ = {}
    elif model_type == 'convnet':
        from cortex.built_ins.networks.conv_decoders import (
            SimpleConvDecoder as Decoder)
        decoder_args_ = {k: v for k, v in convnet_decoder_args_.items()}
    elif model_type == 'mnist':
        from cortex.built_ins.networks.conv_decoders import (
            SimpleConvDecoder as Decoder)
        decoder_args_ = {k: v for k, v in mnist_decoder_args_.items()}
    else:
        raise NotImplementedError(model_type)

    decoder_args_.update(**decoder_args)
    log_x_shape_f, log_x_shape_i = math.modf(math.log(x_shape[0], 2))
    assert(log_x_shape_f == 0)
    log_x_shape_i = int(log_x_shape_i)
    n_steps = log_x_shape_i - 2
    decoder_args_['n_steps'] = max(n_steps, decoder_args_['n_steps'])

    return Decoder, decoder_args_


def to_one_hot(y, K):
    y_ = torch.unsqueeze(y, 1).long()

    one_hot = torch.zeros(y.size(0), K).cuda()
    one_hot.scatter_(1, y_.data.cuda(), 1)
    return torch.tensor(one_hot)


def update_average_model(target_net, source_net, beta):
    param_dict_src = dict(source_net.named_parameters())
    for p_name, p_target in target_net.named_parameters():
        p_source = param_dict_src[p_name]
        assert(p_source is not p_target)
        with torch.no_grad():
            p_target.add_(p_source.sub(p_target).mul(1. - beta))

    buffer_dict_src = dict(source_net.named_buffers())
    for b_name, b_target in target_net.named_buffers():
        b_source = buffer_dict_src[b_name]
        assert(b_source is not b_target)
        with torch.no_grad():
            # Batch Norm statistics are already averaged...
            b_target.copy_(b_source)


def parameters_init(m, nonlinearity=None, output_nonlinearity=None):
    """Helper function to initialize parameters in a network.

    Use `network.apply(parameters_init)`.

        Args:
            m: `torch.nn.Module` to initialize, part of a larger network
            nonlinearity: type applied throughout network except its output
            output_nonlinearity: type applied at the network's output

    """
    nonlinearity = nonlinearity or 'ReLU'
    nonlinearity = 'leaky_relu' if nonlinearity == 'LeakyReLU' else nonlinearity
    nonlinearity = 'relu' if nonlinearity == 'ReLU' else nonlinearity
    if 'final' in getattr(m, 'name', ''):  # 'final' denotes last layer
        nonlinearity = output_nonlinearity or 'linear'
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if nonlinearity in ('relu', 'leaky_relu'):
            # a == 0.02 is the same value as
            # `built_ins.networks.utils.get_nonlinearity`  XXX
            nn.init.kaiming_uniform_(m.weight, a=0.02, mode='fan_in',
                                     nonlinearity=nonlinearity)
        else:
            gain = nn.init.calculate_gain(nonlinearity)
            nn.init.xavier_uniform_(m.weight, gain=gain)
    if isinstance(m, nn.Embedding):
        nn.init.orthogonal_(m.weight)
