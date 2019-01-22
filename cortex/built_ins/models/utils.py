'''Model misc utilities.

'''

import logging
import math

from sklearn import svm
import torch


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


resnet_encoder_args_ = dict(dim_h=64, dim_h_max=1024, batch_norm=True,
                            f_size=3, n_steps=3)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5,
                           pad=2, stride=2, min_dim=7)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def update_encoder_args(x_shape, model_type='convnet', encoder_args=None):
    encoder_args = encoder_args or {}
    if model_type == 'resnet':
        from cortex.built_ins.networks.resnets import ResEncoder as Encoder
        encoder_args_ = {k: v for k, v in resnet_encoder_args_.items()}
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


resnet_decoder_args_ = dict(dim_h=64, dim_h_max=1024, batch_norm=True,
                            f_size=3, n_steps=3)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4,
                           pad=1, stride=2, n_steps=2)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def update_decoder_args(x_shape, model_type='convnet', decoder_args=None):
    decoder_args = decoder_args or {}

    if model_type == 'resnet':
        from cortex.built_ins.networks.resnets import ResDecoder as Decoder
        decoder_args_ = {k: v for k, v in resnet_decoder_args_.items()}
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
    for p in target_net.parameters():
        p._requires_grad(False)
    for p in source_net.parameters():
        p._requires_grad(False)

    param_dict_src = dict(source_net.named_parameters())

    for p_name, p_target in target_net.named_parameters():
        p_source = param_dict_src[p_name]
        assert(p_source is not p_target)
        p_target.add_(p_source.sub(p_target).mul(1. - beta))
