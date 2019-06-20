from torch import nn

from .spectral_norm import SNLinear


class ConditionalBatchNorm(nn.Module):
    def __init__(self, dim_in, num_features, class_bn,
                 spectral_norm=False, **bn_kwargs):
        super(ConditionalBatchNorm, self).__init__()
        assert(issubclass(class_bn, nn.modules.batchnorm._BatchNorm))
        bn_kwargs['affine'] = False
        self.num_features = num_features
        self.batch_norm = class_bn(num_features, **bn_kwargs)
        Linear = SNLinear if spectral_norm else nn.Linear
        self.gain = Linear(dim_in, num_features, bias=False)
        self.bias = Linear(dim_in, num_features, bias=False)

    def forward(self, x, y):
        shape = x.dim() * [1]
        shape[0] = -1
        shape[1] = self.num_features
        gain = self.gain(y).add(1).view(*shape)
        bias = self.bias(y).view(*shape)
        return bias.addcmul(1, self.batch_norm(x), gain)
