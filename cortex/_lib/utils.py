'''Utility methods

'''

import logging
import os

import numpy as np
import torch

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.util')

try:
    _, _columns = os.popen('stty size', 'r').read().split()
    _columns = int(_columns)
except ValueError:
    _columns = 1


def print_section(s):
    '''For printing sections to scripts nicely.
    Args:
        s (str): string of section
    '''
    h = s + ('-' * (_columns - len(s)))
    print(h)


def update_dict_of_lists(d_to_update, **d):
    '''Updates a dict of list with kwargs.

    Args:
        d_to_update (dict): dictionary of lists.
        **d: keyword arguments to append.

    '''
    for k, v in d.items():
        if isinstance(v, dict):
            if k not in d_to_update.keys():
                d_to_update[k] = {}
            update_dict_of_lists(d_to_update[k], **v)
        elif k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]


def update_nested_dict(to_d, from_d, strict=False):
    """Updates a nested dictionary with kwargs.

    Args:
        d_to_update (dict): nested dictionary.
        **d: keyword arguments to update.

    """
    for k, v in from_d.items():
        if isinstance(v, dict):
            if k not in to_d:
                to_d[k] = {}
            update_nested_dict(to_d[k], v)
        else:
            if strict and (k in to_d) and isinstance(to_d[k], dict):
                raise ValueError('Updating dict entry with non-dict.')
            to_d[k] = v


def bad_values(d):
    def is_nan_or_inf(t):
        t_ = torch.as_tensor(t)
        return torch.any(torch.isnan(t_) | torch.isinf(t_))

    failed = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ret = bad_values(v)
        elif isinstance(v, (list, tuple)):
            ret = {}
            for i, v_ in enumerate(v):
                ret_ = v_ if is_nan_or_inf(v_) else False
                if ret_:
                    ret[i] = ret_
        else:
            ret = v if is_nan_or_inf(v) else False

        if ret:
            failed[k] = ret

    return failed if failed else False


def convert_to_numpy(value):
    """NOTE: This method synchronizes with GPU."""
    if torch.is_tensor(value):
        o = value.detach().squeeze().cpu().numpy()
        if len(o.shape) == 0:
            return o.item()
        return o
    elif isinstance(value, list):
        return list(convert_to_numpy(vv) for vv in value)
    elif isinstance(value, tuple):
        return tuple(convert_to_numpy(vv) for vv in value)
    elif isinstance(value, dict):
        return dict((vk, convert_to_numpy(vv)) for vk, vv in value.items())
    return value


def detach_nested(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    elif isinstance(value, list):
        return list(detach_nested(vv) for vv in value)
    elif isinstance(value, tuple):
        return tuple(detach_nested(vv) for vv in value)
    elif isinstance(value, dict):
        return dict((vk, detach_nested(vv)) for vk, vv in value.items())
    return value


def compute_tsne(X, perplexity=40, n_iter=300, init='pca'):
    from sklearn.manifold import TSNE

    tsne = TSNE(2, perplexity=perplexity, n_iter=n_iter, init=init)
    points = X.tolist()
    return tsne.fit_transform(points)


def summarize_results(results, with_std=True):
    results_ = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_[k] = summarize_results(v, with_std=with_std)
        elif isinstance(v, (list, tuple)):
            if len(v) > 0:
                v = list(map(torch.as_tensor, v))
                v = torch.stack(v, dim=-1).squeeze()
                mv = v.mean(dim=-1)
                if with_std:
                    minv = v.min(dim=-1)[0]
                    maxv = v.max(dim=-1)[0]
                    stdv = v.std(dim=-1)
                    results_[k] = (mv, stdv, minv, maxv)
                else:
                    results_[k] = mv
        else:
            v = torch.as_tensor(v).squeeze_()
            results_[k] = (v, float('nan'), v, v) if with_std else v
    return results_
