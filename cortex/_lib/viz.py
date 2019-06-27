"""
Visualization.
"""
from collections import defaultdict
import logging
import math
import os
import subprocess

import matplotlib as mpl
mpl.use('Agg')  # noqa E402
from matplotlib import pylab as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision as tv
import visdom

from . import data, exp
from .config import _yes_no
from .utils import convert_to_numpy, compute_tsne


__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


logger = logging.getLogger('cortex.viz')
config_font = None
visualizer = None
_options = dict(img=None, label_names=None, is_caption=False, is_attribute=False)

CHARS = ['_', '\n', ' ', '!', '"', '%', '&', "'", '(', ')', ',', '-', '.', '/',
         '0', '1', '2', '3', '4', '5', '8', '9', ':', ';', '=', '?', '\\', '`',
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '*', '*',
         '*']
CHAR_MAP = dict((i, CHARS[i]) for i in range(len(CHARS)))


def init(viz_config):
    global visualizer, config_font, viz_process
    if viz_config is not None and ('server' in viz_config.keys() or
                                   'port' in viz_config.keys()):
        server = viz_config.get('server', None)
        port = viz_config.get('port', 8097)
        visualizer = visdom.Visdom(server=server, port=port)
        if not visualizer.check_connection():
            if _yes_no("No Visdom server running on the configured address. "
                       "Do you want to start it?"):
                viz_bash_command = "python -m visdom.server"
                viz_process = subprocess.Popen(viz_bash_command.split())
                logger.info("Using visdom server at %s(:%d)", server, port)
            else:
                visualizer = None
    else:
        if _yes_no("Visdom configuration is not specified. Please run 'cortex setup' "
                   "to configure Visdom server. Do you want to continue with "
                   "the default address ? (localhost:8097)"):
            viz_bash_command = "python -m visdom.server"
            viz_process = subprocess.Popen(viz_bash_command.split())
            visualizer = visdom.Visdom()
            logger.info('Using local visdom server')
        else:
            visualizer = None
    config_font = viz_config.get('font')


def setup(img=None, label_names=None,
          is_caption=False, is_attribute=False, char_map=None):
    global _options, CHAR_MAP
    if img is not None:
        _options['img'] = img
    if label_names is not None:
        _options['label_names'] = label_names
    _options['is_caption'] = is_caption
    _options['is_attribute'] = is_attribute
    if is_caption and is_attribute:
        raise ValueError('Cannot be both attribute and caption')
    if char_map is not None:
        CHAR_MAP = char_map


class VizHandler():
    def __init__(self):
        self.viz_buffers = defaultdict(dict)
        self.output_dirs = exp.OUT_DIRS

    def clear(self):
        for x in self.viz_buffers.values():
            del x
        del self.viz_buffers
        self.viz_buffers = defaultdict(dict)

    def add_image(self, X, name=None, **opts):
        self._add_viz('image', X, name, **opts)

    def add_histogram(self, X, name=None, **opts):
        self._add_viz('histogram', X, name, **opts)

    def add_heatmap(self, X, name=None, **opts):
        self._add_viz('heatmap', X, name, **opts)

    def add_scatter(self, X, name=None, **opts):
        self._add_viz('scatter', X, name, **opts)

    def _add_viz(self, viz_type, X, name=None, **opts):
        if name is None:
            name = str(viz_type)
        name = name.replace(' ', '_')
        viz_map = self.viz_buffers[viz_type]
        if name in viz_map:
            logger.warning("'%s' has already been added to "
                           "visualization. Change `name` kwarg", name)
        if not (torch.is_tensor(X) or
                (isinstance(X, list) and all(torch.is_tensor(t) for t in X)) or
                (isinstance(X, dict) and all(torch.is_tensor(t) for t in X.values()))):
            raise TypeError('tensor or dict/list of tensors expected, got {}'.format(type(X)))
        viz_map[name] = (X, opts)

    def show(self, epoch):
        media_dir = self.output_dirs['image_dir']

        for viz_type, viz_buffer in self.viz_buffers.items():
            for i, (name, (X, opts)) in enumerate(viz_buffer.items()):
                if media_dir:
                    out = os.path.join(media_dir, '{}_{}.png'.format(name, epoch))
                    logger.debug("Saving %s: %s", viz_type, out)
                else:
                    out = None

                if opts.get('win') is None:
                    opts['win'] = viz_type + '_' + str(i)

                save_util = getattr(self, 'save_' + viz_type)
                save_util(X, out_file=out, **opts)
                del X

    @staticmethod
    def plot(epoch, init=False):
        '''Updates the plots for the results.

        Takes the last value from the summary and appends this to the visdom plot.

        '''
        if visualizer is None:
            return

        epoch -= 1

        def get_X_Y_legend(key, label1, v_train, label2, v_test):
            logger.debug("%s::\n%s : %s\n%s : %s",
                         key, label1, v_train, label2, v_test)

            min_e = max(0, epoch - 1)
            if init:
                min_e = 0

            if min_e == epoch:
                Y = [[v_train[0], v_train[0]]]
                X = [[-1, 0]]
            else:
                Y = [v_train[min_e:epoch + 1]]
                X = [range(min_e, epoch + 1)]
            legend = ['{} ({})'.format(key, label1)]

            if v_test is not None:
                if min_e == epoch:
                    Y += [[v_test[0], v_test[0]]]
                    X += [[-1, 0]]
                else:
                    Y += [v_test[min_e:epoch + 1]]
                    X += [range(min_e, epoch + 1)]
                legend += ['{} ({})'.format(key, label2)]

            return X, Y, legend

        def get_list_of_avg(list_of_stats):
            try:
                return list(map(lambda x: x[0], list_of_stats)) if list_of_stats else None
            except TypeError:
                return list_of_stats

        train_summary = exp.SUMMARY['train']
        train_keys = list(train_summary.keys())
        valid_summary = exp.SUMMARY['validate']
        #  valid_keys = list(valid_summary.keys())
        test_summary = exp.SUMMARY['test']
        test_keys = list(test_summary.keys())
        testing_plot_keys = list(set(test_keys) - set(train_keys))

        plot_schemes = [
            (train_keys, 'train', train_summary, 'test', test_summary),
            (testing_plot_keys, 'test', test_summary, 'validation', valid_summary)
            ]
        for keys, label1, summary1, label2, summary2 in plot_schemes:
            for k in keys:
                v_t1 = summary1[k]
                v_t2 = summary2.get(k, None)

                if isinstance(v_t1, dict):
                    Y = []
                    X = []
                    legend = []
                    for vk, vv_t1 in v_t1.items():
                        vv_t2 = v_t2.get(vk) if v_t2 is not None else None
                        X_, Y_, legend_ = get_X_Y_legend(vk,
                                                         label1, get_list_of_avg(vv_t1),
                                                         label2, get_list_of_avg(vv_t2))
                        Y += Y_
                        X += X_
                        legend += legend_
                else:
                    X, Y, legend = get_X_Y_legend(k,
                                                  label1, get_list_of_avg(v_t1),
                                                  label2, get_list_of_avg(v_t2))

                opts = dict(
                    xlabel='epochs',
                    legend=legend,
                    ylabel=k,
                    title=k)

                X = np.array(X).transpose()
                Y = np.array(Y).transpose()

                if init:
                    update = None
                else:
                    update = 'append'

                visualizer.line(
                    Y=Y,
                    X=X,
                    env=exp.NAME,
                    opts=opts,
                    win='line_{}'.format(k),
                    update=update)

    @staticmethod
    def save_text(labels, out_file=None, win='text', title=''):
        if visualizer is None and out_file is None:
            return

        labels = np.argmax(labels, axis=-1)
        char_map = _options['label_names']
        l_ = [''.join([char_map[j] for j in label]) for label in labels]

        if out_file is not None:
            with open(out_file, 'w') as f:
                for l__ in l_:
                    f.write(l__)

        if visualizer is not None:
            visualizer.text('\n'.join(l_), opts=dict(caption=title),
                            win=win, env=exp.NAME)

    @staticmethod
    def save_image(images, out_file=None, win='image', title='',
                   labels=None,
                   tile_shape=1, padding=None, normalize=False, range=None,
                   scale_each=False, pad_value=0, fill_value=255):
        if visualizer is None and out_file is None:
            return

        if isinstance(tile_shape, int):
            tile_shape = (tile_shape, ) * 2
        num_x, num_y = tile_shape

        if isinstance(fill_value, int):
            fill_value = (fill_value, ) * 3

        if isinstance(images, dict):
            labels = list(str(x) for x in images.keys())
            images = torch.stack(list(images.values()), dim=0)
        elif isinstance(images, list):
            images = torch.stack(images, dim=0)
        dim_x, dim_y = images.size()[-2:]
        images = images[:num_x * num_y]

        if labels is not None:
            nlabels = convert_to_numpy(labels)
            del labels
            labels = nlabels
            if isinstance(labels, (tuple, list)):
                labels = zip(*labels)
            if padding is None:
                if _options['is_caption']:
                    padding = 80
                elif _options['is_attribute']:
                    padding = 200
                elif _options['label_names'] is not None:
                    padding = 25
                else:
                    padding = 12

        if padding is None:
            padding = int(math.ceil(max(dim_x, dim_y) / 8))

        grid = tv.utils.make_grid(images, nrow=num_y, padding=padding,
                                  normalize=normalize, range=range,
                                  scale_each=scale_each, pad_value=pad_value)
        arr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(arr)

        if labels is not None:
            idr = ImageDraw.Draw(im)
            for i, label in enumerate(labels):
                x_ = (i // num_y) * (dim_x + padding)
                y_ = (i % num_y) * (dim_y + padding) + padding
                if _options['is_caption']:
                    l_ = ''.join([CHAR_MAP[j] for j in label
                                 if CHAR_MAP[j] != '\n']).strip()
                    if len(l_) == 0:
                        l_ = '<EMPTY>'
                    if len(l_) > 30:
                        l_ = '\n'.join(
                            [l_[x:x + 30] for x in range(0, len(l_), 30)])
                elif _options['is_attribute']:
                    attribs = [j for j, a in enumerate(label) if a == 1]
                    l_ = '\n'.join(_options['label_names'][a] for a in attribs)
                elif _options['label_names'] is not None:
                    l_ = _options['label_names'][label]
                    l_ = l_.replace('_', '\n')
                else:
                    l_ = str(label)
                idr.text((x_, y_), l_, fill=fill_value, font=config_font)

        if visualizer is not None:
            arr = np.array(im)
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)
            visualizer.image(arr, opts=dict(caption=title),
                             win=win, env=exp.NAME)

        if out_file is not None:
            im.save(out_file)

    @staticmethod
    def save_heatmap(X, out_file=None, win='heatmap', title='',
                     cmap='bwr', normalize=False, range=None, scale_each=False,
                     **image_kwargs):
        if visualizer is None and out_file is None:
            return

        if isinstance(X, dict):
            labels = list(str(x) for x in X.keys())
            image_kwargs['labels'] = labels
            X = torch.stack(list(X.values()), dim=0)
        elif isinstance(X, list):
            X = torch.stack(X, dim=0)
        dims = X.dim()
        X = X.cpu().numpy()

        mapping = mpl.cm.ScalarMappable(cmap=cmap)
        mapping.set_clim(range)
        if normalize and not scale_each:
            mapping.set_array(X)
            mapping.autoscale_None()
        hms = []
        for i in range(X.shape[0]):
            if normalize and scale_each:
                mapping.set_array(X[i])
                mapping.autoscale_None()
            heated = mapping.to_rgba(X[i], norm=normalize)
            if dims == 2:
                heated = torch.from_numpy(heated).permute(1, 0)[:3]
                heated = heated.view(3, 1, -1)
            elif dims == 3:
                heated = torch.from_numpy(heated).permute(2, 0, 1)[:3]
            hms.append(heated)

        VizHandler.save_image(hms, out_file, win, title,
                              normalize=False, **image_kwargs)

    @staticmethod
    def save_scatter(points, out_file=None, win='scatter', title='',
                     labels=None, markersize=4, range=(None, None),
                     **scatter_kwargs):
        if visualizer is None and out_file is None:
            return

        points = convert_to_numpy(points)
        if isinstance(points, dict):
            X = np.concatenate(list(points.values()), axis=0)
            Y = []
            for i, v in enumerate(points.values()):
                Y += [(i + 1 if labels is None else labels[i]), ] * v.shape[0]
            label_names = list(map(str, points.keys))
        else:
            if isinstance(points, list):
                X = np.stack(points, axis=0)
            if labels is not None:
                Y = np.asarray(labels)
            else:
                Y = None
            label_names = None

        if X.shape[1] == 1:
            raise NotImplementedError('1D-scatter not supported')
        elif X.shape[1] > 2:
            logger.debug('Scatter greater than 2D. Performing TSNE to 2D')
            X = compute_tsne(X)

        if out_file is not None:
            plt.clf()
            vmin, vmax = range
            plt.scatter(X[:, 0], X[:, 1], c=Y, s=markersize,
                        vmin=vmin, vmax=vmax, **scatter_kwargs)
            plt.legend(loc='upper right')
            plt.savefig(out_file)

        if visualizer is not None:
            visualizer.scatter(
                X=X,
                Y=Y,
                opts=dict(
                    title=title,
                    legend=label_names,
                    markersize=markersize),
                win=win, env=exp.NAME)

    @staticmethod
    def save_histogram(scores, out_file=None, win='histogram', title='',
                       labels=None,
                       bins='auto', range=None, alpha=None, **hist_kwargs):
        if visualizer is None and out_file is None:
            return

        scores = convert_to_numpy(scores)
        if isinstance(scores, dict):
            labels = list(str(x) for x in scores.keys())
            values = list(scores.values())
        elif not isinstance(scores, list):
            values = [scores]

        plt.clf()
        hists, bins = plt.hist(values, bins, range, label=labels,
                               alpha=(1. / len(scores) if alpha is None else alpha),
                               **hist_kwargs)
        if out_file is not None:
            plt.legend(loc='upper right')
            plt.savefig(out_file)

        if visualizer is not None:
            visualizer.stem(
                X=np.column_stack(hists),
                Y=np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]),
                opts=dict(legend=list(scores.keys()), title=title),
                win=win, env=exp.NAME)
