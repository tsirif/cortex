'''Experiment module.

Used for saving, loading, summarizing, etc

'''

import logging
import os
from os import path
from shutil import copyfile, rmtree
import yaml

import torch

from .log_utils import set_file_logger
from .optimizer import get_optimizers_state
from .random import get_prgs_state

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.exp')

# Experiment info
NAME = 'X'
SUMMARY = {'train': {}, 'test': {}}
OUT_DIRS = {}
ARGS = dict(data=dict(), model=dict(), optimizer=dict(), train=dict())
INFO = {'name': NAME, 'epoch': 0, 'train_seed': 'randomness', 'test_seed': None}
DEVICE = torch.device('cpu')


def _file_string(prefix=''):
    if prefix == '':
        return NAME
    return '{}_{}'.format(NAME, prefix)


def configure_from_yaml(config_file=None):
    '''Loads arguments into a yaml file.

    '''
    global ARGS

    if config_file is not None:
        with open(config_file, 'r') as f:
            d = yaml.safe_load(f)
        logger.info('Loading config:\n%s', d)
        ARGS.get('model').update(**d.get('model', {}))
        ARGS.get('optimizer').update(**d.get('optimizer', {}))
        ARGS.get('train').update(**d.get('train', {}))
        ARGS.get('data').update(**d.get('data', {}))


def reload_model(model_to_reload):
    if not path.isfile(model_to_reload):
        raise ValueError('Cannot find {}'.format(model_to_reload))

    copyfile(model_to_reload, model_to_reload + '.bak')

    return torch.load(model_to_reload, map_location='cpu')


def save(model, prefix=''):
    '''Saves a model.

    Args:
        model: Model to save.
        prefix: Prefix for the save file.

    '''
    prefix = _file_string(prefix)
    binary_dir = OUT_DIRS.get('binary_dir', None)
    if binary_dir is None:
        return

    state = dict(
        nets=dict(model.nets),
        prgs=get_prgs_state(),
        optims=get_optimizers_state(),
        info=INFO,
        args=ARGS,
        out_dirs=OUT_DIRS,
        summary=SUMMARY
    )

    file_path = path.join(binary_dir, '{}.t7'.format(prefix))
    logger.debug('Saving checkpoint: %s', file_path)
    torch.save(state, file_path)


def setup_out_dir(out_path, global_out_path, name=None, clean=False):
    '''Sets up the output directory of an experiment.

    '''
    global OUT_DIRS

    if out_path is None:
        if name is None:
            raise ValueError('If `out_path` (-o) argument is not set, you '
                             'must set the `name` (-n)')
        out_path = global_out_path
        if out_path is None:
            raise ValueError('If `--out_path` (`-o`) argument is not set, you '
                             'must set both the name argument and configure '
                             'the out_path entry in `config.yaml`')

    if name is not None:
        out_path = path.join(out_path, name)

    if not path.isdir(out_path):
        logger.info('Creating out path: %s', out_path)
        os.makedirs(out_path, exist_ok=True)

    binary_dir = path.join(out_path, 'binaries')
    image_dir = path.join(out_path, 'images')

    if clean:
        logger.warning('Cleaning directory (cannot be undone)')
        if path.isdir(binary_dir):
            rmtree(binary_dir)
        if path.isdir(image_dir):
            rmtree(image_dir)

    if not path.isdir(binary_dir):
        os.mkdir(binary_dir)
    if not path.isdir(image_dir):
        os.mkdir(image_dir)

    logger.info('Setting out path to: %s', out_path)
    logpath = path.join(out_path, 'out.log')
    logger.info('Logging to: %s', logpath)
    set_file_logger(logpath)

    OUT_DIRS.update(binary_dir=binary_dir, image_dir=image_dir)


def setup_device(device):
    global DEVICE
    if torch.cuda.is_available() and device != 'cpu':
        device = int(device)
        if device < torch.cuda.device_count():
            logger.info('Using CUDA device %d', device)
            DEVICE = torch.device('cuda', device)
        else:
            logger.info('CUDA device %d doesn\'t exists. Using CPU', device)
    else:
        logger.info('Using CPU')
