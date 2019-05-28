'''Argument parsing module

'''

import argparse
import copy
import inspect
import logging
import re
import sys

from sphinxcontrib.napoleon import Config
from sphinxcontrib.napoleon.docstring import GoogleDocstring

from . import data, optimizer, train

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


def parse_kwargs(f):
    kwargs = {}
    sig = inspect.signature(f)
    for i, (sk, sv) in enumerate(sig.parameters.items()):

        if sk == 'self':
            pass
        elif sk == 'kwargs':
            pass
        elif sv.default == inspect.Parameter.empty:
            pass
        else:
            v = copy.deepcopy(sv.default)
            kwargs[sv.name] = v
    return kwargs


def parse_inputs(f):
    args = []
    sig = inspect.signature(f)
    for i, (sk, sv) in enumerate(sig.parameters.items()):

        if sk == 'self':
            pass
        elif sk == 'kwargs':
            pass
        elif sv.default == inspect.Parameter.empty:
            args.append(sv.name)
    return args


def parse_docstring(f):
    if f.__doc__ is None:
        f.__doc__ = 'TODO\n TODO'
    doc = inspect.cleandoc(f.__doc__)
    config = Config()
    google_doc = GoogleDocstring(doc, config)
    rst = str(google_doc)
    param_regex = r':param (?P<param>\w+): (?P<doc>.*)'
    m = re.findall(param_regex, rst)
    args_help = dict((k, v) for k, v in m)
    return args_help


def parse_header(f):
    if f.__doc__ is None:
        f.__doc__ = 'TODO\n TODO'
    doc = inspect.cleandoc(f.__doc__)
    config = Config()
    google_doc = GoogleDocstring(doc, config)
    rst = str(google_doc)
    lines = [l for l in rst.splitlines() if len(l) > 0]
    if len(lines) >= 2:
        return lines[:2]
    elif len(lines) == 1:
        return lines[0]
    else:
        return None, None


data_help = parse_docstring(data.setup)
data_args = parse_kwargs(data.setup)
train_help = parse_docstring(train.main_loop)
train_args = parse_kwargs(train.main_loop)
optimizer_help = parse_docstring(optimizer.setup)
optimizer_args = parse_kwargs(optimizer.setup)

default_args = dict(data=data_args, optimizer=optimizer_args, train=train_args)
default_help = dict(data=data_help, optimizer=optimizer_help, train=train_help)

logger = logging.getLogger('cortex.parsing')


def make_argument_parser() -> argparse.ArgumentParser:
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=50, width=100))
    parser.add_argument(
        '-o',
        '--out_path',
        default=None,
        help=('Output path directory. All model results will go'
              ' here. If a new directory, a new one will be '
              'created, as long as parent exists.'))
    parser.add_argument(
        '-n',
        '--name',
        default=None,
        help=('Name of the experiment. If given, base name of '
              'output directory will be `--name`. If not given,'
              ' name will be the base name of the `--out_path`'))
    parser.add_argument('-r', '--reload', type=str, default=None,
                        help=('Path to model to reload.'))
    parser.add_argument('-a', '--autoreload', default=False,
                        action='store_true')
    parser.add_argument('-R', '--networks_to_reload', type=str, nargs='+',
                        default=None)
    parser.add_argument('-L', '--load_networks',
                        type=str, default=None,
                        help=('Path to model to reload. Does not load args,'
                              ' info, etc'))
    parser.add_argument('-m', '--meta', type=str, default=None)
    parser.add_argument('-c', '--config', default=None,
                        help='Configuration yaml file.')
    #  'See `exps/` for examples'))  # TODO
    parser.add_argument('-k', '--clean', action='store_true', default=False,
                        help='Cleans the output directory. '
                             'This cannot be undone!')
    parser.add_argument('--verbosity', '-v', action='count', default=0,
                        help='Verbosity of the logging '
                             '(0-warning, 1-info, 2-debug, 3++-for root logger).')
    parser.add_argument('-d', '--device', type=str, default='0',
                        help="Enter a number to try declare the CUDA device to use,"
                             " else enter 'cpu' to use CPU.")
    parser.add_argument('--seed', type=str, default='randomness',
                        help="A string which will be used to seed "
                             "experiment's PRGs.")
    parser.add_argument('--test-seed', type=str,
                        help="A string which will be used to fix the seed "
                             "used in experiment's testing PRGs. If `None`, "
                             "then model testing happens with the same PRGs as training.")
    parser.add_argument('--fastdebug', type=int,
                        help="If not None, run `while True` loops for this many "
                             "in order to debug.")
    parser.add_argument('-V', '--viz', default=False, action='store_true',
                        help='Enable visualization.')

    return parser


class StoreDictKeyPair(argparse.Action):
    """Parses key value pairs from command line."""
    def __call__(self, parser, namespace, values, option_string=None):
        if '__' in values:
            raise ValueError('Private or protected values not allowed.')

        values_ = values

        try:
            # Puts quotes on things not currently in the Namespace
            while True:
                try:
                    eval(values)
                    break
                except NameError as e:
                    name = str(e).split(' ')[1][1:-1]
                    p = '(?<!\'){}(?!\')'.format(name)
                    values = re.sub(p, "'{}'".format(name), values)

            d = eval(values)
        except:
            d = str(values_)

        setattr(namespace, self.dest, d)


def str2bool(v):
    """Converts a string to boolean.
    E.g., yes -> True, no -> False.
    Args:
        v: String to convert.
    Returns:
        True or False
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _parse_model(model, subparser):
    global default_args
    kwargs = dict((k, v) for k, v in model.kwargs.items())
    model_defaults = model.defaults
    model_defaults_model = model.defaults.pop('model', {})
    update_args(model_defaults_model, kwargs)
    helps = model.help

    for k, v in kwargs.items():
        help = helps.get(k, None)
        _parse_kwargs(k, v, help, subparser)

    default_args = dict((k, v) for k, v in default_args.items())
    update_args(model_defaults, default_args)

    for key, args in default_args.items():
        _parse_defaults(key, args, subparser)


def _parse_defaults(key, args, subparser):
    """Parses the default values of a model for the command line.

    Args:
        key: Key of argument.
        args: values of argument.
        subparser: subparser to parse values to so values show up on command line..

    Returns:

    """
    for k, v in args.items():
        arg_str = '--' + key[0] + '.' + k
        help = default_help[key][k]
        dest = key + '.' + k

        if isinstance(v, dict):
            dstr = ','.join(
                ['{}={}'.format(str(vk), str(vv)) for vk, vv in v.items()])
            dstr = dstr.replace(' ', '')
            metavar = 'dict(<k1=v1>, ...)' + ' defaults=dict(' + dstr + ')'

            subparser.add_argument(
                arg_str,
                dest=dest,
                default=None,
                action=StoreDictKeyPair,
                help=help,
                metavar=metavar)
        elif isinstance(v, bool) and not v:
            action = 'store_true'
            dest = key + '.' + k
            subparser.add_argument(arg_str, dest=dest,
                                   action=action, default=False,
                                   help=help)
        elif isinstance(v, bool):
            type_ = type(v)
            metavar = '<' + type_.__name__ + \
                      '> (default=' + str(v) + ')'
            dest = key + '.' + k
            subparser.add_argument(
                arg_str,
                dest=dest,
                default=True,
                metavar=metavar,
                type=str2bool,
                help=help)
        else:
            type_ = type(v) if v is not None else str
            metavar = '<' + type_.__name__ + \
                      '> (default=' + str(v) + ')'
            subparser.add_argument(
                arg_str,
                dest=dest,
                default=v,
                metavar=metavar,
                type=type_,
                help=help)


def _parse_kwargs(k, v, help, subparser):
    arg_str = '--' + k
    choices = None

    if isinstance(v, dict):
        dstr = ','.join(
            ['{}={}'.format(str(vk), str(vv)) for vk, vv in v.items()])
        dstr = dstr.replace(' ', '')
        metavar = 'dict(<k1>=<v1>, ...)' + ' defaults=dict(' + dstr + ')'

        subparser.add_argument(
            arg_str,
            dest=k,
            default=v,
            action=StoreDictKeyPair,
            help=help,
            metavar=metavar)
    elif isinstance(v, bool) and not v:
        action = 'store_true'
        subparser.add_argument(
            arg_str, dest=k, action=action, default=False, help=help)
    elif isinstance(v, bool):
        type_ = type(v)
        metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
        subparser.add_argument(
            arg_str,
            dest=k,
            default=True,
            metavar=metavar,
            type=str2bool,
            help=help)
    else:
        type_ = type(v) if v is not None else str
        metavar = '<' + type_.__name__ + '> (default=' + str(v) + ')'
        subparser.add_argument(
            arg_str,
            dest=k,
            choices=choices,
            metavar=metavar,
            default=v,
            type=type_,
            help=help)


def parse_args(models, model=None):
    """Parse the command line arguments.

    Args:
        models: dictionary of models.

    Returns:
        dictionary of kwargs.

    """

    parser = make_argument_parser()

    if model is None:
        subparsers = parser.add_subparsers(
            title='Cortex',
            help='Select an architecture.',
            description='Cortex is a wrapper '
                        'around pytorch that makes training models '
                        'more convenient.',
            dest='command')

        subparsers.add_parser(
            'setup', help='Setup cortex configuration.',
            description='Initializes or updates the `.cortex.yml` file.')

        for k, model in models.items():
            model_help, model_description = parse_header(model)
            subparser = subparsers.add_parser(
                k,
                help=model_help,
                description=model_description,
                formatter_class=lambda prog: argparse.HelpFormatter(
                    prog, max_help_position=50, width=100))

            _parse_model(model, subparser)

    else:
        _parse_model(model, parser)

    command = sys.argv[1:]

    idx = []
    for i, c in enumerate(command):
        if c.startswith('-') and not(c.startswith('--')):
            idx.append(i)

    header = []

    # argparse is picky about ordering
    for i in idx[::-1]:
        a = None

        if i + 1 < len(command):
            a = command[i + 1]

        if a is not None and (a.startswith('-') or a.startswith('--')):
            a = None

        if a is not None:
            a = command.pop(i + 1)
            c = command.pop(i)
            header += [c, a]

        else:
            c = command.pop(i)
            header.append(c)

    command = header + command

    args = parser.parse_args(command)
    if not hasattr(args, 'command'):
        args.command = None

    return args


def update_args(kwargs, kwargs_to_update):
    """Updates kwargs from another set of kwargs.

    Does dictionary traversal.

    Args:
        kwargs: dictionary to update with.
        kwargs_to_update: dictionary to update.

    """
    def _update_args(from_kwargs, to_kwargs):
        for k, v in from_kwargs.items():
            if isinstance(v, dict) and k not in to_kwargs:
                to_kwargs[k] = v
            elif isinstance(v, dict) and isinstance(to_kwargs[k], dict):
                _update_args(v, to_kwargs[k])
            else:
                to_kwargs[k] = v

    _update_args(kwargs, kwargs_to_update)
