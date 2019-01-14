'''Builds arch

'''

import copy
import logging
import time

from . import data, exp, optimizer
from .parsing import parse_docstring, parse_inputs, parse_kwargs
from .handlers import (aliased, prefixed,
                       NetworkHandler, LossHandler,
                       ResultsHandler, TunablesHandler)
from .utils import (bad_values, update_dict_of_lists)
from .viz import VizHandler


__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.models')

MODEL_PLUGINS = {}


def register_model(plugin):
    '''

    Args:
        plugin: TODO

    Returns:
        TODO

    '''

    global MODEL_PLUGINS

    if plugin.__name__ in MODEL_PLUGINS:
        raise KeyError('{} already registered under the same name.'
                       .format(plugin.__name__))

    MODEL_PLUGINS[plugin.__name__] = plugin()


def get_model(model_name):
    try:
        return MODEL_PLUGINS[model_name]
    except KeyError:
        raise KeyError('Model {} not found. Available: {}'
                       .format(model_name, tuple(MODEL_PLUGINS.keys())))


class PluginType(type):
    def __new__(metacls, name, bases, attrs):
        cls = super(PluginType, metacls).__new__(metacls, name, bases, attrs)

        help = {}
        kwargs = {}
        args = set()

        for key in ['build', 'routine', 'visualize',
                    'train_step', 'autotune', 'eval_step']:
            if hasattr(cls, key):
                attr = getattr(cls, key)
                help_ = parse_docstring(attr)
                kwargs_ = parse_kwargs(attr)
                args_ = set(parse_inputs(attr))

                for k, v in help_.items():
                    if k in help and v != help[k]:
                        metacls._warn_inconsitent_help(key, k, v, kwargs[k])

                for k, v in kwargs_.items():
                    if k in kwargs and v != kwargs[k] and v is not None:
                        metacls._warn_inconsitent_kwargs(key, k, v, kwargs[k])

                help.update(**help_)

                for k, v in kwargs_.items():
                    if k not in kwargs or (k in kwargs and v is not None):
                        kwargs[k] = v
                args |= args_

        cls._help = help
        cls._kwargs = kwargs
        cls._args = args

        return cls

    def _warn_inconsitent_help(cls, k, v, v_):
        logger.warning('Inconsistent docstring found with argument {k}. '
                       'Using {v} instead of {v_}'.format(k=k, v=v, v_=v_))

    def _warn_inconsitent_kwargs(cls, k, v, v_):
        logger.warning('Inconsistent keyword defaults found with argument {k}. '
                       'Using {v} instead of {v_}'.format(k=k, v=v, v_=v_))


class ModelPluginBase(metaclass=PluginType):
    '''
    TODO
    '''

    _viz = VizHandler()
    _data = data.DATA_HANDLER
    _optimizers = optimizer.OPTIMIZERS

    _training_nets = dict()
    _all_nets = NetworkHandler(allow_overwrite=False)

    _all_losses = LossHandler(_all_nets)
    _all_results = ResultsHandler()

    _all_validation = ResultsHandler(allow_overwrite=False)
    _all_tunables = TunablesHandler()
    _all_epoch_results = ResultsHandler()
    _all_epoch_losses = ResultsHandler()
    _all_epoch_times = ResultsHandler()

    def __init__(self, contract=None):
        '''

        Args:
            contract: A dictionary of strings which specify naming w.r.t.
                the model that creates this model.
        '''

        self._contract = None
        self._train = False
        self._models = []
        self.name = self.__class__.__name__

        if contract:
            contract = self._check_contract(contract)
            self._accept_contract(contract)

        self._all_tunables.share_global_kwargs(self._kwargs)
        self._tunables = aliased(self._all_tunables,
                                 aliases=self.contract_kwargs)
        self._nets = aliased(self._all_nets, aliases=self.contract_nets)
        self._losses = aliased(self._all_losses, aliases=self.contract_nets)

        self._results = prefixed(self._all_results, prefix=self.name)
        self._epoch_results = prefixed(
            self._all_epoch_results, prefix=self.name)
        self._epoch_losses = prefixed(
            self._all_epoch_losses, prefix=self.name)
        self._epoch_times = self._all_epoch_times

        self.wrap_functions()

    def wrap_functions(self):
        self._wrap_routine()
        self.visualize = self._wrap(self.visualize)
        self.train_step = self._wrap_step(self.train_step)
        self.autotune = self._wrap(self.autotune)
        self.eval_step = self._wrap_step(self.eval_step, train=False)
        self.train_loop = self._wrap_loop(self.train_loop, train=True)
        self.eval_loop = self._wrap_loop(self.eval_loop, train=False)
        self.build = self._wrap(self.build)

    @classmethod
    def _reset_class(cls):
        '''Resets the static variables.

        '''
        cls._kwargs.clear()
        cls._help.clear()
        cls._training_nets.clear()

        cls._all_nets.clear()
        cls._all_losses.clear()
        cls._all_results.clear()

        cls._all_validation.clear()
        cls._all_tunables.clear()
        cls._all_epoch_results.clear()
        cls._all_epoch_losses.clear()
        cls._all_epoch_times.clear()

    def _reset_epoch(self):
        self._all_validation.clear()
        self._all_tunables.clear()
        self._all_epoch_results.clear()
        self._all_epoch_losses.clear()
        self._all_epoch_times.clear()

    def _get_id(self, fn):
        '''Gets a unique identifier for a function.

        Args:
            fn: a callable.

        Returns:
            An indetifier.

        '''
        return fn

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def contract_nets(self):
        return self._contract['nets'] if self._contract is not None else {}

    @property
    def contract_kwargs(self):
        return self._contract['kwargs'] if self._contract is not None else {}

    @property
    def contract_inputs(self):
        return self._contract['inputs'] if self._contract is not None else {}

    def inputs(self, *keys):
        '''Pulls inputs from the data.

        This uses the contract to pull the right key from the data.

        Args:
            keys: List of string variable names.

        Returns:
            Tensor variables.

        '''
        input_dict = self.contract_inputs

        inputs = []
        for k in keys:
            key = input_dict.get(k, k)
            inp = self.data[key]
            inputs.append(inp)

        if len(inputs) == 0:
            return None
        elif len(inputs) == 1:
            return inputs[0]
        else:
            return inputs

    @property
    def help(self):
        return self._help

    @property
    def results(self):
        return self._results

    @property
    def validation(self):
        return self._all_validation

    @property
    def tunables(self):
        return self._tunables

    @property
    def nets(self):
        return self._nets

    @property
    def losses(self):
        return self._losses

    @property
    def viz(self):
        return self._viz

    @property
    def data(self):
        return self._data

    def _set_train(self):
        self._train = True
        for m in self._models:
            m._set_train()

    def _set_eval(self):
        self._train = False
        for m in self._models:
            m._set_eval()

    def __setattr__(self, key, value):
        '''Sets an attribute for the model.

        Overriding is done to handle adding a ModelPlugin attribute to this
        object.

        '''
        if isinstance(value, ModelPluginBase):
            model = value

            # Translate aggregated model's kwargs to self's
            kwargs = dict((model.contract_kwargs.get(k, k), v)
                          for k, v in model.kwargs.items() if v is not None)
            help = dict((model.contract_kwargs.get(k, k), v)
                        for k, v in model.help.items())

            # Complete self's kwargs with aggregated model's
            for k, v in kwargs.items():
                if k not in self._kwargs or self._kwargs[k] is None:
                    self._kwargs[k] = copy.deepcopy(v)
            self._all_tunables.share_global_kwargs(self.kwargs)
            for k, v in help.items():
                if k not in self.help:
                    self.help[k] = help[k]
            # Make total kwargs known to the tree of models with self as root
            model._set_kwargs(self._kwargs)

            # Overwrite model's name by its usage
            model.name = key
            model._epoch_results = prefixed(
                model._all_epoch_results, prefix=model.name)
            model._epoch_losses = prefixed(
                self._all_epoch_losses, prefix=model.name)
            self._models.append(model)

        super().__setattr__(key, value)

    def _set_kwargs(self, kwargs):
        self._kwargs = kwargs
        for model in self._models:
            model._set_kwargs(kwargs)

    def _check_contract(self, contract):
        '''Checks the compatability of the contract.

        Checks the keys in the contract to make sure they correspond to inputs
        or hyperparameters of functions in this class.

        Args:
            contract: Dictionary contract.

        Returns:
            A cleaned up version of the contract.

        '''
        kwargs = contract.pop('kwargs', {})
        nets = contract.pop('nets', {})
        inputs = contract.pop('inputs', {})

        if len(contract) > 0:
            raise KeyError('Unknown keys in contract: {}'
                           .format(tuple(contract.keys())))

        for k, v in kwargs.items():
            if k not in self._kwargs:
                raise KeyError('Invalid contract: {} does not have any '
                               'arguments called {}'
                               .format(self.__class__.__name__, k))

            if not isinstance(v, str):
                raise TypeError('Contract values must be strings.')

        for k, v in inputs.items():
            if k not in self._args:
                raise KeyError('Invalid contract: {} does not have any '
                               'inputs called {}'
                               .format(self.__class__.__name__, k))

            if not isinstance(v, str):
                raise TypeError('Contract values must be strings.')

        return dict(inputs=inputs, kwargs=kwargs, nets=nets)

    def _accept_contract(self, contract):
        '''Accepts the contract.

        Args:
            contract: Dictionary contract.

        '''
        if self._contract is not None:
            raise ValueError('Cannot accept more than one contract.')

        self._contract = contract

    def _wrap(self, fn):
        '''Wraps methods to allow for auto inputs and kwargs.

        Args:
            fn: A callable.

        Returns:
            A wrapped version of the callable.

        '''

        def _fetch_kwargs(**kwargs_):
            kwarg_dict = self.contract_kwargs
            kwarg_keys = parse_kwargs(fn).keys()

            kwargs = dict()
            for k in kwarg_keys:
                key = kwarg_dict.get(k, k)
                try:
                    value = self.kwargs[key]
                except KeyError:
                    value = kwargs_.get(key)
                kwargs[k] = value

            return kwargs

        def _fetch_inputs():
            if self._contract is not None:
                input_dict = self._contract['inputs']
            else:
                input_dict = {}
            input_keys = parse_inputs(fn)

            inputs = []
            for k in input_keys:
                key = input_dict.get(k, k)
                if key == 'args':
                    continue
                value = self.data[key]
                inputs.append(value)
            return inputs

        def wrapped(*args, auto_input=False, **kwargs_):
            kwargs = _fetch_kwargs(**kwargs_)
            for k, v in kwargs_.items():
                if isinstance(v, dict) and (k in kwargs and
                                            isinstance(kwargs[k], dict)):
                    kwargs[k].update(**v)
                elif v is not None:
                    kwargs[k] = v
                elif v is None and k not in kwargs:
                    kwargs[k] = v
            if auto_input:
                args = _fetch_inputs()
            return fn(*args, **kwargs)

        return wrapped

    def _wrap_routine(self):
        '''Wraps the routine.

        Set to `requires_grad` for models that are trained with this routine.

        '''

        fn = self.routine
        fn = self._wrap(fn)

        def wrapped(*args, **kwargs):
            fid = self._get_id(fn)

            training_nets = []
            if fid not in self._training_nets:
                loss_contributions = LossHandler(self._all_nets)
                self._losses = aliased(loss_contributions,
                                       aliases=self.contract_nets)
                result_contributions = ResultsHandler()
                self._results = prefixed(result_contributions,
                                         prefix=self.name)
                fn(*args, **kwargs)
                training_nets = []
                for k in loss_contributions.keys():
                    training_nets.append(k)
                self._training_nets[fid] = training_nets
            else:
                training_nets = self._training_nets[fid]

            for k, net in self.nets.items():
                training = self._train and k in training_nets
                #  net.train()
                if training:
                    optimizer = self._optimizers.get(k)
                    if optimizer is not None:
                        optimizer.zero_grad()
                for p in net.parameters():
                    p.requires_grad_(training)

            # Create fresh handlers to isolate this routine's contributions
            loss_contributions = LossHandler(self._all_nets)
            self._losses = aliased(loss_contributions,
                                   aliases=self.contract_nets)
            result_contributions = ResultsHandler()
            self._results = prefixed(result_contributions, prefix=self.name)

            start = time.time()
            output = fn(*args, **kwargs)
            self._check_bad_values()
            end = time.time()

            # Append results with a prefixed key to epoch results list
            for k, v in result_contributions.items():
                self._all_results[k] = v
            update_dict_of_lists(self._epoch_results, **self._results)
            self._results = prefixed(self._all_results, prefix=self.name)

            # Append routine times with a prefixed key to epoch times list
            update_dict_of_lists(self._epoch_times,
                                 **{self.name: end - start})

            # Append loss contributions with a prefixed key to epoch losses list
            losses = dict()
            for k, v in loss_contributions.items():  # k here is unaliased name
                losses[k] = v.item()
                self._all_losses[k] += v
            # unaliased name will be prefixed to show contribution
            update_dict_of_lists(self._epoch_losses, **losses)
            self._losses = aliased(self._all_losses, aliases=self.contract_nets)

            return output

        self.routine = wrapped

    def _wrap_step(self, fn, train=True):
        '''Wraps the training or evaluation step.

        Args:
            fn: Callable step function.
            train (bool): For train or eval step.

        Returns:
            Wrapped version of the function.

        '''

        fn = self._wrap(fn)

        def wrapped(*args, _init=False, **kwargs):
            if train and not _init:
                self._set_train()
                for net in self.nets.values():
                    net.train()
            else:
                self._set_eval()
                for net in self.nets.values():
                    net.eval()

            output = fn(*args, **kwargs)
            self._all_losses.clear()
            self._all_results.clear()

            return output

        return wrapped

    def _wrap_loop(self, fn, train=True):
        '''Wraps a loop.

        Args:
            fn: Callable loop function.
            train (bool): For train or eval loop.

        Returns:
            Wrapped version of the function.

        '''

        data_mode = 'train' if train else 'test'

        if train:
            epoch_str = 'Training {} (epoch {}): '
        else:
            epoch_str = 'Evaluating {} (epoch {}): '

        def wrapped(epoch, data_mode=data_mode, use_pbar=True, **kwargs):
            self._reset_epoch()
            self.data.reset(data_mode, string=epoch_str.format(exp.NAME, epoch),
                            make_pbar=use_pbar)
            fn(**kwargs)

            results = self._all_epoch_results
            results['losses'] = dict(self._all_epoch_losses)
            results['times'] = dict(self._all_epoch_times)
            if train:
                results['validate'] = dict(self._all_validation)
                results['tunables'] = dict(self._all_tunables)

        return wrapped

    def _get_training_nets(self):
        '''Retrieves the training nets for the object.

        '''

        training_nets = []
        for v in self._training_nets.values():
            training_nets += v

        return training_nets

    def _check_bad_values(self):
        '''Check for bad numbers.

        This checks the results and the losses for nan or inf.

        '''

        bads = bad_values(self.results)
        if bads:
            print(
                'Bad values found in results (quitting): {} \n All:{}'.format(
                    bads, self.results))
            exit(0)

        bads = bad_values(self.losses)
        if bads:
            print(
                'Bad values found in losses (quitting): {} \n All:{}'.format(
                    bads, self.losses))
            exit(0)

    def reload_nets(self, nets_to_reload):
        if nets_to_reload:
            self.nets._handler.load(**nets_to_reload)
