'''Handlers.

'''

from collections.abc import MutableMapping
import logging

import torch

from . import exp


logger = logging.getLogger('cortex.handlers')


class Handler(MutableMapping):
    _type = None
    _get_error_string = 'Attribute `{}` not found. Available: {}'

    def __init__(self, allow_overwrite=True, **kwargs):
        self._allow_overwrite = allow_overwrite
        self._locked = False
        self.__dict__.update(**kwargs)

    def _check_keyvalue(self, key, value):
        if self._type and not isinstance(value, self._type):
            raise TypeError('Invalid type ({}), expected {}.'
                            .format(type(value), self._type))

    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            d = dict((k, v) for k, v in self.__dict__.items()
                     if not k.startswith('_'))
            raise KeyError(self._get_error_string.format(key, tuple(d.keys())))

    def __delitem__(self, key):
        del self.__dict__[key]

    def __setitem__(self, key, value):
        self._check_keyvalue(key, value)

        if self._locked:
            raise KeyError('Handler is locked.')

        if not self._allow_overwrite and hasattr(self, key):
            raise KeyError('Overwriting keys not allowed.')
        self.__dict__[key] = value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super().__setattr__(key, value)

        self._check_keyvalue(key, value)

        if self._locked:
            raise KeyError('Handler is locked.')

        if not self._allow_overwrite and hasattr(self, key):
            raise KeyError('Overwriting keys not allowed.')
        super().__setattr__(key, value)

    def __iter__(self):
        d = dict((k, v) for k, v in self.__dict__.items()
                 if not k.startswith('_'))
        return iter(d)

    def __len__(self):
        d = dict((k, v) for k, v in self.__dict__.items()
                 if not k.startswith('_'))
        return len(d)

    def lock(self):
        self._locked = True

    def __str__(self):
        d = dict((k, v) for k, v in self.__dict__.items()
                 if not k.startswith('_'))
        return d.__str__()


def convert_nested_dict_to_handler(d, _class=Handler):  # XXX
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        d[k] = convert_nested_dict_to_handler(v)

    return _class(**d)


class AliasedHandler(Handler):
    def __init__(self, handler, aliases=None):
        self._aliases = aliases or {}
        self._handler = handler

    def __getattr__(self, item):
        if item.startswith('_'):
            try:
                return super().__getitem__(item)
            except KeyError:
                return getattr(self._handler, item)
        item = self._aliases.get(item, item)
        return self._handler.__getitem__(item)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super().__setattr__(key, value)

        if key in self._aliases.values():
            raise KeyError('Name clash. Key is a value in the set of aliases.')

        key = self._aliases.get(key, key)
        self._handler.__setattr__(key, value)

    def __getitem__(self, item):
        if item.startswith('_'):
            return super().__getitem__(item)
        item = self._aliases.get(item, item)
        return self._handler.__getitem__(item)

    def __setitem__(self, key, value):
        if key.startswith('_'):
            return super().__setitem__(key, value)

        if key in self._aliases.values():
            raise KeyError('Name clash. Key is a value in the set of aliases.')

        key = self._aliases.get(key, key)
        return self._handler.__setitem__(key, value)

    def __str__(self):
        r_aliases = dict((v, k) for k, v in self._aliases.items())
        d = dict((r_aliases.get(k, k), v)
                 for k, v in self._handler.__dict__.items()
                 if not k.startswith('_'))
        return d.__str__()

    def __len__(self):
        return len(self._handler)

    def __iter__(self):
        r_aliases = dict((v, k) for k, v in self._aliases.items())
        for k in self._handler:
            yield r_aliases.get(k, k)

    def __delitem__(self, key):
        key = self._aliases.get(key, key)
        self._handler.__delattr__(key)


def aliased(handler, aliases=None):
    return AliasedHandler(handler, aliases=aliases)


class PrefixedHandler(Handler):
    def __init__(self, handler, prefix=None):
        self._prefix = prefix or ''
        self._handler = handler

    def __getattr__(self, item):
        if item.startswith('_'):
            return super().__getattr__(item)

        item = self._prefix + '_' + item
        return self._handler.__getitem__(item)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super().__setattr__(key, value)

        key = self._prefix + '_' + key
        return self._handler.__setattr__(key, value)

    def __getitem__(self, item):
        if item.startswith('_'):
            return super().__getitem__(item)

        item = self._prefix + '_' + item
        return self._handler.__getitem__(item)

    def __setitem__(self, key, value):
        if key.startswith('_'):
            return super().__setitem__(key, value)

        key = self._prefix + '_' + key
        return self._handler.__setitem__(key, value)

    def __str__(self):
        d = dict((k[len(self._prefix) + 1:], v)
                 for k, v in self._handler.__dict__.items()
                 if not k.startswith('_') and k.startswith(self._prefix + '_'))
        return d.__str__()

    def __len__(self):
        d = dict((k[len(self._prefix) + 1:], v)
                 for k, v in self._handler.__dict__.items()
                 if not k.startswith('_') and k.startswith(self._prefix + '_'))
        return len(d)

    def __iter__(self):
        for item in self._handler:
            if item.startswith(self._prefix + '_'):
                yield item[len(self._prefix) + 1:]

    def __delitem__(self, key):
        key = self._prefix + '_' + key
        self._handler.__delattr__(key)


def prefixed(handler, prefix=None):
    return PrefixedHandler(handler, prefix=prefix)


class NetworkHandler(Handler):
    _type = torch.nn.Module
    _get_error_string = 'Network `{}` not found. You must register ' \
                        'it in a `build` method of some model (as an attr of its nets).' \
                        ' Found: {}'

    def reload(self, **kwargs):
        for key, network in kwargs.items():
            logger.info("Reloading '%s' network.", key)
            try:
                self[key]
                self.__dict__[key].load_state_dict(network.state_dict())
            except KeyError:
                raise

    def __setitem__(self, key, value):
        logger.debug("Registering '%s' network:\n%s", key, value)
        super().__setitem__(key, value)

    def __setattr__(self, key, value):
        if not key.startswith('_'):
            logger.debug("Registering '%s' network:\n%s", key, value)
        super().__setattr__(key, value)


ResultsHandler = Handler


class LossHandler(Handler):
    '''Simple dict-like container for losses
    '''

    _type = torch.Tensor
    _get_error_string = 'Loss `{}` not found. You must add it as a dict entry'

    def __init__(self, nets, *args, **kwargs):
        self._nets = nets
        super().__init__(*args, **kwargs)

    def _check_key(self, k):
        if k not in self._nets:
            raise AttributeError(
                'Keyword `{}` not in the model_handler. Found: {}.'.format(
                    k, tuple(self._nets.keys())))

        return True

    def _check_keyvalue(self, k, v):
        super()._check_keyvalue(k, v)
        if len(v.size()) > 0:
            raise ValueError(
                'Loss must be a scalar. Got {}'.format(v.size()))

        self._check_key(k)

        return True

    def __setitem__(self, key, value):
        self._check_keyvalue(key, value)

        if self._locked:
            raise KeyError('Handler is locked.')
        self.__dict__[key] = value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super().__setattr__(key, value)

        self._check_keyvalue(key, value)
        super().__setattr__(key, value)

    def __missing__(self):
        return torch.zeros([])

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            self._check_key(key)
            return self.__missing__()

    def __getattr__(self, key):
        return self.__getitem__(key)


class TunablesHandler(Handler):
    '''Simple dict-like container for tunable hyperparameters.'''

    _type = None
    _get_error_string = 'kwarg `{}` not declared. Available: {}'

    def __init__(self, gkwargs=None, **kwargs):
        self._gkwargs = gkwargs or dict()
        super().__init__(**kwargs)

    def share_global_kwargs(self, gkwargs):
        """Share global kwargs object to complete functions' kwargs."""
        self._gkwargs = gkwargs

    def __setitem__(self, key, value):
        if not key.startswith('_'):
            if key not in self._gkwargs:
                raise KeyError(self._get_error_string.format(
                    key, tuple(self._gkwargs.keys())))
            # Modify shared dictionary, so that changes take effect immediately
            self._gkwargs[key] = value
            # Modify experiment's state, so that changes are resumable
            exp.ARGS['model'][key] = value
        super().__setitem__(key, value)

    def __setattr__(self, key, value):
        if not key.startswith('_'):
            if key not in self._gkwargs:
                raise KeyError(self._get_error_string.format(
                    key, tuple(self._gkwargs.keys())))
            # Modify shared dictionary, so that changes take effect immediately
            self._gkwargs[key] = value
            # Modify experiment's state, so that changes are resumable
            exp.ARGS['model'][key] = value
        super().__setattr__(key, value)
