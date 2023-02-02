#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : environ_v2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/17/2019
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

r"""This file defines a global variable ``configs``, which can be used as a (nested) dictionary to store
configuration values. There are three modes of using this variable:

**Definition mode**: enabled by :func:`jacinle.def_configs`. In this mode, you can access the ``configs``
variable to define keys and their default values. For example::

    >>> from jacinle.config.environ_v2 import def_configs, configs
    >>> with def_configs():
    ...     configs.model.name = 'resnet18'

In this case, the key ``model.name`` will be defined with the default value ``'resnet18'``.

**Setting mode**: enabled by :func:`jacinle.set_configs`. In this mode, you can access the ``configs``
variable to set values. For example::

    >>> from jacinle.config.environ_v2 import set_configs, configs
    >>> with set_configs():
    ...     configs.model.name = 'resnet50'

In this case, the key ``model.name`` will be set to ``'resnet50'``.

**Reading mode**: this is the default mode. In this mode, you can access the ``configs`` variable to
read values. For example::

    >>> from jacinle.config.environ_v2 import configs
    >> print(configs.model.name)

Here is a more complete example::

    >>> from jacinle.config.environ_v2 import def_configs, set_configs, configs
    >>> with def_configs():
    ...     configs.model.name = 'resnet18'
    ...     configs.model.num_classes = 1000
    ...     configs.model.num_layers = 18

    >>> with set_configs():
    ...     configs.model.name = 'resnet50'
    ...     configs.model.num_layers = 50
    ...     configs.model.num_filters = 64

    >>> print(configs.model.name)  # 'resnet50'
    >>> print('Undefined keys:', configs.find_undefined_values())  # ['model.num_filters']

Note that, we have also provided a helper function ``configs.find_undefined_values`` to find all
undefined keys in the ``configs`` variables (i.e., those keys used in ``set_configs`` but not defined
in ``def_configs``). This can be used as a sanity check to make sure that all keys are defined.

Note that, a definition of a key in ``def_configs`` can be later than its usage in ``set_configs``.
This allows you to have a global configuration file that sets all keys, while the definition of each
key is in the corresponding module (data, model, etc.)

When a key has not been defined or set, reading it will raise an error.
"""

import functools
import contextlib
from jacinle.utils.printing import kvprint, kvformat
from jacinle.utils.meta import dict_deep_kv, run_once

__all__ = ['configs', 'def_configs', 'def_configs_func', 'set_configs', 'set_configs_func', 'StrictG']

ENABLE_CONFIG_AUTOINIT = False
ENABLE_DEF_CONFIG = False


@contextlib.contextmanager
def set_configs():
    """A context manager to enable configuration setting mode.
    See the module docstring :mod:`jacinle.configs.environ_v2` for more details."""
    global ENABLE_CONFIG_AUTOINIT
    assert not ENABLE_CONFIG_AUTOINIT
    ENABLE_CONFIG_AUTOINIT = True
    yield
    assert ENABLE_CONFIG_AUTOINIT
    ENABLE_CONFIG_AUTOINIT = False


@contextlib.contextmanager
def def_configs():
    """A context manager to enable configuration definition mode.
    See the module docstring :mod:`jacinle.configs.environ_v2` for more details."""
    global ENABLE_DEF_CONFIG
    with set_configs():
        assert not ENABLE_DEF_CONFIG
        ENABLE_DEF_CONFIG = True
        yield
        assert ENABLE_DEF_CONFIG
        ENABLE_DEF_CONFIG = False


def set_configs_func(func):
    """A decorator to enable configuration setting mode when calling a function.
    See the module docstring :mod:`jacinle.configs.environ_v2` for more details."""
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with set_configs():
            func(*args, **kwargs)
    return wrapped


def def_configs_func(func):
    """A decorator to enable configuration definition mode when calling a function.
    See the module docstring :mod:`jacinle.configs.environ_v2` for more details."""
    @functools.wraps(func)
    @run_once
    def wrapped(*args, **kwargs):
        with def_configs():
            func(*args, **kwargs)
    return wrapped


class StrictG(dict):
    """A strictly-managed dictionary that supports three-mode access (define, set, read)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['__defined_kvs__'] = dict()
        self['__defined_info__'] = dict()

    def __getattr__(self, k):
        """Get the value of a key. In the read mode, if the key has not been defined, raise an error.
        In the set/define mode, if the key has not been defined, define it with a nested ``StrictG``."""
        if k not in self:
            if ENABLE_CONFIG_AUTOINIT:
                self[k] = StrictG()
            else:
                raise AttributeError
        return self[k]

    def __setattr__(self, k, v):
        """Set the value of a key. It performs the following checks:

        - If the current mode is not set mode, raise an error.
        - If the key has not been defined, raise an error.
        """
        if not ENABLE_CONFIG_AUTOINIT:
            raise AttributeError('Cannot set value in the read mode.')

        if ENABLE_DEF_CONFIG:
            if k in self.__defined_kvs__ or (k in self and isinstance(self[k], StrictG)):
                raise AttributeError('Key "{}" has already been implicitly or explicitly defined.'.format(k))
            self.__defined_kvs__[k] = v
            self.setdefault(k, v)
            self.validate(k)
        else:
            self[k] = v
            self.validate(k)

    def def_(self, name, type=None, choices=None, default=None, help=None):
        """Define a key. If the key has already been defined, raise an error. This function is
        more flexible than ``__setattr__`` because it allows you to specify the type, choices, and
        default value of the key. It also allows you to specify a help message for the key."""
        if name in self.__defined_info__ or (name in self and isinstance(self[name], StrictG)):
            raise AttributeError('Key "{}" has already been implicitly or explicitly defined.'.format(name))

        self.__defined_info__[name] = {
            'type': type,
            'choices': choices,
            'help': help
        }
        if default is not None:
            self.__defined_kvs__[name] = default
            self.setdefault(name, default)
        if name in self:
            self.validate(name)

    def validate(self, name):
        """Validate the value of a key based on the type, choices, and default value specified."""
        if name in self.__defined_info__:
            info = self.__defined_info__[name]

            value = self[name]
            if info['type'] is not None:
                if not isinstance(value, info['type']):
                    raise TypeError('Key "{}" does not satisfy the type constraint: {}, got {}.'.format(name, info['type'], type(value)))
            if info['choices'] is not None:
                if value not in info['choices']:
                    raise TypeError('Key "{}" should be one of the: {}, got {}.'.format(name, info['choices'], value))

    def find_undefined_values(self, global_prefix: str = 'configs'):
        """Find all undefined keys in the dictionary. This function is used to check whether
        all keys have been defined.

        Args:
            global_prefix: the prefix of the global configuration dictionary when printing the
                undefined keys.
        """
        undefined = list()

        def dfs(d, prefix):
            for k, v in d.items():
                if isinstance(v, StrictG):
                    dfs(v, prefix + '.' + k)
                else:
                    if k not in d.__defined_kvs__ and not (k.startswith('__') and k.endswith('__')):
                        undefined.append(prefix + '.' + k)
        try:
            dfs(self, global_prefix)
        finally:
            del dfs

        return undefined

    def format(self, sep=': ', end='\n'):
        """Format the dictionary into a string."""
        return kvformat(dict(dict_deep_kv(self)), sep=sep, end=end)

    def print(self, sep=': ', end='\n', file=None):
        """Print the dictionary."""
        return kvprint(dict(dict_deep_kv(self)), sep=sep, end=end, file=file)


configs = StrictG()
"""The global configuration dictionary."""

